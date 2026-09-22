"""Offline five-round feedback-loop experiment for AgentFace.

The script deliberately avoids starting the web UI, FastAPI, or database. It
uses one fixed SDXL base model, FPEM automatic scoring, and a bounded batch
feedback update over the same 0--5 beautification controls used by AgentFace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
AGENTFACE_ROOT = EXPERIMENT_ROOT.parent
DEFAULT_CONFIG = EXPERIMENT_ROOT / "config" / "experiment_2.json"

PARAM_LABELS = {
    "skin_smoothing": "skin smoothing",
    "whitening": "skin brightening",
    "blemish_removal": "blemish removal",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_experiment_path(value: str) -> Path:
    return (EXPERIMENT_ROOT / value).resolve()


def clamp(value: float) -> float:
    return max(0.0, min(5.0, value))


def score_to_star(fpem_score: float) -> int:
    """Map FPEM's native approximately-1--5 score to integer stars once."""
    return max(1, min(5, math.floor(float(fpem_score) + 0.5)))


def fit_square(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(size / image.width, size / image.height)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
    )
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def level(value: float) -> str:
    if value <= 1.0:
        return "very subtle"
    if value <= 2.0:
        return "subtle"
    if value <= 3.0:
        return "moderate"
    if value <= 4.0:
        return "strong"
    return "very strong"


def build_prompt(base_prompt: str, params: dict[str, float], learnable: list[str]) -> str:
    clauses = [
        # Keep a readable semantic cue for SDXL while preserving the continuous
        # control value.  The former prompt used the cue alone, so small
        # feedback updates that did not cross a cue boundary produced an
        # identical text prompt (and therefore an identical seeded image).
        f"{level(params[name])} {PARAM_LABELS[name]} "
        f"(strength {params[name]:.2f})"
        for name in learnable
        if name in PARAM_LABELS
    ]
    return f"{base_prompt}, apply " + ", ".join(clauses) + ", keep the result natural"


def candidate_for_image(
    global_params: dict[str, float],
    filename: str,
    round_index: int,
    learnable: list[str],
    exploration_delta: float,
) -> dict[str, float]:
    """Create a small, deterministic image-level proposal around global preference.

    AgentFace updates preferences toward liked session parameters and away from
    disliked ones. The proposal is the session parameter for this offline
    batch experiment. It is deterministic from image name and round so the
    run is reproducible and does not use an extra random seed.
    """
    digest = hashlib.sha256(f"AgentFace-exp2|{round_index}|{filename}".encode("utf-8")).digest()
    candidate = dict(global_params)
    for offset, name in enumerate(learnable):
        signed = -1.0 if digest[offset] < 128 else 1.0
        magnitude = exploration_delta * (0.65 + 0.35 * digest[offset + len(learnable)] / 255.0)
        candidate[name] = round(clamp(global_params[name] + signed * magnitude), 4)
    return candidate


def preflight(experiment: dict[str, Any], limit: int | None) -> list[Path]:
    input_dir = resolve_experiment_path(experiment["input_dir"])
    manifest_path = resolve_experiment_path(experiment["input_manifest"])
    model_path = resolve_experiment_path(experiment["model_path"])
    scorer_python = Path(experiment["scorer_python"])
    scorer_script = resolve_experiment_path(experiment["scorer_script"])
    scorer_root = resolve_experiment_path(experiment["scorer_root"])
    scorer_checkpoint = resolve_experiment_path(experiment["scorer_checkpoint"])

    required = [
        manifest_path,
        model_path / "model_index.json",
        model_path / "unet" / "diffusion_pytorch_model.fp16.safetensors",
        model_path / "vae" / "diffusion_pytorch_model.fp16.safetensors",
        scorer_python,
        scorer_script,
        scorer_root / "nets" / "Clips.py",
        scorer_checkpoint,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing experiment component(s): " + "; ".join(missing))

    manifest = read_json(manifest_path)
    expected_count = int(experiment["image_count"])
    if int(manifest.get("selection_count", 0)) != expected_count:
        raise ValueError(f"Input manifest must contain {expected_count} images")
    files = sorted([path for path in input_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if len(files) != expected_count:
        raise ValueError(f"Input folder has {len(files)} images; expected {expected_count}")
    return files[:limit] if limit is not None else files


def load_pipeline(model_path: Path) -> StableDiffusionXLImg2ImgPipeline:
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
        use_safetensors=True,
    )
    if os.getenv("SDXL_CPU_OFFLOAD", "0") == "1":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()
    return pipe


def generate_round(
    experiment: dict[str, Any],
    inputs: list[Path],
    candidates: dict[str, dict[str, float]],
    output_dir: Path,
    overwrite: bool,
) -> None:
    model_path = resolve_experiment_path(experiment["model_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline(model_path)
    try:
        for position, image_path in enumerate(inputs, start=1):
            destination = output_dir / f"{image_path.stem}.png"
            if destination.exists() and not overwrite:
                print(f"Keeping {destination}", flush=True)
                continue
            with Image.open(image_path) as source:
                init_image = fit_square(source, int(experiment["resolution"]))
            prompt = build_prompt(
                experiment["base_prompt"],
                candidates[image_path.name],
                experiment["learnable_parameters"],
            )
            generator = torch.Generator(device="cuda").manual_seed(int(experiment["seed"]))
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=experiment["negative_prompt"],
                    image=init_image,
                    strength=float(experiment["strength"]),
                    num_inference_steps=int(experiment["steps"]),
                    guidance_scale=float(experiment["guidance_scale"]),
                    generator=generator,
                    height=int(experiment["resolution"]),
                    width=int(experiment["resolution"]),
                ).images[0]
            result.save(destination)
            print(f"Generated {position}/{len(inputs)} {destination.name}", flush=True)
    finally:
        del pipe
        torch.cuda.empty_cache()


def score_round(experiment: dict[str, Any], output_dir: Path, score_base: Path, overwrite: bool) -> dict[str, Any]:
    score_json = score_base.with_suffix(".json")
    if not score_json.exists() or overwrite:
        command = [
            str(resolve_experiment_path(experiment["scorer_python"])),
            str(resolve_experiment_path(experiment["scorer_script"])),
            "--root",
            str(resolve_experiment_path(experiment["scorer_root"])),
            "--checkpoint",
            str(resolve_experiment_path(experiment["scorer_checkpoint"])),
            "--input",
            str(output_dir),
            "--output",
            str(score_base),
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False, env=os.environ.copy())
        print(result.stdout, end="", flush=True)
        if result.returncode != 0:
            print(result.stderr, end="", flush=True)
            raise RuntimeError(f"FPEM scorer failed with exit code {result.returncode}")
    return read_json(score_json)


def feedback_text(star: int, params: dict[str, float]) -> str:
    warnings: list[str] = []
    if params["skin_smoothing"] >= 4.0:
        warnings.append("可能过度磨皮")
    if params["whitening"] >= 4.0:
        warnings.append("可能太白")
    if warnings:
        return " / ".join(warnings)
    if star >= 4:
        return "效果自然"
    if star <= 2:
        return "总体评分偏低"
    return "总体效果一般"


def summarize_and_update(
    experiment: dict[str, Any],
    scores: dict[str, Any],
    candidates: dict[str, dict[str, float]],
    global_params: dict[str, float],
) -> tuple[dict[str, Any], dict[str, float]]:
    rows = scores.get("rows", [])
    if not rows:
        raise RuntimeError("FPEM produced no score rows")
    expected = set(candidates)
    observed = {Path(row["image"]).with_suffix(".png").name for row in rows}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(f"Scorer/image mismatch; missing={missing[:3]} extra={extra[:3]}")

    details: list[dict[str, Any]] = []
    learnable = list(experiment["learnable_parameters"])
    for row in sorted(rows, key=lambda item: item["image"]):
        filename = Path(row["image"]).with_suffix(".png").name
        raw_score = float(row["fpem_score"])
        star = score_to_star(raw_score)
        params = candidates[filename]
        details.append(
            {
                "image": filename,
                "fpem_score": raw_score,
                "star": star,
                "feedback": feedback_text(star, params),
                "candidate_params": params,
            }
        )

    raw_scores = [item["fpem_score"] for item in details]
    stars = [item["star"] for item in details]
    feedback_counts = Counter(item["feedback"] for item in details)
    update_config = experiment["feedback_update"]
    if update_config["method"] != "batch_directional_fpem":
        raise ValueError(f"Unsupported feedback update: {update_config['method']}")

    # A batch has one hundred images. Applying the single-session update 100
    # times in sequence turns small score noise into a runaway parameter jump.
    # Instead, compare the mean FPEM score of small higher/lower proposals for
    # each parameter, then take one trust-region-limited step per round.
    updated = dict(global_params)
    parameter_feedback: dict[str, dict[str, float]] = {}
    score_scale = float(update_config["score_scale"])
    max_step = float(update_config["max_step"])
    bounds = update_config["bounds"]
    for key in learnable:
        higher = [item["fpem_score"] for item in details if item["candidate_params"][key] > global_params[key]]
        lower = [item["fpem_score"] for item in details if item["candidate_params"][key] < global_params[key]]
        if not higher or not lower:
            raise RuntimeError(f"Round has no balanced proposals for {key}")
        higher_mean = sum(higher) / len(higher)
        lower_mean = sum(lower) / len(lower)
        signal = higher_mean - lower_mean
        applied_step = max_step * math.tanh(signal / score_scale)
        lower_bound, upper_bound = (float(value) for value in bounds[key])
        updated[key] = round(max(lower_bound, min(upper_bound, global_params[key] + applied_step)), 4)
        parameter_feedback[key] = {
            "higher_candidate_mean_fpem": round(higher_mean, 6),
            "lower_candidate_mean_fpem": round(lower_mean, 6),
            "signal": round(signal, 6),
            "applied_step": round(applied_step, 6),
        }

    summary = {
        "count": len(details),
        "mean_fpem_score": round(sum(raw_scores) / len(raw_scores), 6),
        "min_fpem_score": round(min(raw_scores), 6),
        "max_fpem_score": round(max(raw_scores), 6),
        "mean_star": round(sum(stars) / len(stars), 6),
        "star_counts": {str(key): value for key, value in sorted(Counter(stars).items())},
        "feedback_counts": dict(sorted(feedback_counts.items())),
        "parameter_feedback": parameter_feedback,
        "details": details,
    }
    return summary, updated


def run(experiment: dict[str, Any], run_name: str, rounds: int, limit: int | None, overwrite: bool) -> None:
    inputs = preflight(experiment, limit)
    state_dir = EXPERIMENT_ROOT / "state" / run_name
    output_root = EXPERIMENT_ROOT / "outputs" / run_name
    score_root = EXPERIMENT_ROOT / "scores" / run_name
    state_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    score_root.mkdir(parents=True, exist_ok=True)

    run_contract = {
        "run_name": run_name,
        "input_count": len(inputs),
        "model_id": experiment["model_id"],
        "fixed_settings": {
            key: experiment[key]
            for key in ("resolution", "steps", "guidance_scale", "strength", "seed", "base_prompt", "negative_prompt")
        },
        "learnable_parameters": experiment["learnable_parameters"],
        "star_mapping": experiment["star_mapping"],
    }
    write_json(state_dir / "run_contract.json", run_contract)
    global_params = {key: float(value) for key, value in experiment["default_beautify_params"].items()}

    for round_index in range(rounds):
        round_state_path = state_dir / f"round_{round_index:02d}.json"
        if round_state_path.exists() and not overwrite:
            previous = read_json(round_state_path)
            global_params = {key: float(value) for key, value in previous["global_params_after"].items()}
            print(f"Reusing completed round {round_index}", flush=True)
            continue

        candidates_path = state_dir / f"round_{round_index:02d}_candidates.json"
        if candidates_path.exists() and not overwrite:
            candidates = read_json(candidates_path)["candidates"]
        else:
            candidates = {
                image_path.name: candidate_for_image(
                    global_params,
                    image_path.name,
                    0,
                    experiment["learnable_parameters"],
                    float(experiment["exploration_delta"]),
                )
                for image_path in inputs
            }
            write_json(
                candidates_path,
                {
                    "round": round_index,
                    "global_params_before": global_params,
                    "candidates": candidates,
                },
            )

        output_dir = output_root / f"round_{round_index:02d}"
        print(f"ROUND {round_index}: generating {len(inputs)} images", flush=True)
        generate_round(experiment, inputs, candidates, output_dir, overwrite)
        score_base = score_root / f"round_{round_index:02d}"
        print(f"ROUND {round_index}: scoring {len(inputs)} images", flush=True)
        scores = score_round(experiment, output_dir, score_base, overwrite)
        summary, global_params_after = summarize_and_update(experiment, scores, candidates, global_params)
        round_state = {
            "round": round_index,
            "global_params_before": global_params,
            "global_params_after": global_params_after,
            "summary": summary,
        }
        write_json(round_state_path, round_state)
        print(
            "ROUND_SUMMARY "
            + json.dumps(
                {
                    "round": round_index,
                    "mean_fpem_score": summary["mean_fpem_score"],
                    "mean_star": summary["mean_star"],
                    "global_params_after": global_params_after,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        global_params = global_params_after


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default="official")
    parser.add_argument("--rounds", type=int, help="override configured round count")
    parser.add_argument("--limit", type=int, help="run only the first N inputs; for smoke tests")
    parser.add_argument("--overwrite", action="store_true", help="regenerate existing outputs in this run name")
    parser.add_argument("--check", action="store_true", help="validate files only; do not run inference")
    args = parser.parse_args()

    config = read_json(args.config.resolve())
    experiment = config["experiment"]
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    rounds = args.rounds if args.rounds is not None else int(experiment["rounds"])
    if rounds <= 0 or rounds > int(experiment["rounds"]):
        raise ValueError("--rounds must be between 1 and the configured round count")
    inputs = preflight(experiment, args.limit)
    print(f"PRECHECK ready: {len(inputs)} input(s), model={experiment['model_id']}", flush=True)
    if not args.check:
        run(experiment, args.run_name, rounds, args.limit, args.overwrite)


if __name__ == "__main__":
    main()
