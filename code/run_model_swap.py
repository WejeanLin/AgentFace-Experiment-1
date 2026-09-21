"""Offline model-swap runner for AgentFace experiment 1.

This script deliberately does not start FastAPI or the web UI.  It applies one
fixed SDXL Img2Img setup to the same input images and changes only the model
directory.  Run ``python run_model_swap.py --check`` before inference.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "experiment_1.json"


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def has_any_file(directory: Path, names: tuple[str, ...]) -> bool:
    return any((directory / name).is_file() for name in names)


def model_problems(
    model_path: Path,
    single_file: Path | None = None,
    expected_size: int | None = None,
) -> list[str]:
    """Return missing components needed by SDXL Img2Img before loading weights."""
    if single_file is not None:
        if not single_file.is_file():
            return ["single-file checkpoint"]
        if expected_size is not None and single_file.stat().st_size != expected_size:
            return [
                f"single-file checkpoint has unexpected size "
                f"({single_file.stat().st_size} bytes)"
            ]
        return []

    checks = {
        "model_index.json": ("model_index.json",),
        "UNet": (
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
            "unet/diffusion_pytorch_model.safetensors.index.json",
        ),
        "VAE": (
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors.index.json",
        ),
        "text encoder": (
            "text_encoder/model.fp16.safetensors",
            "text_encoder/model.safetensors",
            "text_encoder/model.safetensors.index.json",
        ),
        "text encoder 2": (
            "text_encoder_2/model.fp16.safetensors",
            "text_encoder_2/model.safetensors",
            "text_encoder_2/model.safetensors.index.json",
        ),
        "tokenizer": ("tokenizer/tokenizer_config.json",),
        "tokenizer 2": ("tokenizer_2/tokenizer_config.json",),
        "scheduler": ("scheduler/scheduler_config.json",),
    }
    problems = [label for label, names in checks.items() if not has_any_file(model_path, names)]

    # A partly interrupted Hugging Face transfer can leave a file in place
    # while its contents are far too small for an SDXL UNet.  Catch that case
    # before the much less clear error raised by diffusers at load time.
    unet = model_path / "unet" / "diffusion_pytorch_model.fp16.safetensors"
    if not unet.is_file():
        unet = model_path / "unet" / "diffusion_pytorch_model.safetensors"
    if unet.is_file() and unet.stat().st_size < 1_000_000_000:
        problems.append(f"UNet file is unexpectedly small ({unet.stat().st_size // 1_000_000} MB)")
    return problems


def print_check(config: dict) -> bool:
    root = ROOT
    all_ready = True
    for model in config["models"]:
        path = resolve_path(root, model["path"])
        single_file = (
            resolve_path(root, model["single_file"])
            if model.get("single_file")
            else None
        )
        problems = model_problems(path, single_file, model.get("expected_size"))
        declared_status = model["status"]
        usable = declared_status == "ready" and not problems
        state = "READY" if usable else "NOT READY"
        if problems:
            detail = "missing or invalid: " + ", ".join(problems)
        elif declared_status != "ready":
            detail = model.get("status_reason", f"registry status is {declared_status}")
        else:
            detail = "complete"
        print(f"[{state}] {model['id']}: {path} ({detail})")
        all_ready = all_ready and usable
    return all_ready


def fit_square(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(size / image.width, size / image.height)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
    )
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def load_pipe(
    model_path: Path,
    configured_variant: str | None,
    single_file: Path | None,
    config_path: Path | None,
) -> StableDiffusionXLImg2ImgPipeline:
    kwargs = {
        "torch_dtype": torch.float16,
        "local_files_only": True,
        "use_safetensors": True,
    }
    if single_file is not None:
        if config_path is None:
            raise ValueError("A local SDXL config directory is required for a single-file checkpoint.")
        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            str(single_file), config=str(config_path), **kwargs
        )
    else:
        variant = configured_variant
        if not variant and (model_path / "unet" / "diffusion_pytorch_model.fp16.safetensors").is_file():
            variant = "fp16"
        if variant:
            kwargs["variant"] = variant
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_path, **kwargs)
    if os.getenv("SDXL_CPU_OFFLOAD", "0") == "1":
        # Keep inactive model components in CPU memory.  This preserves all
        # image-generation settings while allowing the experiment to coexist
        # with other jobs on a shared 24 GB GPU.
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()
    return pipe


def run(config: dict, selected_models: set[str] | None, overwrite: bool) -> None:
    experiment = config["experiment"]
    input_dir = resolve_path(ROOT, experiment["input_dir"])
    output_root = resolve_path(ROOT, experiment["output_dir"])
    image_paths = [input_dir / name for name in experiment["images"]]
    missing_images = [str(path) for path in image_paths if not path.is_file()]
    if missing_images:
        raise FileNotFoundError("Missing input images: " + ", ".join(missing_images))

    models = [m for m in config["models"] if selected_models is None or m["id"] in selected_models]
    if not models:
        raise ValueError("No model matches --model.")

    for model in models:
        model_path = resolve_path(ROOT, model["path"])
        single_file = (
            resolve_path(ROOT, model["single_file"])
            if model.get("single_file")
            else None
        )
        config_path = (
            resolve_path(ROOT, model["config_path"])
            if model.get("config_path")
            else None
        )
        problems = model_problems(model_path, single_file, model.get("expected_size"))
        if model["status"] != "ready" or problems:
            detail = ", ".join(problems) if problems else f"registry status is {model['status']}"
            raise RuntimeError(f"{model['id']} is not ready: {detail}")

        print(f"Loading {model['id']} from {model_path}", flush=True)
        pipe = load_pipe(model_path, model.get("variant"), single_file, config_path)
        try:
            for image_path in image_paths:
                destination = output_root / image_path.stem / f"{model['id']}.png"
                if destination.is_file() and not overwrite:
                    print(f"Keeping existing {destination}", flush=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                init_image = fit_square(Image.open(image_path), int(experiment["resolution"]))
                generator = torch.Generator(device="cuda").manual_seed(int(experiment["seed"]))
                with torch.inference_mode():
                    result = pipe(
                        prompt=experiment["prompt"],
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
                print(f"Saved {destination}", flush=True)
        finally:
            del pipe
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true", help="only verify local files; do not load a model")
    parser.add_argument("--model", action="append", help="model id to run; repeat for more than one")
    parser.add_argument("--overwrite", action="store_true", help="replace an already generated output")
    args = parser.parse_args()

    config = read_config(args.config.resolve())
    if args.check:
        print_check(config)
        return
    run(config, set(args.model) if args.model else None, args.overwrite)


if __name__ == "__main__":
    main()
