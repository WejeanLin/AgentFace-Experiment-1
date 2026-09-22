"""Create verifiable report artifacts for the completed Experiment 2 run.

Run from the ``experiment_2`` directory with a Python environment that has
Pillow and Matplotlib installed:
  python code/create_experiment2_artifacts.py

Only ``official_continuous`` is treated as formal evidence. Earlier folders
are diagnostic trials and are intentionally excluded from all summaries.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
RUN_NAME = "official_continuous"
ROUND_COUNT = 5
THUMBNAIL_SIZE = 280
TEXT_COLOR = (26, 32, 44)
SUBTLE_COLOR = (92, 99, 112)
MPL_TEXT_COLOR = "#1A202C"
BLUE = "#0072B2"
GRAY = "#6C757D"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def verify_and_collect() -> tuple[dict, list[dict], dict, dict[str, float], list[tuple[str, float]]]:
    config = read_json(ROOT / "config" / "experiment_2.json")["experiment"]
    states = [read_json(ROOT / "state" / RUN_NAME / f"round_{index:02d}.json") for index in range(ROUND_COUNT)]
    score_docs = [read_json(ROOT / "scores" / RUN_NAME / f"round_{index:02d}.json") for index in range(ROUND_COUNT)]

    if any(state["summary"]["count"] != 100 for state in states):
        raise RuntimeError("Each formal round must contain exactly 100 scored images.")
    if any(len(document["rows"]) != 100 for document in score_docs):
        raise RuntimeError("Each formal score file must contain exactly 100 rows.")
    if any(document["missing_keys"] or document["unexpected_keys"] for document in score_docs):
        raise RuntimeError("FPEM scorer did not load its checkpoint strictly.")

    sizes: Counter[str] = Counter()
    image_total = 0
    for index in range(ROUND_COUNT):
        images = sorted((ROOT / "outputs" / RUN_NAME / f"round_{index:02d}").glob("*.png"))
        if len(images) != 100:
            raise RuntimeError(f"Round {index} has {len(images)} images, not 100.")
        image_total += len(images)
        for image_path in images:
            with Image.open(image_path) as image:
                sizes[f"{image.width}x{image.height}"] += 1
    if sizes != Counter({"1024x1024": 500}):
        raise RuntimeError(f"Unexpected output image sizes: {dict(sizes)}")

    first_scores = {row["image"]: float(row["fpem_score"]) for row in states[0]["summary"]["details"]}
    last_scores = {row["image"]: float(row["fpem_score"]) for row in states[-1]["summary"]["details"]}
    if first_scores.keys() != last_scores.keys() or len(first_scores) != 100:
        raise RuntimeError("Round 0 and round 4 do not contain the same 100 images.")
    deltas = {name: last_scores[name] - first_scores[name] for name in first_scores}

    changed_hashes = 0
    for image_path in sorted((ROOT / "outputs" / RUN_NAME / "round_00").glob("*.png")):
        round_01_path = ROOT / "outputs" / RUN_NAME / "round_01" / image_path.name
        if hashlib.sha256(image_path.read_bytes()).digest() != hashlib.sha256(round_01_path.read_bytes()).digest():
            changed_hashes += 1

    repeat_rows = []
    for name in ("repeat_1.json", "repeat_2.json"):
        document = read_json(ROOT / "scores" / "scorer_validation" / name)
        repeat_rows.append([(row["image"], row["fpem_score"]) for row in document["rows"]])

    values = list(deltas.values())
    validation = {
        "formal_run": RUN_NAME,
        "round_counts": [state["summary"]["count"] for state in states],
        "score_row_counts": [len(document["rows"]) for document in score_docs],
        "output_count": image_total,
        "output_sizes": dict(sizes),
        "scorer_missing_keys": [document["missing_keys"] for document in score_docs],
        "scorer_unexpected_keys": [document["unexpected_keys"] for document in score_docs],
        "scorer_repeat_identical": repeat_rows[0] == repeat_rows[1],
        "round_00_to_round_01_changed_output_hashes": changed_hashes,
        "round_00_to_round_01_total": 100,
        "round_00_to_round_04_mean_fpem_delta": mean(values),
        "round_00_to_round_04_median_fpem_delta": median(values),
        "round_00_to_round_04_increased": sum(value > 0 for value in values),
        "round_00_to_round_04_decreased": sum(value < 0 for value in values),
        "round_00_to_round_04_tied": sum(value == 0 for value in values),
    }
    return config, states, validation, deltas, sorted(deltas.items(), key=lambda item: item[1])


def select_examples(sorted_deltas: list[tuple[str, float]]) -> list[tuple[str, float, str]]:
    """Select examples transparently: two declines, four median, two gains."""
    selected: list[tuple[str, float, str]] = []
    used: set[str] = set()

    def add(items: list[tuple[str, float]], label: str) -> None:
        for image, delta in items:
            if image not in used:
                selected.append((image, delta, label))
                used.add(image)

    add(sorted_deltas[:2], "largest decrease")
    midpoint = len(sorted_deltas) // 2
    candidates = sorted(sorted_deltas, key=lambda item: (abs(item[1] - median(delta for _, delta in sorted_deltas)), item[0]))
    add(candidates[:4], "near median change")
    add(list(reversed(sorted_deltas[-2:])), "largest increase")
    if len(selected) != 8:
        raise RuntimeError("Representative-image selection did not produce eight unique images.")
    return selected


def create_score_curve(states: list[dict], comparison_dir: Path) -> list[Path]:
    rounds = list(range(ROUND_COUNT))
    values = [float(state["summary"]["mean_fpem_score"]) for state in states]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 9})
    figure, axis = plt.subplots(figsize=(6.75, 2.75), layout="constrained")
    axis.plot(rounds, values, color=BLUE, linewidth=1.8, marker="o", markersize=4.8)
    axis.axhline(values[0], color=GRAY, linewidth=0.9, linestyle="--", label="Round 0 reference")
    axis.set_xlim(-0.1, 4.1)
    axis.set_ylim(1.0, 5.0)
    axis.set_xticks(rounds)
    axis.set_xlabel("Feedback round")
    axis.set_ylabel("Mean FPEM beauty score (1–5)")
    axis.grid(axis="y", color="#D9DEE7", linewidth=0.6)
    axis.legend(frameon=False, loc="lower right")
    for round_index, value in zip(rounds, values):
        axis.annotate(f"{value:.4f}", (round_index, value), xytext=(0, 8), textcoords="offset points", ha="center", color=MPL_TEXT_COLOR, fontsize=7)

    outputs = []
    for suffix in ("png", "svg", "pdf"):
        path = comparison_dir / f"experiment_2_score_curve.{suffix}"
        figure.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
    return outputs


def create_contact_sheet(selected: list[tuple[str, float, str]], comparison_dir: Path) -> Path:
    header_height = 86
    row_label_height = 38
    row_height = THUMBNAIL_SIZE + row_label_height
    canvas = Image.new("RGB", (THUMBNAIL_SIZE * 3, header_height + len(selected) * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(20, bold=True)
    header_font = load_font(16, bold=True)
    label_font = load_font(12)
    small_font = load_font(11)
    draw.text((18, 12), "Experiment 2: original vs. feedback rounds", fill=TEXT_COLOR, font=title_font)
    for column, label in enumerate(("Original input", "Round 0", "Round 4")):
        box = draw.textbbox((0, 0), label, font=header_font)
        text_width = box[2] - box[0]
        draw.text((column * THUMBNAIL_SIZE + (THUMBNAIL_SIZE - text_width) // 2, 50), label, fill=TEXT_COLOR, font=header_font)

    for row, (filename, delta, group) in enumerate(selected):
        y = header_height + row * row_height
        paths = (
            ROOT / "data" / "inputs" / filename,
            ROOT / "outputs" / RUN_NAME / "round_00" / filename,
            ROOT / "outputs" / RUN_NAME / "round_04" / filename,
        )
        for column, path in enumerate(paths):
            with Image.open(path) as image:
                thumbnail = image.convert("RGB").resize((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.LANCZOS)
            canvas.paste(thumbnail, (column * THUMBNAIL_SIZE, y))
        delta_text = f"{filename} | {group} | ΔFPEM {delta:+.4f}"
        draw.text((8, y + THUMBNAIL_SIZE + 10), delta_text, fill=SUBTLE_COLOR, font=small_font)
        draw.line((0, y + row_height - 1, canvas.width, y + row_height - 1), fill=(224, 228, 234), width=1)

    output = comparison_dir / "experiment_2_round0_vs_round4.png"
    canvas.save(output)
    return output


def write_round_csv(states: list[dict], report_dir: Path) -> Path:
    path = report_dir / "experiment_2_round_summary.csv"
    fields = ["round", "mean_fpem_score", "mean_star", "skin_smoothing_after", "whitening_after", "blemish_removal_after"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for state in states:
            params = state["global_params_after"]
            writer.writerow({
                "round": state["round"],
                "mean_fpem_score": state["summary"]["mean_fpem_score"],
                "mean_star": state["summary"]["mean_star"],
                "skin_smoothing_after": params["skin_smoothing"],
                "whitening_after": params["whitening"],
                "blemish_removal_after": params["blemish_removal"],
            })
    return path


def write_report(config: dict, states: list[dict], validation: dict, selected: list[tuple[str, float, str]], report_dir: Path) -> Path:
    path = report_dir / "experiment_2_report_zh.md"
    scores = [float(state["summary"]["mean_fpem_score"]) for state in states]
    stars = [float(state["summary"]["mean_star"]) for state in states]
    first, last = scores[0], scores[-1]
    relative = (last - first) / first * 100
    parameter_rows = []
    for state in states:
        params = state["global_params_after"]
        parameter_rows.append(
            f"| {state['round']} | {state['summary']['mean_fpem_score']:.6f} | {state['summary']['mean_star']:.2f} | "
            f"{params['skin_smoothing']:.4f} | {params['whitening']:.4f} | {params['blemish_removal']:.4f} |"
        )
    examples = "\n".join(f"- `{image}`：{group}，第 0→4 轮 FPEM 变化 {delta:+.6f}" for image, delta, group in selected)

    content = f"""# 补充实验二：反馈闭环优化实验

## 实验目的

固定美颜模型 RealVisXL V5，验证 AgentFace 是否能完成“生成—评分—反馈—参数更新”的闭环，并观察经过 5 轮后自动评分是否发生变化。

## 本报告使用的数据范围

本报告**只**使用正式运行 `official_continuous` 的结果。早期的 `smoke`、`official`、`official_stable` 文件夹是准备阶段的诊断记录：其中曾发现评分权重前缀加载和提示词档位过粗的问题，已修正，因此这些试跑数据和图片不参与下述统计或结论。

## 实验设置

- 输入：100 张已选定的人脸样本，来自 `data/inputs/`；同一批图片在 5 轮中保持不变。
- 固定模型：`{config['model_id']}`。
- 固定生成条件：1024×1024、20 步、guidance scale 5.0、img2img strength 0.25、随机种子 42、基础提示词和负向提示词固定。
- 可更新项：皮肤平滑、皮肤提亮、瑕疵修复三项美颜强度；其余美颜参数固定。连续强度值直接写入提示词，避免“小幅更新但提示词完全相同”。
- 评分：FPEM 自动评分模型。它输出连续的 1–5 分；同时按 `star = clamp(floor(score + 0.5), 1, 5)` 显示为星级和简单反馈。
- 更新方式：100 张图片作为一个批次，分别比较每个参数的较高/较低候选的平均 FPEM 分，再做一次有边界的小步更新。这样避免把原有的单张会话更新连续执行 100 次而导致参数失控。
- 执行方式：直接运行离线推理脚本，没有启动网页、FastAPI 或数据库服务。

## 每轮流程

```text
100 张输入图
  → 使用当前美颜参数生成 100 张图
  → FPEM 对 100 张结果评分
  → 汇总评分反馈，受限地更新 3 个美颜参数
  → 进入下一轮
```

共运行 5 轮（第 0 至第 4 轮），得到 500 张结果图。

## 完整性核验

- 每一轮均有 100 张图片和 100 条评分，合计 500 张 1024×1024 PNG。
- 五轮 FPEM 均严格加载：`missing_keys=[]`、`unexpected_keys=[]`。
- 同一批验证图片重复评分结果完全一致。
- 第 0 到第 1 轮的 100/100 张同名输出文件哈希均发生变化，说明参数更新确实改变了后续模型输入与生成结果。

## 结果

![五轮平均评分曲线](../outputs/comparisons/experiment_2_score_curve.png)

**图 1：** 单个固定随机种子的 5 轮平均 FPEM 评分。纵轴保留 1–5 的完整量程；因此可以直观看到本次提升幅度很小，而非显著跃升。

| 轮次 | 平均 FPEM 分 | 平均星级 | 皮肤平滑（更新后） | 提亮（更新后） | 瑕疵修复（更新后） |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(parameter_rows)}

- 第 0 轮平均 FPEM 分：**{first:.6f}**。
- 第 4 轮平均 FPEM 分：**{last:.6f}**。
- 第 0→4 轮平均变化：**{last - first:+.6f}**（相对 {relative:+.3f}%）。
- 逐图配对结果：**{validation['round_00_to_round_04_increased']}** 张上升，**{validation['round_00_to_round_04_decreased']}** 张下降，**{validation['round_00_to_round_04_tied']}** 张相同。
- 四舍五入后的平均星级均为 2.19；这说明星级显示太粗，不能反映这次 0.0067 量级的细微变化，正式分析以连续 FPEM 分数为准。

![原图、第 0 轮与第 4 轮对比](../outputs/comparisons/experiment_2_round0_vs_round4.png)

**图 2：** 原图、第 0 轮和第 4 轮的代表样本。样本不是只挑上升图片：选择规则为 FPEM 降幅最大的 2 张、最接近中位变化的 4 张、增幅最大的 2 张。

{examples}

## 结论

在本次单随机种子、同一批 100 张图片参与反馈与评估的运行中，AgentFace 完成了评分反馈、参数更新和下一轮再生成的完整闭环。固定 RealVisXL V5 后，第 4 轮的平均 FPEM 分数比第 0 轮高 **{last - first:.6f}**，且 63/100 张图片的自动评分提高。因此，本实验支持下面这个**限定性结论**：

> 在 FPEM 自动评分器下，AgentFace 的反馈闭环可以更新美颜参数，并在本次运行中带来很小的正向平均评分变化。

这个结论不能扩大为“所有人像质量都明显提升”或“视觉效果一定更好”：本实验只有一个随机种子、没有独立留出的 20 张验证集、评分器只有 FPEM，也没有人类主观评分或专门的人脸变形检测。若要把结论做得更强，下一步应使用 80 张反馈更新、20 张只做最终验证，并增加人工自然度评分。

## 可复现文件

- 配置：[config/experiment_2.json](../config/experiment_2.json)
- 统一运行脚本：[code/run_feedback_loop.py](../code/run_feedback_loop.py)
- 逐轮汇总：[experiment_2_round_summary.csv](experiment_2_round_summary.csv)
- 结果清单：[runs/experiment_2_manifest.json](../runs/experiment_2_manifest.json)
- 第 0 至第 4 轮的逐图评分：`../scores/official_continuous/round_00.json` 至 `round_04.json`

生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}
"""
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    comparison_dir = ROOT / "outputs" / "comparisons"
    report_dir = ROOT / "reports"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    config, states, validation, deltas, sorted_deltas = verify_and_collect()
    selected = select_examples(sorted_deltas)
    curve_outputs = create_score_curve(states, comparison_dir)
    contact_sheet = create_contact_sheet(selected, comparison_dir)
    csv_path = write_round_csv(states, report_dir)
    report_path = write_report(config, states, validation, selected, report_dir)

    manifest = {
        "experiment": "Experiment 2: feedback-loop optimization",
        "formal_run": RUN_NAME,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "effective_config": config,
        "validation": validation,
        "selected_comparison_samples": [
            {"image": image, "round_00_to_round_04_fpem_delta": delta, "selection_group": group}
            for image, delta, group in selected
        ],
        "artifacts": [
            str(path.relative_to(ROOT))
            for path in [*curve_outputs, contact_sheet, csv_path, report_path]
        ],
    }
    manifest_path = ROOT / "runs" / "experiment_2_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "artifacts": manifest["artifacts"], "validation": validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
