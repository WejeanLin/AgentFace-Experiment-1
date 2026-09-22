# AgentFace 补充实验记录

这个公开仓库记录了 AgentFace 的两个离线补充实验。实验均直接调用推理脚本完成，不启动网页、FastAPI 或数据库服务。

| 实验 | 要验证什么 | 正式结果 |
| --- | --- | --- |
| 实验一：美颜模型可插拔性 | 同一位置能否替换不同的开源美颜模型 | 4 个模型均在统一流程下完成推理，生成 4×4=16 张结果图。 |
| 实验二：反馈闭环优化 | 评分反馈能否更新参数，并影响后续生成 | 固定 RealVisXL V5，运行 5 轮×100 张；第 4 轮 FPEM 平均分比第 0 轮高 0.006702，100 张中 63 张提高、37 张降低。 |

实验二的分数增幅很小，因此只说明：在本次单随机种子、FPEM 自动评分条件下，反馈闭环产生了小幅正向平均变化。它不表示所有图片的视觉效果都有明显提升，也不等同于人工主观评价。

## 实验一：模型可插拔性

实验一固定输入图片、提示词、尺寸、步数、随机种子和保存规则，只替换前置美颜模型：

- RealVisXL V5：皮肤瑕疵修复方向；
- Juggernaut XL v9：皮肤纹理和光影方向；
- CyberRealistic XL：眼睛、嘴唇等五官细节方向；
- epiCRealism XL：整体人物真实感方向。

4 张示例人像分别经过 4 个模型处理，共得到 16 张效果图。该实验验证的是统一接口下模型能够替换和正常运行，并不比较哪个模型绝对更好。

![实验一：原图与四个模型结果](outputs/comparisons/experiment_1_overview.png)

- [实验一中文报告](reports/实验一_美颜模型可插拔性验证.md)
- [固定实验配置](config/experiment_1.json)
- [统一推理脚本](code/run_model_swap.py)
- [输入与结果哈希清单](runs/experiment_1_manifest.json)

## 实验二：评分反馈闭环优化

实验二固定 RealVisXL V5 及生成设置，用 100 张选定样本连续运行 5 轮。每轮先生成 100 张图片，再用 FPEM 自动评分；系统依据整批评分，对皮肤平滑、提亮、瑕疵修复三个参数进行受限的小步更新，并进入下一轮。

正式运行名为 `official_continuous`。完整性核验结果如下：

- 5 轮均生成并评分 100 张图，共 500 张 1024×1024 结果；
- FPEM 在 5 轮中均严格加载，未出现缺失或意外权重；
- 同一批图片重复评分一致；
- 第 0→1 轮的 100 张同名输出均发生变化，说明更新确实进入了后续推理；
- 第 0→4 轮平均 FPEM：2.038012 → 2.044714（+0.006702，+0.329%）；
- 第 0→4 轮逐图配对：63 张提高，37 张降低，0 张相同。

![实验二：5 轮平均评分](experiment_2/outputs/comparisons/experiment_2_score_curve.png)

![实验二：原图、第 0 轮与第 4 轮代表样本](experiment_2/outputs/comparisons/experiment_2_round0_vs_round4.png)

- [实验二中文报告：目的、过程、设置、结果和限制](experiment_2/reports/experiment_2_report_zh.md)
- [实验二固定配置](experiment_2/config/experiment_2.json)
- [反馈闭环推理脚本](experiment_2/code/run_feedback_loop.py)
- [生成报告和对比图的脚本](experiment_2/code/create_experiment2_artifacts.py)
- [逐轮评分汇总 CSV](experiment_2/reports/experiment_2_round_summary.csv)
- [参数、产物与核验清单](experiment_2/runs/experiment_2_manifest.json)
- [第 0 至第 4 轮逐图评分](experiment_2/scores/official_continuous/)

## 仓库内容与未公开内容

本仓库包含实验脚本、固定配置、哈希/评分记录、实验报告及必要的结果对比图，方便查看实验过程和结论。

以下内容刻意不上传：

- SDXL、美颜模型和 FPEM 的权重；
- 实验二的 100 张原始输入和 500 张全量结果图；
- 实验室服务端代码、数据库和运行日志。

因此，实验二的完整输入和全量输出仍保留在实验室服务器；公开仓库保留了可核验的统计、逐图分数、代表性对比图和用于复现流程的脚本。使用任何模型或数据集前，请自行遵守其许可证和使用要求。

## 复现提示

实验一需要把四个模型权重放入根目录下的 `models/`，然后执行：

```bash
pip install -r requirements-offline.txt
python code/run_model_swap.py --check
python code/run_model_swap.py
```

实验二还需要准备 100 张输入图到 `experiment_2/data/inputs/`、合法取得 FPEM 评分器及其权重，并按本机环境修改 `experiment_2/config/experiment_2.json` 中的评分器 Python 路径。准备完成后，在仓库根目录执行：

```bash
python experiment_2/code/run_feedback_loop.py --check
python experiment_2/code/run_feedback_loop.py --run-name official_continuous
python experiment_2/code/create_experiment2_artifacts.py
```

原始 AgentFace 项目：<https://github.com/GDUE-DVL/AgentFace>
