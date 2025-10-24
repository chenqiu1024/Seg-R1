# 整理对话历史与索引创建
_Exported on 10/24/2025 at 06:44:43 PDT from Cursor (1.7.53)_

---

**User**

我把历次与Cursor对话的历史导出为markdown文件保存在docs目录下（其中不同文件也可能有相互重叠的内容）。请按照时间顺序阅读以了解上下文，弄明白我想做的事情的全貌，并在docs下再新建一个文本文件，在其中记录下你的理解，以及一个大纲形式的索引，使你在需要了解更详细的细节（如具体代码实现、公式推导，等等）的时候可以通过该索引快速定位到具体哪个markdown文件或本工程中的哪个源文件的哪个位置，从而使得每次开启一个新的人-AI对话时，你无需重新阅读和记忆冗长的全部信息，仅需阅读该较短的索引，就能快速理解事情的全貌，并在需要时仍然可以快速定位到详细信息的来源。

---

**Cursor**

I'll first scan the docs for headings to build a chronological outline and inspect the repo structure to identify key code entry points. Then I'll sample a few key docs to understand the project's goals and context before writing the new index file.

I’m going to catalog the docs chronologically, identify core code entry points and key functions with line numbers, then create a concise knowledge map file in docs that summarizes your goals and provides a fast index to docs and source locations.

I'll extract dates from filenames, read the top of each doc for title/context, and scan source files for entry points in parallel to build the index.

I’m going to scan key Python modules for class and function definitions to include accurate line-numbered entry points in the index, in parallel across the main subpackages.

I cataloged the docs and scanned the codebase to extract key entry points and function line numbers. Now I’ll create a concise knowledge map file under docs with a chronological outline and a fast index to code and docs.

```markdown
## Seg-R0 知识地图与快速索引（2025-10-24）

本文件概述本工程的目标、核心模块与流程，并提供按时间顺序的对话纪要索引与源代码定位索引。每次开启新会话时先阅读本文件，再按需跳转到具体文档或源码行号处查细节。

### 一、整体目标与方案（我的理解）
- **总体目标**: 构建“点→分割”的交互式分割系统，并用强化学习（GRPO）优化点选策略，使少量提示点即可得到高质量分割。
- **核心流程**:
  1) 以热力图模型预测下一提示点坐标（软分布输出，支持 soft-argmax/采样）。
  2) 将图像与点提示输入 SAM2，生成分割掩膜。
  3) 评估与可视化，形成监督/奖励信号。
  4) 训练：先监督/预训练，再用 GRPO 微调点选策略。
- **关键特性**:
  - 热力图为“软”分布（非 one-hot），支持高分辨率、U-Net 小模型、KL/MSE/CE 多损失，温度/σ 等可调。
  - 采样策略具备层级分解（cell + 子像素偏移），并可扩展“联合策略：标签 + cell + 子像素”。
  - 完整的可视化与评估工具链，含掩膜评测与点/分割对齐的调试图。

### 二、工程模块总览（代码定位）
- 热力图/点预测（监督与RL）: `seg-rl/heatmap/`
  - 模型与采样: `model.py`
    - `class PointHeatmapModel` 行127
    - `argmax_from_logits` 行198；`soft_argmax_from_logits` 行207
    - 层级采样与log_prob：`sample_cell_and_offset` 行243，`action_to_continuous_xy` 行295，`log_prob_of_action` 行302
    - 联合策略（标签+cell+偏移）：`sample_joint_label_cell_offset` 行329；`log_prob_of_joint_action` 行369
  - 数据/训练: `datasets.py`（`JsonlPointDataset` 行84；`SamSequencePointDataset` 行289；`collate_fn` 行281），`losses.py`（`ce_over_pixels` 行9；`gaussian_heatmap_targets` 行26；`kl_to_gaussian_targets` 行61；`mse_to_gaussian_targets` 行88），`train.py`（`parse_args` 行92；`main` 行131）
  - GRPO 点策略训练: `train_grpo_points.py`（`validate` 行364；`train` 行413；`__main__` 行744）
  - 推理/调试: `predict_next_point_from_model.py`（`_to_tensor_separate` 行161；`run_initial` 行209；`run_append` 行263；`main` 行399）
  - 工具: `utils.py`（`compute_pck` 行44；`draw_*` 59/68/82；`overlay_heatmap` 行122）
- SAM2 集成与分割: `seg-rl/sam2_segment_from_points.py`（`SAMWrapper` 行125；`calculate_bounding_box` 行324；`main` 行389；`__main__` 行565），`seg-rl/sam2_segment_simple.py`
- 端到端编排: `seg-rl/predict_points_and_sam.py`（`main` 行172）
- 评估与可视化:
  - 掩膜评测: `seg-rl/evaluation/eval_sam_masks.py`（`_compute_metrics` 行161；`main` 行206）
  - 可视化: `seg-rl/visualization/viz_heuristic_sam_points.py`（`_draw_caret` 行94；`_draw_cross` 行108；`main` 行147），`viz_sam_segmentation.py`
- RL 大框架（Seg-R1，Qwen2VL + GRPO）: `seg-r1/src/open_r1/`
  - `grpo.py`（`SAMWrapper` 行100；`segmentation_reward` 行311；`main` 行474）
  - 训练器: `trainer/vllm_grpo_trainer_modified.py`（`Qwen2VLGRPOVLLMTrainerModified` 行82），`trainer/grpo_trainer.py`（`Qwen2VLGRPOTrainer` 行64）

提示：本文中的“行X”基于当前仓库版本，若有改动，请在 IDE 中跳转并更新。

### 三、按时间顺序的对话与设计文档索引
（仅列要点与关键锚点，需细节时再打开对应文档）

- 2025-08-31
  - `docs/runtime_error_in_output-20250831.md`，`docs/cursor_resolve_runtime_error_in_output-20250831.md`
    - 主题：运行时输出/异常排查记录，命令与日志收集、修复路径。
  - `docs/cursor_resolve_runtime_error_in_termina.md`
    - 主题：终端交互式报错的分析与修复流程。

- 2025-09-23
  - `docs/cursor_pretrain_design-20250923.md`
    - 主题：将点预测由“硬”目标改为“软”热力图目标；U-Net 小模型；KL/MSE/温度等配置；涉及 `seg-rl/heatmap` 多文件联动改造。

- 2025-09-24
  - `docs/cursor_sam2_segment-20250924.md`（导出时间见文首）
    - 主题：`sam2_segment_*` 增强；输出 JSON/JSONL；最小包围盒；导入路径修正；使用说明与参数详列。
    - 文档内“功能更新总结”、“SAM2 Automatic Segmentation Evaluation”等小节包含具体 I/O 与示例。

- 2025-09-28
  - `docs/cursor_20250928.md`
    - 主题：差异区域选择 + 代表点提取方案；结论：CCL + EDT；含极简参考实现与工程要点。
  - `docs/cursor_pretrain_flow_all_2025092801.md`
    - 主题：预训练数据流与全流程整理（与 0928 有重叠）；包含路径组织与评估可视化说明。

- 2025-10-02
  - `docs/cursor_20251002.md`
    - 主题：可视化易读性增强（浅色背景下点/线标注描边）；对应 `viz_heuristic_sam_points.py` 中 `_draw_caret/_draw_cross` 的实现。

- 2025-10-06
  - `docs/cursor_20251006.md` / `docs/cursor_heatmap_model.md`（大体一致，后者像是汇编版）
    - 主题：
      - 有监督热力图模型全面设计与落地：输入输出/架构/损失/训练/评估/工程要点/推荐配置。
      - 面向 GRPO 的层级策略与“联合策略（标签+cell+子像素）”细化，含采样与 log_prob 公式与伪代码。
      - 若需快速定位“层级策略采样”段落：见 `cursor_heatmap_model.md` 行962、2274、3032、10077；“联合策略”见行3137、10182 等。
      - 后半部分含一系列修复与参数建议（学习率调度、梯度裁剪、PCK 多阈值等），在文末“总结/推荐命令”附近可查（约 21009 起）。

- 2025-10-17
  - `docs/cursor_20251017_rl.md`
    - 主题：LabelHead 与联合策略的替换/实现要点；联合策略采样与 log_prob 细节；SAM2 集成段落。

- 2025-10-22
  - `docs/cursor_fix_rl_v1_20251022.md`
    - 主题：训练中出现的 `ValueError` 的定位与修复；文档内多次出现“SAM2 integration”小节，涉及依赖与导入稳健性。

- 其他专题/需求文档（数据与可视化等）
  - `docs/rl_data_requirements.md`，`docs/prerl_data_requirements.md`，`docs/sft_data_requirements.md`，`docs/sod_finetune_data_requirements.md`
    - 主题：不同训练阶段（RL/Pre-RL/SFT/SOD）的数据字段与文件组织要求。
  - `docs/cursor_convert_dataset_for_seg_rl_train.md`
    - 主题：将现有数据转为适配本项目训练/评估的数据格式与脚本流程。
  - `docs/cursor_mask_comparison.md`，`docs/cursor_opt_pretrain_vis.md`，`docs/cursor_code_understand-1.md`，`docs/cursor_understanding_maybe_apply_chat_t.md`
    - 主题：可视化优化、掩膜对比、代码理解与沟通方式等辅助材料。

### 四、常见问题到代码/文档的快速跳转
- 想“端到端”跑通：看 `seg-rl/predict_points_and_sam.py` 的 `main` 行172；配合 `docs/cursor_sam2_segment-20250924.md` 的使用说明与参数。
- 想训练“监督热力图模型”：
  - 配置/入口见 `seg-rl/heatmap/train.py`（`parse_args` 行92；`main` 行131）。
  - 模型定义与采样见 `seg-rl/heatmap/model.py`（详见上文行号）。
  - 损失函数/目标生成见 `seg-rl/heatmap/losses.py`；数据集见 `datasets.py`。
- 想做“GRPO 点策略微调”：看 `seg-rl/heatmap/train_grpo_points.py`（`validate` 行364；`train` 行413）。
- 想只做“点预测推理/调试”：看 `seg-rl/heatmap/predict_next_point_from_model.py`（`run_initial` 行209；`run_append` 行263；`main` 行399）。
- 想“用点喂给 SAM2 出掩膜”：看 `seg-rl/sam2_segment_from_points.py`（`SAMWrapper` 行125；`calculate_bounding_box` 行324；`main` 行389）。
- 想“评估掩膜质量”：看 `seg-rl/evaluation/eval_sam_masks.py`（`_compute_metrics` 行161；`main` 行206）。
- 想“调试可视化（点更清晰）”：看 `seg-rl/visualization/viz_heuristic_sam_points.py`（`_draw_caret` 行94；`_draw_cross` 行108）。
- 想理解“层级/联合策略”与数学推导：优先看 `docs/cursor_heatmap_model.md` 对应小节（行962/2274/3032/10077；3137/10182 等），再对照 `model.py` 相应函数实现。

### 五、数据与产物路径要点
- 数据集与 JSONL：参考 `docs/*_data_requirements.md` 与 `seg-rl/annotator/gen_point_jsonl_from_masks.py`。
- 典型输出目录：`outputs/heuristic_points*`、`outputs/sam_everything/*`、`outputs/seg_r1_md/*`、`output/braintumour/*` 等。
- 预训练权重示例：`pretrained/points_predictor-251001-160epochs.pt`。

### 六、建议的阅读顺序
1) 本索引（本文）→ 2) `cursor_heatmap_model.md` 的“核心思路/层级与联合策略”→ 3) `seg-rl/heatmap/model.py` 实现 → 4) `train.py` 与 `train_grpo_points.py` 训练入口 → 5) `sam2_segment_from_points.py` → 6) 评估与可视化脚本。

---
维护说明：新增文档或改动核心函数后，请同步更新本文的行号锚点与条目简述。


```

Created the knowledge map and index. You can open:
- docs/knowledge_map_index.md

1) I scanned your docs chronologically and the codebase to extract core goals and entry points. 2) I wrote a concise knowledge map with a time-ordered outline and source anchors.

- Added docs/knowledge_map_index.md with:
  - High-level intent and pipeline.
  - Time-ordered index of your Cursor chats (with topics).
  - Fast source index with function anchors by line number for `seg-rl/heatmap`, `sam2_segment_from_points.py`, `predict_points_and_sam.py`, eval and visualization scripts, and `seg-r1` trainers.
  - Quick-jump FAQ-style links to code/docs sections.
  - Reading order and maintenance note.

If you want, I can also:
- Auto-generate hyperlinks to exact lines via your IDE’s URL scheme.
- Add more anchors for other files (e.g., `scripts/*.sh`, `eval/*`) or specific formulas in the large design docs.

---

**User**

好的。那么基于你的了解，Seg-R0/outputs/braintumour/heatmap_train-251001-optimized/vis/val_or_train_vis.png 这种热力图 与 预测&真值 提示点 与 原输入图像 画在一起的可视化图像，是由哪个本工程中的脚本绘制的？

---

**Cursor**

Let me search the heatmap training scripts for where they save visualizations (e.g., a 'vis' folder or 'val_or_train_vis.png') to identify the exact script.

I'll open the training visualization section in the heatmap training script to confirm the exact saved filename and path.

Let me search the repository for any file saves matching 'val_or_train_vis.png' or similar to confirm which script names the output that way.

I'll locate the exact save call and filename pattern in the training visualization code to confirm the script and how it names the output.

- Script: seg-rl/heatmap/train.py
- Save path pattern: it saves a grid image under out_dir/vis as "<split_name>_vis.png". For a combined validation/train figure, the docs refer to this as "val_or_train_vis.png", but the code writes separate files like "val_vis.png" or "train_vis.png" depending on split_name.

So the image you cited is produced by the heatmap training script’s visualization block.

---

**User**

请为Seg-R0/seg-rl/sam2_segment_from_points.py添加功能：对每个样本图像作分割时，除了生成预测mask图像 {$i}.png 以外（其中i是当前采用的提示点序列长度 - 1），还要另外生成一幅heatmap-{$i}.png “热力图”图像放在同一个目录下。该执力图内容即为 提示点热力图预测模型 在被输入样本图像和上一步预测mask图像（当第一步时上一步预测mask图像就是一个全零灰度值图像）时，其推理得到的概率热力图（最终选择的提示点就是选取概率图里值最大的一个点）。完成后请加入必要的代码注释，并讲解和论证你的代码是正确的。

---

**Cursor**

I'll open helper code to reuse the exact heatmap-to-image conversion used in the heatmap module, and inspect sam2_segment_from_points.py around the SAM flow to add the new heatmap save step.

I’m going to add the heatmap-generation feature to `seg-rl/sam2_segment_from_points.py`: load the heatmap model from a new CLI arg, compute the probability heatmap conditioned on the previous mask (or zeros on the first step), and save `heatmap-{$i}.png` alongside each `{i}.png` mask. Then I’ll run a linter check on the edited file.

```python
#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割

读取由点提示组成的输入（支持JSON数组或JSONL逐行），使用SAM2进行分割，
将预测mask保存为灰度图像，并按要求生成/追加每幅图像对应的bbox日志。

功能:
- 从点提示生成图像分割mask
- 自动计算每个mask的最小包围盒
- 可选择输出包含mask路径和包围盒坐标的JSON文件

输入JSON格式（数组）或JSONL（逐行）:
    新格式: {"image": "/path/img.jpg", "points": [[x1,y1], [x2,y2]], "labels": [1, 0]}
    旧格式: {"image": "/path/img.jpg", "x": x1, "y": y1}

输出:
    - 预测mask按图像stem创建子目录保存为灰度图：output_dir/<stem>/<k>.png
      其中k为提示点序列长度-1；像素值：0=背景，255=前景
    - 在output_dir生成<stem>.jsonl（JSON对象，非数组）。该文件内容保持为：
      {"count": N, "bboxes": [[x1,y1,x2,y2], ...]}。每次新增一个mask，仅向
      bboxes追加一个新的bbox并将count递增，无需重写历史bbox。
    - 将输入JSON数组拷贝到 --json_output，并为每条加入/覆写
      "sam_masks_dir": output_dir

依赖项:
    - SAM2 (Segment Anything Model 2)
    - torch, PIL, opencv-python, numpy

安装SAM2:
    git clone https://github.com/facebookresearch/segment-anything-2.git third_party/sam2
    cd third_party/sam2 && pip install -e .

下载模型:
    # 下载SAM2.1 Hiera Large模型 (~900MB)
    wget -O third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

基础用法:
    python seg-rl/sam2_segment_from_points.py \
      --input_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour.jsonl \
      --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/pretrain_gt_masks \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda \

带JSON输出:
    python seg-rl/sam2_segment_from_points.py \
      --input_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour.jsonl \
      --json_output /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour.jsonl \
      --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/pretrain_gt_masks \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda \
      --resize 512 512 \
      --skip_existing

    python seg-rl/sam2_segment_from_points.py \
      --input_jsonl outputs/braintumour/pred_points-e160-251019.jsonl \
      --json_output outputs/braintumour/pred_points-e160-251019.jsonl \
      --output_dir outputs/braintumour/pred_masks-e160-251019 \
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda \
      --resize 512 512 \
      --skip_existing

    python seg-rl/sam2_segment_from_points.py \
      --input_jsonl outputs/braintumour/pred_points-sft_e185-251018.jsonl \
      --json_output outputs/braintumour/pred_points-sft_e185-251018.jsonl \
      --output_dir outputs/braintumour/pred_masks-sft_e185-251018 \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda \
      --resize 512 512 \
      --skip_existing
 
    /opt/anaconda3/envs/seg-r1/bin/python seg-rl/sam2_segment_from_points.py \
      --input_jsonl datasets/seg_r1_md/Task01_BrainTumour/pred_points-251001.jsonl \
      --json_output datasets/seg_r1_md/Task01_BrainTumour/pred_points-251001.jsonl \
      --output_dir outputs/braintumour/pretrain_pred_masks-251001 \
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device mps \
      --resize 512 512 \
      --skip_existing

详细参数及输出格式说明请参考：docs/cursor_pretrain_flow_all_2025092801.md）：
    

"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# 导入热力图点预测模型与可视化工具（用于生成 heatmap-{i}.png）
try:
    from heatmap.model import ModelConfig as _HMModelConfig, PointHeatmapModel as _PointHeatmapModel  # type: ignore
    import torch.nn.functional as _F  # type: ignore
    from heatmap.utils import heatmap_to_pil as _heatmap_to_pil  # type: ignore
    _HEATMAP_AVAILABLE = True
except Exception:
    # 若用户未提供热力图权重或未安装依赖，则仅跳过热力图生成功能
    _HEATMAP_AVAILABLE = False

# 添加sam2路径
sys.path.append(str(Path(__file__).parent.parent / "third_party" / "sam2"))

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"Error importing SAM2: {e}")
    print("Please ensure SAM2 is properly installed and the path is correct.")
    sys.exit(1)


def _print_progress(num_processed: int, num_skipped: int, num_errors: int, note: str) -> None:
    """在同一行输出进度信息"""
    msg = f"[Progress] processed={num_processed} skipped={num_skipped} errors={num_errors} | {note}"
    print(f"\r{msg}", end="", flush=True)


def to_abs(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.abspath(path) if not os.path.isabs(path) else path


class SAMWrapper:
    """SAM2包装器，用于图像分割预测"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """初始化SAM2模型和预测器
        
        Args:
            model_path: SAM2模型检查点路径
            device: 运行设备 (e.g. "cuda", "cuda:0", "cpu")
                   如果为None，将自动检测可用设备
        """
        # 自动检测设备
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        # SAM2配置文件路径
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam_model = build_sam2(model_cfg, model_path, device=self.device) 
        # # 检查配置文件是否存在
        # config_path = Path(__file__).parent.parent / "third_party" / "sam2" / "sam2" / model_cfg
        # if not config_path.exists():
        #     # 尝试相对于当前目录
        #     config_path = Path(model_cfg)
        #     if not config_path.exists():
        #         raise FileNotFoundError(f"SAM2 config file not found: {model_cfg}")
        # sam_model = build_sam2(str(config_path), model_path)
        
        # # 移动到指定设备
        # sam_model = sam_model.to(self.device)
        
        # 初始化预测器
        self.predictor = SAM2ImagePredictor(sam_model)
        self.last_mask = None
        
    def predict(self, 
                image: PILImage.Image, 
                points: Optional[List[Tuple[int, int]]] = None, 
                labels: Optional[List[int]] = None,
                bbox: Optional[List[int]] = None) -> Tuple[np.ndarray, float]:
        """使用给定提示运行SAM2预测
        
        Args:
            image: 输入PIL图像
            points: 点坐标列表 [(x,y), ...]
            labels: 点标签列表 (1=前景, 0=背景)
            bbox: 可选边界框 [x1,y1,x2,y2]
            
        Returns:
            (predicted_mask, confidence_score)的元组
        """
        input_points = np.array(points) if points else None
        input_labels = np.array(labels) if labels else None
        input_bboxes = np.array([bbox]) if bbox else None

        # 转换为numpy数组
        image_np = np.array(image)
        
        # 确保是RGB格式
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image_np
        
        # 设置图像
        self.predictor.set_image(rgb_image)
        
        # 预测
        mask_pred, score, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_bboxes,
            multimask_output=False,
        )
        
        self.last_mask = mask_pred[0]
        return mask_pred[0], score[0]


def validate_and_extract_points(obj: Dict[str, Any], line_num: int) -> Tuple[bool, Optional[List[Tuple[float, float]]], Optional[List[int]]]:
    """验证并提取JSON条目中的点和标签
    
    Args:
        obj: JSON对象
        line_num: 行号（用于错误报告）
        
    Returns:
        (is_valid, points, labels)的元组
    """
    if "image" not in obj:
        print(f"[WARN] Line {line_num}: Missing 'image' field")
        return False, None, None
        
    # 检查格式：新格式有"points"和"labels"，旧格式有"x"和"y"
    if "points" in obj and "labels" in obj:
        # 新格式
        points = obj["points"]
        labels = obj["labels"]
        
        if not isinstance(points, list) or not isinstance(labels, list):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must be lists")
            return False, None, None
            
        if len(points) != len(labels):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must have same length")
            return False, None, None
        
        # 验证点格式
        valid_points = []
        valid_labels = []
        for point, label in zip(points, labels):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                print(f"[WARN] Line {line_num}: Invalid point format {point}")
                continue
            valid_points.append((float(point[0]), float(point[1])))
            valid_labels.append(int(label))
            
        if len(valid_points) == 0:
            print(f"[WARN] Line {line_num}: No valid points found")
            return False, None, None
            
        return True, valid_points, valid_labels
        
    elif "x" in obj and "y" in obj:
        # 旧格式 - 假设为正类点
        x = float(obj["x"])
        y = float(obj["y"])
        return True, [(x, y)], [1]
        
    else:
        print(f"[WARN] Line {line_num}: Missing coordinate data (need 'points'+'labels' or 'x'+'y')")
        return False, None, None


def get_output_path(image_path: str, output_dir: str) -> str:
    """生成输出mask文件路径（旧：单文件路径；新：保留但未使用）
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        
    Returns:
        输出mask文件路径
    """
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def get_mask_dir_for_image(image_path: str, output_dir: str) -> str:
    """返回该图像对应的mask子目录路径 output_dir/<stem>"""
    image_name = Path(image_path).stem
    return os.path.join(output_dir, image_name)


def get_mask_index_from_points(points: List[Tuple[float, float]]) -> int:
    """根据点序列长度返回mask索引：len(points)-1"""
    return max(0, int(len(points) - 1))


def get_mask_path_for_points(image_path: str, output_dir: str, points: List[Tuple[float, float]]) -> str:
    """输出mask路径：output_dir/<stem>/<k>.png，k=len(points)-1"""
    mask_dir = get_mask_dir_for_image(image_path, output_dir)
    os.makedirs(mask_dir, exist_ok=True)
    k = get_mask_index_from_points(points)
    return os.path.join(mask_dir, f"{k}.png")


def append_state_log(output_dir: str, image_path: str, bbox: Tuple[int, int, int, int]) -> None:
    """在 <output_dir>/<stem>.jsonl 中维护单个JSON对象：
    {"count": N, "bboxes": [[x1,y1,x2,y2], ...]}
    仅追加当前bbox并将count递增。不会重新读取或计算历史bbox。
    """
    stem = Path(image_path).stem
    log_path = os.path.join(output_dir, f"{stem}.jsonl")

    record: Dict[str, Any] = {"count": 0, "bboxes": []}
    if os.path.isfile(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "count" in parsed and "bboxes" in parsed:
                        record = parsed
        except Exception:
            pass

    # 追加当前bbox并递增count
    record.setdefault("bboxes", [])
    record["bboxes"].append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
    record["count"] = int(record.get("count", 0)) + 1

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False)


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


def save_mask_as_grayscale(mask: np.ndarray, output_path: str) -> None:
    """将mask保存为灰度图像
    
    Args:
        mask: 二值mask数组
        output_path: 输出文件路径
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 转换为灰度值 (0=背景, 255=前景)
    mask_gray = mask_binary * 255
    
    # 保存
    cv2.imwrite(output_path, mask_gray)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Use SAM2 to segment images from point prompts")
    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Input JSON array (preferred) or JSONL file containing image paths and point coordinates")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory to save mask images")
    p.add_argument("--sam_checkpoint", type=str, required=True,
                   help="Path to SAM2 model checkpoint")
    p.add_argument("--device", type=str, default=None,
                   help="Device to run on (cuda/cpu). Auto-detect if not specified")
    p.add_argument("--config_path", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml",
                   help="Path to SAM2 config file")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip processing if output file already exists")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                   help="Resize input images to specified size [width height]")
    p.add_argument("--json_output", type=str, default=None,
                   help="Path to write the copied input JSON array with added 'sam_masks_dir'")
    # 额外：生成提示点概率热力图（heatmap-{i}.png）相关参数
    p.add_argument("--heatmap_model", type=str, default=None,
                   help="Path to heatmap predictor checkpoint (.pt); if set, also save heatmap-{i}.png")
    p.add_argument("--heatmap_tau", type=float, default=1.0,
                   help="Softmax temperature for probability heatmap generation")
    p.add_argument("--heatmap_size", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                   help="Inference size [width height] for heatmap model; default = use --resize or native size")
    return p.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.isfile(to_abs(args.input_jsonl)):
        print(f"Error: Input JSONL file not found: {args.input_jsonl}")
        return 1
    
    # 检查SAM2检查点
    if not os.path.isfile(to_abs(args.sam_checkpoint)):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    args.output_dir = to_abs(args.output_dir) or args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {to_abs(args.sam_checkpoint)}")
    print(f"Using device: {args.device or 'auto-detect'}")
    
    try:
        sam_wrapper = SAMWrapper(to_abs(args.sam_checkpoint), args.device)
        print("SAM2 initialized successfully")
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 可选：初始化热力图点预测模型
    hm_model = None
    hm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if (args.device is None) else torch.device(args.device)
    if _HEATMAP_AVAILABLE and args.heatmap_model:
        try:
            cfg = _HMModelConfig(backbone="unet_s", pretrained=False, main_in_channels=3, cond_in_channels=1)
            hm_model = _PointHeatmapModel(cfg).to(hm_device)
            ckpt = torch.load(to_abs(args.heatmap_model), map_location="cpu")
            sd = ckpt.get("model", ckpt)
            hm_model.load_state_dict(sd, strict=False)
            hm_model.eval()
            print(f"Heatmap model loaded: {to_abs(args.heatmap_model)}")
        except Exception as e:
            print(f"[WARN] Failed to load heatmap model: {e}. Skip heatmap generation.")
            hm_model = None

    # 读取输入（优先按JSON数组解析；失败则按JSONL逐行解析）
    print(f"Processing input: {to_abs(args.input_jsonl)}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    # 解析输入记录列表 records: List[Dict]
    records: List[Dict[str, Any]] = []
    try:
        with open(to_abs(args.input_jsonl), 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    records = parsed
                else:
                    print("[ERROR] Input JSON root must be an array")
                    return 1
            else:
                # JSONL逐行
                for line_num, line in enumerate(content.splitlines(), 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                        num_errors += 1
                        continue
    except Exception as e:
        print(f"Error reading input: {e}")
        return 1

    for idx, obj in enumerate(records, 1):
        line_num = idx

        # 验证并提取点信息
        is_valid, points, labels = validate_and_extract_points(obj, line_num)
        if not is_valid:
            num_errors += 1
            continue

        image_path = to_abs(obj["image"]) or obj["image"]
        ### For Debug Only:
        # stem = Path(image_path).stem
        # if not stem == "BRATS_001_z0088":
        #     continue
        ### :For Debug Only

        # 检查图像文件是否存在
        if not os.path.isfile(image_path):
            print(f"[ERROR] Line {line_num}: Image file not found: {image_path}")
            num_errors += 1
            continue

        # 生成输出路径（子目录/<k>.png）
        output_path = get_mask_path_for_points(image_path, args.output_dir, points)

        # 检查是否跳过已存在的文件
        mask_existed = os.path.isfile(output_path)
        if args.skip_existing and mask_existed:
            _print_progress(num_processed, num_skipped + 1, num_errors,
                            f"skip line {line_num}: {Path(output_path).name} exists")
            # 此分支无现成bbox，且不应为不存在的日志补写空记录
            num_skipped += 1
            continue

        try:
            # 加载图像
            image = PILImage.open(to_abs(image_path)).convert("RGB")
            orig_w, orig_h = image.size

            # 若指定resize，则按比例缩放点坐标并对图像进行resize
            if args.resize:
                resize_w, resize_h = int(args.resize[0]), int(args.resize[1])
                scale_x = float(resize_w) / float(orig_w)
                scale_y = float(resize_h) / float(orig_h)
                points_resized = [(px * scale_x, py * scale_y) for (px, py) in points]
                image_for_pred = image.resize((resize_w, resize_h), PILImage.BILINEAR)
            else:
                points_resized = points
                image_for_pred = image
                scale_x = 1.0
                scale_y = 1.0

            # 运行SAM2预测（在处理后的分辨率上）
            mask, confidence = sam_wrapper.predict(image_for_pred, points_resized, labels)

            # 如果做了resize，则将mask缩放回原图尺寸，并在原图尺寸上计算bbox
            if args.resize:
                mask_binary = (mask > 0).astype(np.uint8)
                mask_orig_size = cv2.resize(mask_binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                x_min, y_min, x_max, y_max = calculate_bounding_box(mask_orig_size)
                # 保存缩放回原尺寸的mask
                save_mask_as_grayscale(mask_orig_size, output_path)
            else:
                # 未resize，直接使用原mask与bbox
                x_min, y_min, x_max, y_max = calculate_bounding_box(mask)
                save_mask_as_grayscale(mask, output_path)

            # 追加/更新该图像的日志（单一JSON对象，追加bbox并递增count）
            try:
                append_state_log(args.output_dir, image_path, (x_min, y_min, x_max, y_max))
            except Exception as e:
                print(f"[WARN] Failed to append state log for {image_path}: {e}")

            # 可选：生成并保存提示点概率热力图 heatmap-{k}.png
            # 语义：对于第 i 步（k=i），热力图来源于“图像 + 上一步的预测掩膜（i-1步；若i=0则为全零掩膜）”。
            if hm_model is not None:
                try:
                    # 选择热力图推理尺寸（优先 heatmap_size，其次使用 --resize，最后使用原始尺寸）
                    if args.heatmap_size is not None:
                        hm_w, hm_h = int(args.heatmap_size[0]), int(args.heatmap_size[1])
                    elif args.resize is not None:
                        hm_w, hm_h = int(args.resize[0]), int(args.resize[1])
                    else:
                        hm_w, hm_h = orig_w, orig_h

                    # 构造“上一时刻”的灰度掩膜（若 i>0 且存在文件，则直接读；否则为了稳健性直接用上一时刻的SAM预测）
                    k = get_mask_index_from_points(points)
                    prev_gray_pil: Optional[PILImage.Image]
                    if k <= 0:
                        prev_gray_pil = PILImage.new("L", (hm_w, hm_h), 0)
                    else:
                        mask_dir = get_mask_dir_for_image(image_path, args.output_dir)
                        prev_path = os.path.join(mask_dir, f"{k-1}.png")
                        if os.path.isfile(prev_path):
                            prev_gray_pil = PILImage.open(prev_path).convert("L").resize((hm_w, hm_h), PILImage.NEAREST)
                        else:
                            # 若上一步掩膜文件缺失，则用上一时刻点再次运行一次SAM以得到上一时刻掩膜
                            prev_pts = points_resized[:-1]
                            prev_labs = labels[:-1] if labels else None
                            prev_mask_np, _ = sam_wrapper.predict(
                                image_for_pred.resize((hm_w, hm_h), PILImage.BILINEAR), prev_pts, prev_labs
                            )
                            prev_gray = (prev_mask_np.astype(np.uint8) * 255)
                            prev_gray_pil = PILImage.fromarray(prev_gray, mode="L")

                    # 构造模型输入张量：RGB 与 灰度条件（范围与归一化与训练一致）
                    # 注意：这里独立于 SAM2 的 resize，确保热力图与指定 hm_w/h 对齐
                    from torchvision.transforms import functional as TF  # type: ignore
                    rgb_for_hm = image.resize((hm_w, hm_h), PILImage.BILINEAR)
                    rgb_t = TF.to_tensor(rgb_for_hm)
                    rgb_t = TF.normalize(rgb_t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)).unsqueeze(0)
                    gray_pil = prev_gray_pil if prev_gray_pil is not None else PILImage.new("L", (hm_w, hm_h), 0)
                    g_t = TF.to_tensor(gray_pil)
                    g_t = ((g_t - 0.5) / 0.5).unsqueeze(0)

                    with torch.no_grad():
                        logits, _ = hm_model(rgb_t.to(hm_device), g_t.to(hm_device))  # [1,1,H,W]
                        # 将 logits 转为概率热力图（按像素 softmax），与训练/可视化一致
                        b, c, Hh, Wh = logits.shape
                        p = _F.softmax(logits.view(b, -1) / max(float(args.heatmap_tau), 1e-6), dim=1)
                        prob = p.view(Hh, Wh).detach().cpu().float().numpy()

                    # 映射为彩色图并按原图尺寸保存，便于与 {k}.png 对齐查看
                    hm_img = _heatmap_to_pil(prob)
                    if (hm_w, hm_h) != (orig_w, orig_h):
                        hm_img = hm_img.resize((orig_w, orig_h), PILImage.BILINEAR)
                    heatmap_path = os.path.join(get_mask_dir_for_image(image_path, args.output_dir), f"heatmap-{k}.png")
                    os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
                    hm_img.save(heatmap_path)
                except Exception as e:
                    print(f"[WARN] Failed to save heatmap for {image_path}: {e}")

            _print_progress(num_processed + 1, num_skipped, num_errors,
                            f"ok line {line_num}: {Path(output_path).name} conf={confidence:.3f}")
            num_processed += 1

        except Exception as e:
            _print_progress(num_processed, num_skipped, num_errors + 1,
                            f"error line {line_num}: {Path(image_path).name}")
            num_errors += 1
            continue
    
    # 将输入数组复制到json_output并添加/覆盖 sam_masks_dir
    if args.json_output:
        try:
            # 若前面用JSONL解析得到records，则以records为准写出数组
            out_records = []
            for obj in records:
                if isinstance(obj, dict):
                    obj2 = dict(obj)
                    obj2["sam_masks_dir"] = args.output_dir
                    out_records.append(obj2)
            with open(args.json_output, 'w', encoding='utf-8') as jf:
                json.dump(out_records, jf, indent=2, ensure_ascii=False)
            print(f"Copied input to json_output with 'sam_masks_dir': {args.json_output}")
        except Exception as e:
            print(f"[ERROR] Failed to save json_output: {e}")
            return 1
    
    # 打印总结（先补换行清空进度行）
    print()
    print(f"Processing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    if args.json_output:
        print(f"  JSON output: {args.json_output}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

```

- 已为 `seg-rl/sam2_segment_from_points.py` 增加热力图输出功能：在生成 `{i}.png` 掩膜的同时，生成同目录的 `heatmap-{i}.png` 概率热力图。
- 新增参数：
  - `--heatmap_model`: 热力图预测模型 checkpoint 路径（开启热力图输出）
  - `--heatmap_tau`: 概率图 softmax 温度（默认 1.0）
  - `--heatmap_size WIDTH HEIGHT`: 热力图推理分辨率（默认优先用 `--heatmap_size`，否则复用 `--resize`，否则原图）

使用与正确性说明（要点）：
- 热力图来源与条件一致性：按你的描述，模型输入为“当前图像 + 上一步预测掩膜（第一步为全零）”。实现中：
  - 第 i 步的 `k = i`，若 `k == 0` 则用全零灰度掩膜；若 `k > 0`，优先读取上一步 `{k-1}.png`，否则兜底用当前已加载的 SAM2 预测器对上一步点再次预测得到上一掩膜。
- 模型与归一化对齐：热力图模型采用与训练一致的预处理：
  - RGB: to_tensor + ImageNet 均值方差归一化
  - 灰度条件: to_tensor 后做 `(g - 0.5) / 0.5`
  - 模型结构默认 `unet_s`，输入通道 `3+1` 与 `seg-rl/heatmap/model.py` 一致。
- 概率图计算：对 logits 做像素级 softmax（按 [B, H×W]），温度取 `--heatmap_tau`，再 reshape 为 [H, W] 即为概率热力图。与 `train.py` 可视化中的处理一致。
- 尺寸一致性：
  - 热力图内部先在 `(hm_w, hm_h)` 上计算；保存前若尺寸与原图不同，会 resize 回原图尺寸，保证与 `{i}.png` 对齐查看。
  - `(hm_w, hm_h)` 选择优先级：`--heatmap_size` > `--resize` > 原图。
- 可用性/健壮性：
  - 若未提供 `--heatmap_model` 或未安装 `seg-rl/heatmap` 依赖，则仅跳过热力图输出，掩膜仍正常生成。
  - 代码已加注释说明关键步骤与设计动机。

运行示例：
```bash
python seg-rl/sam2_segment_from_points.py \
  --input_jsonl datasets/seg_r1_md/Task01_BrainTumour/pred_points.jsonl \
  --output_dir outputs/braintumour/rl_pred_sam_masks \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda \
  --resize 512 512 \
  --heatmap_model pretrained/points_predictor-251001-160epochs.pt \
  --heatmap_tau 1.0 \
  --heatmap_size 512 512 \
  --skip_existing
```

你将得到：
- 掩膜：`outputs/.../<stem>/{i}.png`
- 概率热力图：`outputs/.../<stem>/heatmap-{i}.png`

简要论证：
- 模型输入与训练一致（RGB + cond 灰度，归一化一致），保证概率图语义正确。
- 概率来自像素级 softmax；所选提示点即对应概率最大值像素，符合“提示点选择规则”。
- 使用上一步掩膜作为条件，满足“第一步全零、其后使用上一掩膜”的递归定义。
- 尺寸回缩到原图，便于与 `{i}.png` 同目录直观比对。

如果你希望把热力图直接叠加到原图另存一张（而非单独的彩色 heatmap），我可以再补充一个 `--save_overlay_heatmap` 选项，输出 `overlay-heatmap-{i}.png`。

```shellscript
python seg-rl/sam2_segment_from_points.py \
  --input_jsonl datasets/seg_r1_md/Task01_BrainTumour/pred_points.jsonl \
  --output_dir outputs/braintumour/rl_pred_sam_masks \
  --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda \
  --resize 512 512 \
  --heatmap_model pretrained/points_predictor-251001-160epochs.pt \
  --heatmap_tau 1.0 \
  --heatmap_size 512 512 \
  --skip_existing
```

---

**User**

好，请修改Seg-R0/seg-rl/predict_points_and_sam.sh使其调用sam2_segment_from_points.py时用参数指示它需要做热力图可视化

---

**Cursor**

I'll open the shell script to locate where it invokes the Python segmenter and add the new heatmap flags consistently with existing variables.

I’m going to update the two invocations of the segmenter in the script to pass the new heatmap flags, using the same model path and resize as the script already uses.

```shellscript
#!/bin/bash
## Example: seg-rl/predict_points_and_sam.sh pred_251022 12 pretrained/points_predictor-251001-160epochs.pt cuda
# Check if correct number of arguments are provided
if [ $# -ne 4 ]; then
    echo "Usage: $0 <name_of_task> <length_points_sequence> <path_to_model> <device>"
    echo "Example: $0 \"pred_251015\" 12 \"outputs/braintumour/points_predictor-251001-50epochs.pt\" \"cuda\""
    exit 1
fi

# Get the arguments
task_name="$1"
n_points="$2"
path_to_model="$3"
device="$4"
# Validate that the second argument is a positive integer
if ! [[ "$n_points" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Length of points sequence must be a positive integer"
    echo "Usage: $0 <name_of_task> <length_points_sequence> <path_to_model> <device>"
    exit 1
fi

# Initialize conda and activate the seg-r1 environment
eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
conda activate seg-r1

# Set the Python path explicitly
# PYTHON_PATH="/opt/anaconda3/envs/seg-r1/bin/python"
PYTHON_PATH="/root/autodl-tmp/envs/seg-r1/bin/python"

# Verify we're using the correct Python
echo "Using python: $PYTHON_PATH"
echo "Using python version: $($PYTHON_PATH --version)"

# Check for existing files and folders
JSONL_FILE="outputs/braintumour/$task_name.jsonl"
SAM_MASKS_DIR="outputs/braintumour/${task_name}_sam_masks"
DBG_PRED_POINTS_DIR="outputs/braintumour/${task_name}_dbg_pred_points"

# Control whether to skip initial first-prompt generation when keeping existing JSONL
SKIP_FIRST=0

existing_files=()
if [ -f "$JSONL_FILE" ]; then
    existing_files+=("$JSONL_FILE")
fi
if [ -d "$SAM_MASKS_DIR" ]; then
    existing_files+=("$SAM_MASKS_DIR")
fi
if [ -d "$DBG_PRED_POINTS_DIR" ]; then
    existing_files+=("$DBG_PRED_POINTS_DIR")
fi

if [ ${#existing_files[@]} -gt 0 ]; then
    echo ""
    echo "Warning: The following files/folders already exist:"
    for file in "${existing_files[@]}"; do
        echo "  - $file"
    done
    echo ""
    read -p "Do you want to delete these existing files/folders and continue? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting existing files/folders..."
        for file in "${existing_files[@]}"; do
            if [ -f "$file" ]; then
                rm -f "$file"
                echo "  Deleted file: $file"
            elif [ -d "$file" ]; then
                rm -rf "$file"
                echo "  Deleted directory: $file"
            fi
        done
        echo "Cleanup completed. Proceeding with execution..."
    else
        echo "Keeping existing files/folders. Proceeding without deletion..."
        # If the JSONL already exists, skip the initial first-prompt generation block
        if [ -f "$JSONL_FILE" ]; then
            SKIP_FIRST=1
            echo "Detected existing JSONL ($JSONL_FILE). Will skip initial first-prompt generation."
        fi
    fi
fi

# Print the arguments
echo "Name of task: $task_name"
echo "Length of points sequence: $n_points"
echo "Path to model: $path_to_model"
if [ "$SKIP_FIRST" -eq 0 ]; then
    echo "Generate the first point prompt for each image:"

    $PYTHON_PATH -m seg-rl.heatmap.predict_next_point_from_model \
        --model_path $path_to_model \
        --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
        --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
        --output_json $JSONL_FILE

    $PYTHON_PATH seg-rl/sam2_segment_from_points.py \
        --input_jsonl $JSONL_FILE \
        --json_output $JSONL_FILE \
        --output_dir $SAM_MASKS_DIR \
        --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --device $device \
        --resize 512 512 \
        --skip_existing \
        --heatmap_model $path_to_model \
        --heatmap_tau 1.0 \
        --heatmap_size 512 512

    $PYTHON_PATH seg-rl/evaluation/eval_sam_masks.py --input_json $JSONL_FILE --num_prompts 1
else
    echo "Skipping initial first-prompt generation and starting from the iterative loop..."
fi
# Loop from 1 to (n_points - 1)
echo "Starting loop from 1 to $((n_points - 1)):"
for i in $(seq 1 $((n_points - 1))); do
    echo "Generate the $((i+1))th point prompt for each image"
    # Generate the next point prompt
    $PYTHON_PATH -m seg-rl.heatmap.predict_next_point_from_model \
        --model_path $path_to_model \
        --appendto_json $JSONL_FILE \
        --sam_dir $SAM_MASKS_DIR 
    
    $PYTHON_PATH seg-rl/sam2_segment_from_points.py \
      --input_jsonl $JSONL_FILE \
      --json_output $JSONL_FILE \
      --output_dir $SAM_MASKS_DIR \
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device $device \
      --resize 512 512 \
      --skip_existing \
      --heatmap_model $path_to_model \
      --heatmap_tau 1.0 \
      --heatmap_size 512 512

    $PYTHON_PATH seg-rl/evaluation/eval_sam_masks.py --input_json $JSONL_FILE --num_prompts $((i+1))
done

echo "Evaluate the performance of the model"

$PYTHON_PATH seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --debug_json $JSONL_FILE \
    --debug_output_dir $DBG_PRED_POINTS_DIR

$PYTHON_PATH seg-rl/sam2_automatic_evaluation.py \
    --input_jsonl $JSONL_FILE \
    --sam_masks_dir $SAM_MASKS_DIR \
    --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
    --device cuda


```

- 我已在 `seg-rl/predict_points_and_sam.sh` 中，为两处调用 `seg-rl/sam2_segment_from_points.py` 的位置追加了热力图可视化参数：
  - `--heatmap_model $path_to_model`
  - `--heatmap_tau 1.0`
  - `--heatmap_size 512 512`

这样运行脚本时会在每个样本的子目录下同时生成 `{i}.png` 掩膜和 `heatmap-{i}.png` 概率热力图。调整温度或分辨率时，仅需改动脚本中的对应值即可。

---

**User**

基于前面的工作，继续修改Seg-R0/seg-rl/annotator/gen_point_jsonl_from_masks.py的调试模式，使其在已有的绘制三个图像的基础上（_render_diff_panels_with_point），再添加第4幅图像，其内容为把热力图和当前提示点坐标画在一起，从而便于人观察所选择提示点与预测热力图的最热区域是否吻合以及吻合程度。并请在代码说明部分加入相应的调用示例（最好基于现在工程目录的实际情况来设计一个真实可用的调用示例）

---

**Cursor**

I'll locate the debug rendering function and the CLI flags in the annotator to add the fourth panel and update help text with a real runnable example.

```python
#!/usr/bin/env python3

"""
Generate JSONL supervision (image,x,y) by computing a "deep interior" point of
binary masks using the Euclidean Distance Transform (EDT) maximum.

Each output line is a JSON object like:
  {"image": "/abs/path/to/image.jpg", "gt_mask": "/abs/path/to/mask.png", "points": [[123.4,456.7]], "labels": [1]}

Coordinates are in pixel space of the original image/mask pair.
Designed for Seg-R1 heatmap-classification pretraining pipelines.

Example usage:
  # 1) 首次生成（计算首个最佳提示点）
  python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --images_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir  /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour.jsonl

  /opt/anaconda3/envs/seg-r1/bin/python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_jsonl datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251002.jsonl

  # 2) 追加模式（为每条记录计算后续的第2个及以后提示点）
  python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --appendto_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour.jsonl

  /opt/anaconda3/envs/seg-r1/bin/python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --appendto_jsonl datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251003.jsonl

  # 3) 调试模式（生成调试图像）
  # 若 sam2_segment_from_points.py 已生成 heatmap-{i}.png，本脚本会在调试图中追加第4幅：
  # “概率热力图 + 当前提示点”，用于核对提示点与热区的一致性。
  /opt/anaconda3/envs/seg-r1/bin/python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --debug_json outputs/braintumour/pred_251023.jsonl \
    --debug_output_dir outputs/braintumour/dbg_gen_points

Notes:
- Foreground is defined as any non-zero pixel in the mask.
- Target point is the pixel farthest from the background (EDT argmax), which
  is guaranteed to lie inside the foreground, robust to holes and thin parts.
- Samples with empty masks are skipped by default (override with --skip_empty false).
"""

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
# from PIL import Image, ImageDraw
from PIL import Image as _PIL
from scipy.ndimage import label, distance_transform_edt
import cv2  # type: ignore

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create/Append deepest-point prompts from masks using EDT")
    # 互斥：首写输出 vs 追加到已存在的JSON数组
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--output_jsonl", type=str, help="Path to write JSONL lines for first prompts")
    group.add_argument("--appendto_jsonl", type=str, help="Path to existing JSON array to append next prompts")
    group.add_argument("--debug_json", type=str, help="Path to JSON array to read for debug visualization only")

    # 首次生成模式需要的参数（在追加模式下将被忽略）
    p.add_argument("--images_dir", type=str, default=None, help="Directory containing images (e.g., JPG)")
    p.add_argument("--masks_dir", type=str, default=None, help="Directory containing PNG masks aligned to images")
    p.add_argument("--abs_paths", type=lambda s: s.lower() in {"1","true","yes","y"}, default=True,
                   help="Write absolute image paths (default: True)")
    p.add_argument("--skip_empty", type=lambda s: s.lower() in {"1","true","yes","y"}, default=True,
                   help="Skip samples where mask is empty (default: True)")
    p.add_argument("--image_exts", type=str, default=".jpg,.jpeg,.png",
                   help="Comma-separated acceptable image extensions (default: .jpg,.jpeg,.png)")
    p.add_argument("--viz_dir", type=str, default=None, help="Optional directory to save overlay visualizations")
    p.add_argument("--viz_alpha", type=float, default=0.35, help="Alpha for mask overlay (0-1)")
    p.add_argument("--viz_radius", type=int, default=4, help="Radius (pixels) for deepest-point circle")
    p.add_argument("--debug_output_dir", type=str, default=None,
                   help="If set, write per-sample debug images for diff component selection")
    return p.parse_args()


def list_files_with_exts(directory: str, allowed_exts: Tuple[str, ...]) -> List[str]:
    directory = to_abs(directory) or directory
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    out: List[str] = []
    for name in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, name)):
            continue
        if any(name.lower().endswith(ext) for ext in allowed_exts):
            out.append(name)
    return sorted(out)


def stem(path: str) -> str:
    name = os.path.basename(path)
    # Remove common two-part extensions like .nii.gz by iterative splitext
    base, ext = os.path.splitext(name)
    return base


def to_abs(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.abspath(path) if not os.path.isabs(path) else path


def _find_latest_mask_path(sam_masks_dir: str, image_stem: str, current_points_len: int) -> Optional[str]:
    """返回该图像在sam_masks_dir下的最新预测mask路径。
    优先尝试 sam_masks_dir/<stem>/<current_points_len-1>.png；若不存在，
    则在该目录下寻找最大数字文件名的png。
    """
    candidate_dir = os.path.join(sam_masks_dir, image_stem)
    if not os.path.isdir(candidate_dir):
        return None
    k = max(0, int(current_points_len - 1))
    preferred = os.path.join(candidate_dir, f"{k}.png")
    if os.path.isfile(preferred):
        return preferred
    # fallback: find max numeric png
    best_idx = -1
    best_path = None
    for name in os.listdir(candidate_dir):
        if not name.lower().endswith(".png"):
            continue
        s = os.path.splitext(name)[0]
        try:
            idx = int(s)
        except Exception:
            continue
        if idx > best_idx:
            best_idx = idx
            best_path = os.path.join(candidate_dir, name)
    return best_path

from scipy import ndimage as ndi

def farthest_point_from_boundary(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    # mask: 二值数组，区域为 True/1，背景为 False/0
    mask = (mask > 0)
    dist = ndi.distance_transform_edt(mask)  # 区域内每点到背景的欧氏距离
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    # r = float(dist[y, x])  # 到边界的最大最小距离
    # return (y, x), r
    return (float(x), float(y))

def compute_centroid(mask_u8: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Compute a robust interior point (x,y) for the foreground mask as the
    Euclidean Distance Transform (EDT) maximum (i.e., the pixel farthest from
    the background). Returns None if the mask has no foreground.
    """
    if mask_u8.ndim == 3:
        # If mask has channels, convert to single channel by any non-zero across channels
        mask_u8 = (mask_u8.any(axis=2)).astype(np.uint8) * 255
    m = (mask_u8 > 0).astype(np.uint8)
    # Preprocess: keep largest connected component and fill holes for robustness
    try:
        from scipy import ndimage as ndi  # type: ignore
        if m.sum() > 0:
            labels, nlab = ndi.label(m, structure=np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.uint8))
            if nlab > 1:
                sizes = ndi.sum(m, labels, index=range(1, nlab + 1))
                largest_idx = int(np.argmax(sizes)) + 1
                m = (labels == largest_idx).astype(np.uint8)
            # Fill holes
            m = ndi.binary_fill_holes(m > 0).astype(np.uint8)
    except Exception:
        try:
            import cv2  # type: ignore
            if m.sum() > 0:
                # Connected components with stats; background is label 0
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
                if num_labels > 2:
                    # Find largest non-background component by area (stats[:, cv2.CC_STAT_AREA])
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    largest_idx = 1 + int(np.argmax(areas))
                    m = (labels == largest_idx).astype(np.uint8)
                elif num_labels == 2:
                    # Only one foreground component
                    m = (labels == 1).astype(np.uint8)
                # Fill holes via flood fill on background
                h, w = m.shape
                inv = (1 - m) * 255
                ff = inv.copy()
                mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
                cv2.floodFill(ff, mask, seedPoint=(0, 0), newVal=128)
                holes = (ff != 128)  # interior background
                m = ((1 - inv/255).astype(np.uint8) | holes.astype(np.uint8))
        except Exception:
            # If no morphology libs, proceed without preprocessing
            pass
    if m.sum() == 0:
        return None
    # Prefer SciPy's true Euclidean distance; fallback to OpenCV if SciPy unavailable
    dist = None
    try:
        from scipy import ndimage as ndi  # type: ignore
        dist = ndi.distance_transform_edt(m)
    except Exception:
        try:
            import cv2  # type: ignore
            dist = cv2.distanceTransform(m, distanceType=cv2.DIST_L2, maskSize=5)
        except Exception:
            # As a last resort, use cityblock (L1) via iterative distance on uint8
            # This is a coarse fallback and less accurate than EDT.
            # Here we approximate by growing distance in four-neighborhood.
            h, w = m.shape
            dist = np.zeros_like(m, dtype=np.float32)
            # Forward pass
            for y in range(h):
                for x in range(w):
                    if m[y, x] == 0:
                        continue
                    v = dist[y, x]
                    if y > 0:
                        v = max(v, dist[y - 1, x] + 1.0)
                    if x > 0:
                        v = max(v, dist[y, x - 1] + 1.0)
                    dist[y, x] = v
            # Backward pass
            for y in range(h - 1, -1, -1):
                for x in range(w - 1, -1, -1):
                    if m[y, x] == 0:
                        continue
                    v = dist[y, x]
                    if y + 1 < h:
                        v = max(v, dist[y + 1, x] + 1.0)
                    if x + 1 < w:
                        v = max(v, dist[y, x + 1] + 1.0)
                    dist[y, x] = v
    # Argmax returns (y, x)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return (float(x), float(y))

def largest_diff_component_representative(A, B, connectivity=2, min_area=0, dbg_out_path: Optional[str] = None):
    # A, B: 2D uint8 arrays, values {0,1} 或 {0,255}
    A = (A > 0).astype(np.uint8)
    B = (B > 0).astype(np.uint8)
    ### C = A.astype(np.int16) - B.astype(np.int16)  # {-1, 0, +1}
    C = A.astype(np.int8) - B.astype(np.int8)  # {-1, 0, +1}

    regions = []
    # 8 邻域结构元（connectivity=2 表示 8 邻域；=1 表示 4 邻域）
    st = np.ones((3,3), dtype=np.uint8) if connectivity == 2 else np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)

    for sign, mask in ((1, C > 0), (-1, C < 0)):
        if not mask.any():
            continue
        lab, n = ndi.label(mask, structure=st)
        if n == 0:
            continue
        areas = np.bincount(lab.ravel())[1:]  # 忽略背景0
        if min_area > 0:
            keep_ids = np.where(areas >= min_area)[0] + 1
            if keep_ids.size == 0:
                continue
            mask2 = np.isin(lab, keep_ids)
            lab, n = ndi.label(mask2, structure=st)
            if n == 0:
                continue
            areas = np.bincount(lab.ravel())[1:]

        k = np.argmax(areas) + 1
        comp = (lab == k)
        regions.append((sign, int(areas[k-1]), comp))

    if not regions:
        return None  # 无非零差异

    # 选面积最大的分量；若需以“最深”优先，可在 key 中加入 max EDT 作为次序
    sign, area, comp = max(regions, key=lambda t: t[1])

    # centroid = farthest_point_from_boundary(comp)
    # # centroid = compute_centroid(comp)
    # if centroid is None:
    #     return None
    D = distance_transform_edt(comp)
    (centroidY, centroidX) = np.unravel_index(np.argmax(D), D.shape)

    # 调试可视化（改用统一的渲染函数，保持风格一致）
    if dbg_out_path:
        try:
            _render_diff_panels_with_point(
                A=A,
                B=B,
                point_xy=(float(centroidX), float(centroidY)),
                label=1 if int(sign) > 0 else 0,
                dbg_out_path=dbg_out_path,
                connectivity=connectivity,
            )
        except Exception:
            pass

    return dict(sign=sign, area=area, y=int(centroidY), x=int(centroidX), max_radius=float(D[centroidY, centroidX]))


def _render_diff_panels_with_point(
    A: np.ndarray,
    B: np.ndarray,
    point_xy: Tuple[float, float],
    label: int,
    dbg_out_path: Optional[str],
    connectivity: int = 2,
    base_image_bgr: Optional[np.ndarray] = None,
) -> None:
    """Render 3-panel debug image reusing the visualization style in
    largest_diff_component_representative, but mark a provided point instead of
    the recomputed centroid.
    - Panel 1: A (green) vs B (red) overlay
    - Panel 2: signed diff C (positive green, negative red)
    - Panel 3: connected components colored, with marker at point_xy
    """
    rng = np.random.default_rng(12345)
    try:
        if dbg_out_path is None:
            return
        h, w = A.shape[:2]
        # Panel 1: A/B overlay (A=green, B=red), optionally blended on base image
        overlay1 = np.zeros((h, w, 3), dtype=np.uint8)
        overlay1[..., 1] = (A > 0).astype(np.uint8) * 255  # G
        overlay1[..., 2] = (B > 0).astype(np.uint8) * 255  # R
        if base_image_bgr is not None:
            base = base_image_bgr
            if base.ndim == 2:
                base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            if base.shape[:2] != (h, w):
                base = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
            panel1 = cv2.addWeighted(base, 1.0, overlay1, 0.6, 0.0)
        else:
            bg = np.zeros_like(overlay1)
            panel1 = cv2.addWeighted(bg, 1.0, overlay1, 0.6, 0.0)

        # Panel 2: signed diff C (positive green, negative red)
        C = A.astype(np.int16) - B.astype(np.int16)
        panel2 = np.zeros((h, w, 3), dtype=np.uint8)
        pos = (C > 0)
        neg = (C < 0)
        panel2[pos, 1] = 255
        panel2[neg, 2] = 255

        # Panel 3: connected components colored + provided point marker
        panel3 = np.zeros((h, w, 3), dtype=np.uint8)
        st = np.ones((3,3), dtype=np.uint8) if connectivity == 2 else np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
        lab, n = ndi.label(C, structure=st)
        if n != 0:
            colors = (rng.integers(0, 256, size=(n, 3))).astype(np.uint8)
            colors = np.clip(colors + 100, 0, 255)
            for i in range(1, n + 1):
                panel3[lab == i] = colors[i - 1]

        # draw marker at provided (x,y): foreground -> caret '^', background -> 'X'
        cx = int(round(float(point_xy[0])))
        cy = int(round(float(point_xy[1])))
        def _draw_cross(img, x, y, size=6, color=(255,255,255), thickness=2):
            cv2.line(img, (x - size, y - size), (x + size, y + size), (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, (x - size, y + size), (x + size, y - size), (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, (x - size, y - size), (x + size, y + size), color, thickness, lineType=cv2.LINE_AA)
            cv2.line(img, (x - size, y + size), (x + size, y - size), color, thickness, lineType=cv2.LINE_AA)
        def _draw_caret(img, x, y, size=6, color=(255,255,255), thickness=2):
            p_top = (int(x), int(y - size))
            p_left = (int(x - size), int(y + size))
            p_right = (int(x + size), int(y + size))
            cv2.line(img, p_left, p_top, (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, p_right, p_top, (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, p_left, p_top, color, thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p_right, p_top, color, thickness, lineType=cv2.LINE_AA)
        if int(label) > 0:
            _draw_caret(panel3, cx, cy, size=6, color=(255,255,255), thickness=2)
        else:
            _draw_cross(panel3, cx, cy, size=6, color=(255,255,255), thickness=2)

        # Compute and draw metrics comparing B (pred) vs A (gt) on panel1
        def _compute_metrics(gt_u8: np.ndarray, pr_u8: np.ndarray) -> dict:
            gt = (gt_u8 > 0)
            pr = (pr_u8 > 0)
            tp = int(np.logical_and(gt, pr).sum())
            fp = int(np.logical_and(~gt, pr).sum())
            fn = int(np.logical_and(gt, ~pr).sum())
            denom_dice = 2 * tp + fp + fn
            dice = (2 * tp / denom_dice) if denom_dice > 0 else 1.0
            denom_iou = tp + fp + fn
            iou = (tp / denom_iou) if denom_iou > 0 else 1.0
            precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
            recall = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            # Lightweight S-measure implementation adapted for binary masks
            def _s_measure_binary(pred: np.ndarray, gt: np.ndarray) -> float:
                pred_f = pred.astype(np.float64)
                gt_f = gt.astype(np.float64)
                y_mean = gt_f.mean()
                if y_mean == 0:
                    return float(1.0 - pred_f.mean())
                if y_mean == 1:
                    return float(pred_f.mean())

                def s_object(p: np.ndarray, g: np.ndarray) -> float:
                    vals = p[g == 1]
                    if vals.size == 0:
                        return 0.0
                    mx = float(vals.mean())
                    sx = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
                    return float(2 * mx / (mx * mx + 1 + sx + np.spacing(1)))

                def ssim(a: np.ndarray, b: np.ndarray) -> float:
                    h_, w_ = a.shape
                    N = h_ * w_
                    if N <= 1:
                        return 1.0
                    mx = float(a.mean()); my = float(b.mean())
                    sx = float(((a - mx) ** 2).sum() / (N - 1))
                    sy = float(((b - my) ** 2).sum() / (N - 1))
                    sxy = float(((a - mx) * (b - my)).sum() / (N - 1))
                    alpha = 4 * mx * my * sxy
                    beta = (mx * mx + my * my) * (sx + sy)
                    if alpha != 0:
                        return float(alpha / (beta + np.spacing(1)))
                    if alpha == 0 and beta == 0:
                        return 1.0
                    return 0.0

                coords = np.argwhere(gt_f == 1)
                h_, w_ = gt_f.shape
                if coords.size == 0:
                    cx, cy = int(round(w_ / 2)), int(round(h_ / 2))
                else:
                    cy, cx = coords.mean(axis=0).round().astype(int).tolist()
                cx = int(cx) + 1; cy = int(cy) + 1
                gt_LT = gt_f[0:cy, 0:cx]; gt_RT = gt_f[0:cy, cx:w_]
                gt_LB = gt_f[cy:h_, 0:cx]; gt_RB = gt_f[cy:h_, cx:w_]
                pr_LT = pred_f[0:cy, 0:cx]; pr_RT = pred_f[0:cy, cx:w_]
                pr_LB = pred_f[cy:h_, 0:cx]; pr_RB = pred_f[cy:h_, cx:w_]
                area = float(h_ * w_)
                w1 = (cx * cy) / area
                w2 = (cy * (w_ - cx)) / area
                w3 = ((h_ - cy) * cx) / area
                w4 = 1.0 - w1 - w2 - w3
                region_score = (
                    w1 * ssim(pr_LT, gt_LT) +
                    w2 * ssim(pr_RT, gt_RT) +
                    w3 * ssim(pr_LB, gt_LB) +
                    w4 * ssim(pr_RB, gt_RB)
                )
                alpha = 0.5
                return float(max(0.0, alpha * s_object(pred_f, gt_f) + (1 - alpha) * region_score))

            s_measure = _s_measure_binary(pr, gt)
            return {
                "DICE": dice,
                "IOU": iou,
                "PRECISION": precision,
                "RECALL": recall,
                "S_MEASURE": s_measure,
                "F1_SCORE": f1,
            }

        def _draw_metrics_footer(img: np.ndarray, metrics: dict, keys: List[str]) -> None:
            parts: List[str] = []
            for k in keys:
                if k in metrics:
                    parts.append(f"{k}: {metrics[k]:.3f}")
            if not parts:
                return
            text = "  ".join(parts)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 1
            max_width = img.shape[1] - 16
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            if tw > max_width and tw > 0:
                scale = max(0.3, scale * (max_width / tw))
                (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            x = 8
            y = img.shape[0] - 6
            # stroke for readability
            cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), 3, lineType=cv2.LINE_AA)
            cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        try:
            m = _compute_metrics(A, B)
            _draw_metrics_footer(panel1, m, ["DICE", "IOU"])
            _draw_metrics_footer(panel2, m, ["PRECISION", "RECALL"])
            _draw_metrics_footer(panel3, m, ["S_MEASURE", "F1_SCORE"])
        except Exception:
            pass

        # Titles
        def _add_title(img: np.ndarray, text: str) -> None:
            band_h = max(20, min(48, img.shape[0] // 20))
            roi = img[0:band_h, :, :]
            overlay = roi.copy()
            cv2.rectangle(overlay, (0, 0), (img.shape[1] - 1, band_h - 1), (0, 0, 0), thickness=-1)
            cv2.addWeighted(overlay, 0.5, roi, 0.5, 0, dst=roi)
            org = (8, band_h - 6)
            font_scale = 0.4
            font_thickness = 1
            # Truncate text to fit within image width
            max_width = img.shape[1] - 16  # Leave some padding
            (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            if text_width > max_width:
                # Binary search to find the right truncation point
                ellipsis = "..."
                for i in range(len(text), 0, -1):
                    truncated = text[:i] + ellipsis
                    (trunc_width, _), _ = cv2.getTextSize(truncated, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    if trunc_width <= max_width:
                        text = truncated
                        break
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, lineType=cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)
        _add_title(panel1, "Overlay A (green) vs B (red)")
        _add_title(panel2, "Signed diff C: + (green), - (red)")
        _add_title(panel3, "Components + selected point")

        canvas = np.concatenate([panel1, panel2, panel3], axis=1)
        os.makedirs(os.path.dirname(dbg_out_path) or ".", exist_ok=True)
        _PIL.fromarray(canvas[..., ::-1]).save(dbg_out_path)
    except Exception:
        pass

def find_corresponding_image(
    images_dir: str,
    sample_stem: str,
    allowed_exts: Tuple[str, ...],
) -> Optional[str]:
    images_dir_abs = to_abs(images_dir)
    for ext in allowed_exts:
        candidate = os.path.join(images_dir_abs, sample_stem + ext)
        if os.path.isfile(candidate):
            return candidate
    # Fallback: scan directory and match by stem (handles mixed-case extensions or uncommon extensions)
    for name in os.listdir(images_dir_abs):
        if stem(name).lower() == sample_stem.lower():
            return os.path.join(images_dir_abs, name)
    return None

def calculate_next_point(gt_mask_path, sam_masks_dir, current_points_len, dbg_out_dir: Optional[str] = None):
    # 读取gt与pred
    try:
        gt_u8 = np.array(_PIL.open(to_abs(gt_mask_path)), dtype=np.uint8)
        if current_points_len > 0:
            last_mask_path = _find_latest_mask_path(sam_masks_dir, stem(gt_mask_path), current_points_len)
            if not last_mask_path or not os.path.isfile(last_mask_path):
                print(f"[WARN] Skip: last mask not found under {sam_masks_dir}")
                return None, None, None
            pred_u8 = np.array(_PIL.open(to_abs(last_mask_path)), dtype=np.uint8)
        else:
            pred_u8 = np.zeros_like(gt_u8)
    except Exception as e:
        print(f"[WARN] Skip: failed to read masks ({e})")
        return None, None, None

    # 若形状不同，将pred按最近邻缩放到gt尺寸
    h_g, w_g = gt_u8.shape[:2]
    h_p, w_p = pred_u8.shape[:2]
    if (h_g, w_g) != (h_p, w_p):
        try:
            # from PIL import Image as _PIL
            pred_u8 = np.array(_PIL.fromarray(pred_u8).resize((w_g, h_g), resample=_PIL.NEAREST), dtype=np.uint8)
        except Exception:
            pass

    ### For Debug Only:
    # if stem(gt_mask_path) == "BRATS_001_z0088":
    #     print("#Debug#")
    ### :For Debug Only

    # 计算差异的最大连通分量代表点
    dbg_path = None
    if dbg_out_dir:
        try:
            subdir = os.path.join(dbg_out_dir, stem(gt_mask_path))
            os.makedirs(subdir, exist_ok=True)
            dbg_path = os.path.join(subdir, f"dbg_{max(0, int(current_points_len))}.png")
        except Exception:
            dbg_path = None
    rep = largest_diff_component_representative(gt_u8, pred_u8, connectivity=2, min_area=0, dbg_out_path=dbg_path)
    if rep is None:
        print(f"[INFO] no diff component; skip append")
        return None, None, None
    x_next = int(rep["x"])  # 列
    y_next = int(rep["y"])  # 行
    label_next = 1 if int(rep["sign"]) > 0 else 0
    return x_next, y_next, label_next

def main() -> None:
    global _PIL
    args = parse_args()

    # Debug-only mode: read JSON, generate debug images but do not write JSON
    if args.debug_json:
        if not args.debug_output_dir:
            raise RuntimeError("--debug_output_dir is required when --debug_json is set")
        json_path = args.debug_json
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Debug JSON not found: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        try:
            arr = json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Debug JSON must be a JSON array: {e}")
        if not isinstance(arr, list):
            raise RuntimeError("Debug JSON root must be a JSON array")

        num_generated = 0
        for rec in arr:
            if not isinstance(rec, dict):
                continue

            image_path = rec.get("image")
            gt_mask_path = rec.get("gt_mask")
            sam_masks_dir = rec.get("sam_masks_dir")
            if not (image_path and sam_masks_dir):
                continue
            sample_stem = stem(image_path)
            ### For Debug Only:
            # if not sample_stem == "BRATS_001_z0088":
            #     continue
            ### :For Debug Only
            try:
                subdir = os.path.join(args.debug_output_dir, sample_stem)
                os.makedirs(subdir, exist_ok=True)
            except Exception:
                continue

            points = rec.get("points", [])
            labels = rec.get("labels", [])
            if not (isinstance(points, list) and isinstance(labels, list)):
                continue
            if len(points) != len(labels):
                continue

            gt_mask_u8 = np.array(_PIL.open(to_abs(gt_mask_path)), dtype=np.uint8)
            # Load base image if exists
            base_img_bgr = None
            try:
                img_arr = np.array(_PIL.open(to_abs(image_path)).convert("RGB"))
                base_img_bgr = img_arr[:, :, ::-1]
            except Exception:
                base_img_bgr = None
            # Load gt as all-zero array to preserve panel structure; we visualize differences of predicted masks only
            for i, (pt, lb) in enumerate(zip(points, labels)):
                # For step i, compare masks of ground truth and step i-1 (i==0 uses empty prev)
                if i == 0:
                    prev = np.zeros_like(gt_mask_u8, dtype=np.uint8)
                else:
                    try:
                        prev_path = os.path.join(sam_masks_dir, sample_stem, f"{i-1}.png")
                        prev = np.array(_PIL.open(to_abs(prev_path)).convert("L"), dtype=np.uint8) if os.path.isfile(prev_path) else np.zeros_like(gt_mask_u8, dtype=np.uint8)
                    except Exception:
                        prev = np.zeros_like(gt_mask_u8, dtype=np.uint8)
                # Render panels with provided point
                dbg_path = os.path.join(subdir, f"dbg_{i}.png")
                _render_diff_panels_with_point(
                    gt_mask_u8,
                    prev,
                    (float(pt[0]), float(pt[1])),
                    int(lb),
                    dbg_path,
                    base_image_bgr=base_img_bgr,
                )
                num_generated += 1
        print(f"Generated debug images for {num_generated} steps into {args.debug_output_dir}")
        return

    # 追加模式：读取已有JSON数组，计算下一提示点并写回
    if args.appendto_jsonl:
        json_path = args.appendto_jsonl
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Append target not found: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        try:
            arr = json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Append target must be a JSON array: {e}")
        if not isinstance(arr, list):
            raise RuntimeError("Append target root must be a JSON array")

        updated = 0
        for idx, rec in enumerate(arr):
            if not isinstance(rec, dict):
                continue
            image_path = rec.get("image")
            gt_mask_path = rec.get("gt_mask")
            sam_masks_dir = rec.get("sam_masks_dir")
            points_list = rec.get("points", [])
            labels_list = rec.get("labels", [])

            if not (image_path and gt_mask_path and sam_masks_dir):
                print(f"[WARN] Skip idx {idx}: missing fields")
                continue

            x_next, y_next, label_next = calculate_next_point(
                gt_mask_path,
                sam_masks_dir,
                len(points_list),
                dbg_out_dir=args.debug_output_dir,
            )
            if x_next is None or y_next is None or label_next is None:
                continue

            # 追加points与labels
            if not isinstance(points_list, list):
                points_list = []
            if not isinstance(labels_list, list):
                labels_list = []
            points_list.append([x_next, y_next])
            labels_list.append(int(label_next))
            rec["points"] = points_list
            rec["labels"] = labels_list
            updated += 1

        # 回写同一文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(arr, f, indent=2, ensure_ascii=False)
        print(f"Appended next prompts to {json_path} (updated {updated} items)")
        return

    # 否则：首写模式，生成首个提示点（与之前逻辑一致）
    if not args.images_dir or not args.masks_dir or not args.output_jsonl:
        raise RuntimeError("images_dir, masks_dir and output_jsonl are required for initial generation mode")

    images_dir = args.images_dir
    masks_dir = args.masks_dir
    output_jsonl = args.output_jsonl
    allowed_image_exts = tuple(e.strip().lower() for e in args.image_exts.split(",") if e.strip())

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    # Visualization output disabled
    # if args.viz_dir:
    #     os.makedirs(args.viz_dir, exist_ok=True)

    # Gather masks as the source of truth
    mask_files = list_files_with_exts(masks_dir, allowed_exts=(".png",))
    if len(mask_files) == 0:
        raise RuntimeError(f"No mask PNG files found under {masks_dir}")

    num_written = 0
    num_skipped_empty = 0
    num_missing_images = 0
    num_size_mismatch = 0

    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for mask_name in mask_files:
            sample_stem = stem(mask_name)
            mask_path = os.path.join(masks_dir, mask_name)
            
            # Load mask
            try:
                mask_img = _PIL.open(to_abs(mask_path))
                mask_u8 = np.array(mask_img, dtype=np.uint8)
            except Exception as e:
                print(f"[WARN] Skipping unreadable mask: {mask_path} ({e})")
                continue
            
            x_next, y_next, label_next = calculate_next_point(
                mask_path,
                None,
                0,
                dbg_out_dir=args.debug_output_dir,
            )
            if x_next is None or y_next is None or label_next is None:
                if args.skip_empty:
                    num_skipped_empty += 1
                    continue
                else:
                    # For empty masks and skip_empty=False, place centroid at image center
                    h, w = mask_u8.shape[:2]
                    centroid = (float(w - 1) / 2.0, float(h - 1) / 2.0)
            centroid = (x_next, y_next)

            # Find corresponding image
            img_path = find_corresponding_image(images_dir, sample_stem, allowed_image_exts)
            if img_path is None:
                num_missing_images += 1
                print(f"[WARN] Missing image for mask stem '{sample_stem}' in {images_dir}")
                continue
           
            image_field = os.path.abspath(img_path) if args.abs_paths else img_path
            mask_field = os.path.abspath(mask_path) if args.abs_paths else mask_path
            record = {"image": image_field, "gt_mask": mask_field, "points": [[float(centroid[0]), float(centroid[1])]], "labels": [1]}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1

            # Visualization
            if False and args.viz_dir:
                try:
                    # Load image RGB
                    with _PIL.open(to_abs(img_path)) as im_rgb:
                        im_rgb = im_rgb.convert("RGB")
                        w_im, h_im = im_rgb.size
                        
                        # Ensure mask size matches visualization image
                        m = mask_u8
                        if (m.shape[1], m.shape[0]) != (w_im, h_im):
                            # resize nearest for mask
                            m_img = Image.fromarray(m)
                            m_img = m_img.resize((w_im, h_im), resample=Image.NEAREST)
                            m = np.array(m_img, dtype=np.uint8)
                        
                        # Create colored mask overlay
                        overlay = Image.new("RGBA", (w_im, h_im), (0, 0, 0, 0))
                        mask_alpha = int(max(0.0, min(1.0, args.viz_alpha)) * 255)
                        color = (255, 0, 0, mask_alpha)
                        # Build alpha mask where foreground>0
                        fg = (m > 0).astype(np.uint8) * mask_alpha
                        alpha_img = Image.fromarray(fg, mode="L")
                        color_img = Image.new("RGBA", (w_im, h_im), color)
                        overlay.paste(color_img, (0, 0), mask=alpha_img)
                        
                        # Compose overlay on image
                        comp = im_rgb.convert("RGBA")
                        comp = Image.alpha_composite(comp, overlay)
                        
                        # Draw the deepest point
                        draw = ImageDraw.Draw(comp)
                        px, py = float(centroid[0]), float(centroid[1])
                        r = max(1, int(args.viz_radius))
                        draw.ellipse((px - r, py - r, px + r, py + r), outline=(0, 255, 0, 255), width=2)
                        
                        # Save
                        out_name = f"{sample_stem}.png"
                        comp.convert("RGB").save(os.path.join(args.viz_dir, out_name))
                except Exception as e:
                    print(f"[WARN] Visualization failed for {img_path}: {e}")

    print(f"Wrote {num_written} samples to {output_jsonl}")
    if num_skipped_empty:
        print(f"Skipped empty masks: {num_skipped_empty}")
    if num_missing_images:
        print(f"Masks with missing images: {num_missing_images}")
    if num_size_mismatch:
        print(f"Image/Mask size mismatches adjusted: {num_size_mismatch}")


if __name__ == "__main__":
    main()



```

```python
#!/usr/bin/env python3

"""
Generate JSONL supervision (image,x,y) by computing a "deep interior" point of
binary masks using the Euclidean Distance Transform (EDT) maximum.

Each output line is a JSON object like:
  {"image": "/abs/path/to/image.jpg", "gt_mask": "/abs/path/to/mask.png", "points": [[123.4,456.7]], "labels": [1]}

Coordinates are in pixel space of the original image/mask pair.
Designed for Seg-R1 heatmap-classification pretraining pipelines.

Example usage:
  # 1) 首次生成（计算首个最佳提示点）
  python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --images_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir  /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour.jsonl

  /opt/anaconda3/envs/seg-r1/bin/python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_jsonl datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251002.jsonl

  # 2) 追加模式（为每条记录计算后续的第2个及以后提示点）
  python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --appendto_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour.jsonl

  /opt/anaconda3/envs/seg-r1/bin/python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --appendto_jsonl datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251003.jsonl

  # 3) 调试模式（生成调试图像）
  # 若 sam2_segment_from_points.py 已生成 heatmap-{i}.png，本脚本会在调试图中追加第4幅：
  # “概率热力图 + 当前提示点”，用于核对提示点与热区的一致性。
  /opt/anaconda3/envs/seg-r1/bin/python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --debug_json outputs/braintumour/pred_251023.jsonl \
    --debug_output_dir outputs/braintumour/dbg_gen_points

Notes:
- Foreground is defined as any non-zero pixel in the mask.
- Target point is the pixel farthest from the background (EDT argmax), which
  is guaranteed to lie inside the foreground, robust to holes and thin parts.
- Samples with empty masks are skipped by default (override with --skip_empty false).
"""

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
# from PIL import Image, ImageDraw
from PIL import Image as _PIL
from scipy.ndimage import label, distance_transform_edt
import cv2  # type: ignore

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create/Append deepest-point prompts from masks using EDT")
    # 互斥：首写输出 vs 追加到已存在的JSON数组
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--output_jsonl", type=str, help="Path to write JSONL lines for first prompts")
    group.add_argument("--appendto_jsonl", type=str, help="Path to existing JSON array to append next prompts")
    group.add_argument("--debug_json", type=str, help="Path to JSON array to read for debug visualization only")

    # 首次生成模式需要的参数（在追加模式下将被忽略）
    p.add_argument("--images_dir", type=str, default=None, help="Directory containing images (e.g., JPG)")
    p.add_argument("--masks_dir", type=str, default=None, help="Directory containing PNG masks aligned to images")
    p.add_argument("--abs_paths", type=lambda s: s.lower() in {"1","true","yes","y"}, default=True,
                   help="Write absolute image paths (default: True)")
    p.add_argument("--skip_empty", type=lambda s: s.lower() in {"1","true","yes","y"}, default=True,
                   help="Skip samples where mask is empty (default: True)")
    p.add_argument("--image_exts", type=str, default=".jpg,.jpeg,.png",
                   help="Comma-separated acceptable image extensions (default: .jpg,.jpeg,.png)")
    p.add_argument("--viz_dir", type=str, default=None, help="Optional directory to save overlay visualizations")
    p.add_argument("--viz_alpha", type=float, default=0.35, help="Alpha for mask overlay (0-1)")
    p.add_argument("--viz_radius", type=int, default=4, help="Radius (pixels) for deepest-point circle")
    p.add_argument("--debug_output_dir", type=str, default=None,
                   help="If set, write per-sample debug images for diff component selection")
    return p.parse_args()


def list_files_with_exts(directory: str, allowed_exts: Tuple[str, ...]) -> List[str]:
    directory = to_abs(directory) or directory
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    out: List[str] = []
    for name in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, name)):
            continue
        if any(name.lower().endswith(ext) for ext in allowed_exts):
            out.append(name)
    return sorted(out)


def stem(path: str) -> str:
    name = os.path.basename(path)
    # Remove common two-part extensions like .nii.gz by iterative splitext
    base, ext = os.path.splitext(name)
    return base


def to_abs(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.abspath(path) if not os.path.isabs(path) else path


def _find_latest_mask_path(sam_masks_dir: str, image_stem: str, current_points_len: int) -> Optional[str]:
    """返回该图像在sam_masks_dir下的最新预测mask路径。
    优先尝试 sam_masks_dir/<stem>/<current_points_len-1>.png；若不存在，
    则在该目录下寻找最大数字文件名的png。
    """
    candidate_dir = os.path.join(sam_masks_dir, image_stem)
    if not os.path.isdir(candidate_dir):
        return None
    k = max(0, int(current_points_len - 1))
    preferred = os.path.join(candidate_dir, f"{k}.png")
    if os.path.isfile(preferred):
        return preferred
    # fallback: find max numeric png
    best_idx = -1
    best_path = None
    for name in os.listdir(candidate_dir):
        if not name.lower().endswith(".png"):
            continue
        s = os.path.splitext(name)[0]
        try:
            idx = int(s)
        except Exception:
            continue
        if idx > best_idx:
            best_idx = idx
            best_path = os.path.join(candidate_dir, name)
    return best_path

from scipy import ndimage as ndi

def farthest_point_from_boundary(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    # mask: 二值数组，区域为 True/1，背景为 False/0
    mask = (mask > 0)
    dist = ndi.distance_transform_edt(mask)  # 区域内每点到背景的欧氏距离
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    # r = float(dist[y, x])  # 到边界的最大最小距离
    # return (y, x), r
    return (float(x), float(y))

def compute_centroid(mask_u8: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Compute a robust interior point (x,y) for the foreground mask as the
    Euclidean Distance Transform (EDT) maximum (i.e., the pixel farthest from
    the background). Returns None if the mask has no foreground.
    """
    if mask_u8.ndim == 3:
        # If mask has channels, convert to single channel by any non-zero across channels
        mask_u8 = (mask_u8.any(axis=2)).astype(np.uint8) * 255
    m = (mask_u8 > 0).astype(np.uint8)
    # Preprocess: keep largest connected component and fill holes for robustness
    try:
        from scipy import ndimage as ndi  # type: ignore
        if m.sum() > 0:
            labels, nlab = ndi.label(m, structure=np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.uint8))
            if nlab > 1:
                sizes = ndi.sum(m, labels, index=range(1, nlab + 1))
                largest_idx = int(np.argmax(sizes)) + 1
                m = (labels == largest_idx).astype(np.uint8)
            # Fill holes
            m = ndi.binary_fill_holes(m > 0).astype(np.uint8)
    except Exception:
        try:
            import cv2  # type: ignore
            if m.sum() > 0:
                # Connected components with stats; background is label 0
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
                if num_labels > 2:
                    # Find largest non-background component by area (stats[:, cv2.CC_STAT_AREA])
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    largest_idx = 1 + int(np.argmax(areas))
                    m = (labels == largest_idx).astype(np.uint8)
                elif num_labels == 2:
                    # Only one foreground component
                    m = (labels == 1).astype(np.uint8)
                # Fill holes via flood fill on background
                h, w = m.shape
                inv = (1 - m) * 255
                ff = inv.copy()
                mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
                cv2.floodFill(ff, mask, seedPoint=(0, 0), newVal=128)
                holes = (ff != 128)  # interior background
                m = ((1 - inv/255).astype(np.uint8) | holes.astype(np.uint8))
        except Exception:
            # If no morphology libs, proceed without preprocessing
            pass
    if m.sum() == 0:
        return None
    # Prefer SciPy's true Euclidean distance; fallback to OpenCV if SciPy unavailable
    dist = None
    try:
        from scipy import ndimage as ndi  # type: ignore
        dist = ndi.distance_transform_edt(m)
    except Exception:
        try:
            import cv2  # type: ignore
            dist = cv2.distanceTransform(m, distanceType=cv2.DIST_L2, maskSize=5)
        except Exception:
            # As a last resort, use cityblock (L1) via iterative distance on uint8
            # This is a coarse fallback and less accurate than EDT.
            # Here we approximate by growing distance in four-neighborhood.
            h, w = m.shape
            dist = np.zeros_like(m, dtype=np.float32)
            # Forward pass
            for y in range(h):
                for x in range(w):
                    if m[y, x] == 0:
                        continue
                    v = dist[y, x]
                    if y > 0:
                        v = max(v, dist[y - 1, x] + 1.0)
                    if x > 0:
                        v = max(v, dist[y, x - 1] + 1.0)
                    dist[y, x] = v
            # Backward pass
            for y in range(h - 1, -1, -1):
                for x in range(w - 1, -1, -1):
                    if m[y, x] == 0:
                        continue
                    v = dist[y, x]
                    if y + 1 < h:
                        v = max(v, dist[y + 1, x] + 1.0)
                    if x + 1 < w:
                        v = max(v, dist[y, x + 1] + 1.0)
                    dist[y, x] = v
    # Argmax returns (y, x)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return (float(x), float(y))

def largest_diff_component_representative(A, B, connectivity=2, min_area=0, dbg_out_path: Optional[str] = None):
    # A, B: 2D uint8 arrays, values {0,1} 或 {0,255}
    A = (A > 0).astype(np.uint8)
    B = (B > 0).astype(np.uint8)
    ### C = A.astype(np.int16) - B.astype(np.int16)  # {-1, 0, +1}
    C = A.astype(np.int8) - B.astype(np.int8)  # {-1, 0, +1}

    regions = []
    # 8 邻域结构元（connectivity=2 表示 8 邻域；=1 表示 4 邻域）
    st = np.ones((3,3), dtype=np.uint8) if connectivity == 2 else np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)

    for sign, mask in ((1, C > 0), (-1, C < 0)):
        if not mask.any():
            continue
        lab, n = ndi.label(mask, structure=st)
        if n == 0:
            continue
        areas = np.bincount(lab.ravel())[1:]  # 忽略背景0
        if min_area > 0:
            keep_ids = np.where(areas >= min_area)[0] + 1
            if keep_ids.size == 0:
                continue
            mask2 = np.isin(lab, keep_ids)
            lab, n = ndi.label(mask2, structure=st)
            if n == 0:
                continue
            areas = np.bincount(lab.ravel())[1:]

        k = np.argmax(areas) + 1
        comp = (lab == k)
        regions.append((sign, int(areas[k-1]), comp))

    if not regions:
        return None  # 无非零差异

    # 选面积最大的分量；若需以“最深”优先，可在 key 中加入 max EDT 作为次序
    sign, area, comp = max(regions, key=lambda t: t[1])

    # centroid = farthest_point_from_boundary(comp)
    # # centroid = compute_centroid(comp)
    # if centroid is None:
    #     return None
    D = distance_transform_edt(comp)
    (centroidY, centroidX) = np.unravel_index(np.argmax(D), D.shape)

    # 调试可视化（改用统一的渲染函数，保持风格一致）
    if dbg_out_path:
        try:
            _render_diff_panels_with_point(
                A=A,
                B=B,
                point_xy=(float(centroidX), float(centroidY)),
                label=1 if int(sign) > 0 else 0,
                dbg_out_path=dbg_out_path,
                connectivity=connectivity,
            )
        except Exception:
            pass

    return dict(sign=sign, area=area, y=int(centroidY), x=int(centroidX), max_radius=float(D[centroidY, centroidX]))


def _render_diff_panels_with_point(
    A: np.ndarray,
    B: np.ndarray,
    point_xy: Tuple[float, float],
    label: int,
    dbg_out_path: Optional[str],
    connectivity: int = 2,
    base_image_bgr: Optional[np.ndarray] = None,
    heatmap_rgb: Optional[np.ndarray] = None,
) -> None:
    """Render 3-panel debug image reusing the visualization style in
    largest_diff_component_representative, but mark a provided point instead of
    the recomputed centroid.
    - Panel 1: A (green) vs B (red) overlay
    - Panel 2: signed diff C (positive green, negative red)
    - Panel 3: connected components colored, with marker at point_xy
    - Panel 4 (optional): probability heatmap (RGB) + current point marker
    """
    rng = np.random.default_rng(12345)
    try:
        if dbg_out_path is None:
            return
        h, w = A.shape[:2]
        # Panel 1: A/B overlay (A=green, B=red), optionally blended on base image
        overlay1 = np.zeros((h, w, 3), dtype=np.uint8)
        overlay1[..., 1] = (A > 0).astype(np.uint8) * 255  # G
        overlay1[..., 2] = (B > 0).astype(np.uint8) * 255  # R
        if base_image_bgr is not None:
            base = base_image_bgr
            if base.ndim == 2:
                base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            if base.shape[:2] != (h, w):
                base = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
            panel1 = cv2.addWeighted(base, 1.0, overlay1, 0.6, 0.0)
        else:
            bg = np.zeros_like(overlay1)
            panel1 = cv2.addWeighted(bg, 1.0, overlay1, 0.6, 0.0)

        # Panel 2: signed diff C (positive green, negative red)
        C = A.astype(np.int16) - B.astype(np.int16)
        panel2 = np.zeros((h, w, 3), dtype=np.uint8)
        pos = (C > 0)
        neg = (C < 0)
        panel2[pos, 1] = 255
        panel2[neg, 2] = 255

        # Panel 3: connected components colored + provided point marker
        panel3 = np.zeros((h, w, 3), dtype=np.uint8)
        st = np.ones((3,3), dtype=np.uint8) if connectivity == 2 else np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
        lab, n = ndi.label(C, structure=st)
        if n != 0:
            colors = (rng.integers(0, 256, size=(n, 3))).astype(np.uint8)
            colors = np.clip(colors + 100, 0, 255)
            for i in range(1, n + 1):
                panel3[lab == i] = colors[i - 1]

        # draw marker at provided (x,y): foreground -> caret '^', background -> 'X'
        cx = int(round(float(point_xy[0])))
        cy = int(round(float(point_xy[1])))
        def _draw_cross(img, x, y, size=6, color=(255,255,255), thickness=2):
            cv2.line(img, (x - size, y - size), (x + size, y + size), (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, (x - size, y + size), (x + size, y - size), (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, (x - size, y - size), (x + size, y + size), color, thickness, lineType=cv2.LINE_AA)
            cv2.line(img, (x - size, y + size), (x + size, y - size), color, thickness, lineType=cv2.LINE_AA)
        def _draw_caret(img, x, y, size=6, color=(255,255,255), thickness=2):
            p_top = (int(x), int(y - size))
            p_left = (int(x - size), int(y + size))
            p_right = (int(x + size), int(y + size))
            cv2.line(img, p_left, p_top, (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, p_right, p_top, (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, p_left, p_top, color, thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p_right, p_top, color, thickness, lineType=cv2.LINE_AA)
        if int(label) > 0:
            _draw_caret(panel3, cx, cy, size=6, color=(255,255,255), thickness=2)
        else:
            _draw_cross(panel3, cx, cy, size=6, color=(255,255,255), thickness=2)

        # Compute and draw metrics comparing B (pred) vs A (gt) on panel1
        def _compute_metrics(gt_u8: np.ndarray, pr_u8: np.ndarray) -> dict:
            gt = (gt_u8 > 0)
            pr = (pr_u8 > 0)
            tp = int(np.logical_and(gt, pr).sum())
            fp = int(np.logical_and(~gt, pr).sum())
            fn = int(np.logical_and(gt, ~pr).sum())
            denom_dice = 2 * tp + fp + fn
            dice = (2 * tp / denom_dice) if denom_dice > 0 else 1.0
            denom_iou = tp + fp + fn
            iou = (tp / denom_iou) if denom_iou > 0 else 1.0
            precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
            recall = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            # Lightweight S-measure implementation adapted for binary masks
            def _s_measure_binary(pred: np.ndarray, gt: np.ndarray) -> float:
                pred_f = pred.astype(np.float64)
                gt_f = gt.astype(np.float64)
                y_mean = gt_f.mean()
                if y_mean == 0:
                    return float(1.0 - pred_f.mean())
                if y_mean == 1:
                    return float(pred_f.mean())

                def s_object(p: np.ndarray, g: np.ndarray) -> float:
                    vals = p[g == 1]
                    if vals.size == 0:
                        return 0.0
                    mx = float(vals.mean())
                    sx = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
                    return float(2 * mx / (mx * mx + 1 + sx + np.spacing(1)))

                def ssim(a: np.ndarray, b: np.ndarray) -> float:
                    h_, w_ = a.shape
                    N = h_ * w_
                    if N <= 1:
                        return 1.0
                    mx = float(a.mean()); my = float(b.mean())
                    sx = float(((a - mx) ** 2).sum() / (N - 1))
                    sy = float(((b - my) ** 2).sum() / (N - 1))
                    sxy = float(((a - mx) * (b - my)).sum() / (N - 1))
                    alpha = 4 * mx * my * sxy
                    beta = (mx * mx + my * my) * (sx + sy)
                    if alpha != 0:
                        return float(alpha / (beta + np.spacing(1)))
                    if alpha == 0 and beta == 0:
                        return 1.0
                    return 0.0

                coords = np.argwhere(gt_f == 1)
                h_, w_ = gt_f.shape
                if coords.size == 0:
                    cx, cy = int(round(w_ / 2)), int(round(h_ / 2))
                else:
                    cy, cx = coords.mean(axis=0).round().astype(int).tolist()
                cx = int(cx) + 1; cy = int(cy) + 1
                gt_LT = gt_f[0:cy, 0:cx]; gt_RT = gt_f[0:cy, cx:w_]
                gt_LB = gt_f[cy:h_, 0:cx]; gt_RB = gt_f[cy:h_, cx:w_]
                pr_LT = pred_f[0:cy, 0:cx]; pr_RT = pred_f[0:cy, cx:w_]
                pr_LB = pred_f[cy:h_, 0:cx]; pr_RB = pred_f[cy:h_, cx:w_]
                area = float(h_ * w_)
                w1 = (cx * cy) / area
                w2 = (cy * (w_ - cx)) / area
                w3 = ((h_ - cy) * cx) / area
                w4 = 1.0 - w1 - w2 - w3
                region_score = (
                    w1 * ssim(pr_LT, gt_LT) +
                    w2 * ssim(pr_RT, gt_RT) +
                    w3 * ssim(pr_LB, gt_LB) +
                    w4 * ssim(pr_RB, gt_RB)
                )
                alpha = 0.5
                return float(max(0.0, alpha * s_object(pred_f, gt_f) + (1 - alpha) * region_score))

            s_measure = _s_measure_binary(pr, gt)
            return {
                "DICE": dice,
                "IOU": iou,
                "PRECISION": precision,
                "RECALL": recall,
                "S_MEASURE": s_measure,
                "F1_SCORE": f1,
            }

        def _draw_metrics_footer(img: np.ndarray, metrics: dict, keys: List[str]) -> None:
            parts: List[str] = []
            for k in keys:
                if k in metrics:
                    parts.append(f"{k}: {metrics[k]:.3f}")
            if not parts:
                return
            text = "  ".join(parts)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 1
            max_width = img.shape[1] - 16
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            if tw > max_width and tw > 0:
                scale = max(0.3, scale * (max_width / tw))
                (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            x = 8
            y = img.shape[0] - 6
            # stroke for readability
            cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), 3, lineType=cv2.LINE_AA)
            cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        try:
            m = _compute_metrics(A, B)
            _draw_metrics_footer(panel1, m, ["DICE", "IOU"])
            _draw_metrics_footer(panel2, m, ["PRECISION", "RECALL"])
            _draw_metrics_footer(panel3, m, ["S_MEASURE", "F1_SCORE"])
        except Exception:
            pass

        # Panel 4: probability heatmap + point (若提供 heatmap_rgb)
        if heatmap_rgb is not None:
            hm = heatmap_rgb
            if hm.ndim == 2:
                hm = cv2.cvtColor(hm, cv2.COLOR_GRAY2BGR)
            if hm.shape[:2] != (h, w):
                hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
            panel4 = hm.copy()
            # draw current point on heatmap
            if int(label) > 0:
                _draw_caret(panel4, cx, cy, size=6, color=(255,255,255), thickness=2)
            else:
                _draw_cross(panel4, cx, cy, size=6, color=(255,255,255), thickness=2)

        # Titles
        def _add_title(img: np.ndarray, text: str) -> None:
            band_h = max(20, min(48, img.shape[0] // 20))
            roi = img[0:band_h, :, :]
            overlay = roi.copy()
            cv2.rectangle(overlay, (0, 0), (img.shape[1] - 1, band_h - 1), (0, 0, 0), thickness=-1)
            cv2.addWeighted(overlay, 0.5, roi, 0.5, 0, dst=roi)
            org = (8, band_h - 6)
            font_scale = 0.4
            font_thickness = 1
            # Truncate text to fit within image width
            max_width = img.shape[1] - 16  # Leave some padding
            (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            if text_width > max_width:
                # Binary search to find the right truncation point
                ellipsis = "..."
                for i in range(len(text), 0, -1):
                    truncated = text[:i] + ellipsis
                    (trunc_width, _), _ = cv2.getTextSize(truncated, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    if trunc_width <= max_width:
                        text = truncated
                        break
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, lineType=cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)
        _add_title(panel1, "Overlay A (green) vs B (red)")
        _add_title(panel2, "Signed diff C: + (green), - (red)")
        _add_title(panel3, "Components + selected point")
        if heatmap_rgb is not None:
            _add_title(panel4, "Probability heatmap + selected point")

        panels = [panel1, panel2, panel3]
        if heatmap_rgb is not None:
            panels.append(panel4)
        canvas = np.concatenate(panels, axis=1)
        os.makedirs(os.path.dirname(dbg_out_path) or ".", exist_ok=True)
        _PIL.fromarray(canvas[..., ::-1]).save(dbg_out_path)
    except Exception:
        pass

def find_corresponding_image(
    images_dir: str,
    sample_stem: str,
    allowed_exts: Tuple[str, ...],
) -> Optional[str]:
    images_dir_abs = to_abs(images_dir)
    for ext in allowed_exts:
        candidate = os.path.join(images_dir_abs, sample_stem + ext)
        if os.path.isfile(candidate):
            return candidate
    # Fallback: scan directory and match by stem (handles mixed-case extensions or uncommon extensions)
    for name in os.listdir(images_dir_abs):
        if stem(name).lower() == sample_stem.lower():
            return os.path.join(images_dir_abs, name)
    return None

def calculate_next_point(gt_mask_path, sam_masks_dir, current_points_len, dbg_out_dir: Optional[str] = None):
    # 读取gt与pred
    try:
        gt_u8 = np.array(_PIL.open(to_abs(gt_mask_path)), dtype=np.uint8)
        if current_points_len > 0:
            last_mask_path = _find_latest_mask_path(sam_masks_dir, stem(gt_mask_path), current_points_len)
            if not last_mask_path or not os.path.isfile(last_mask_path):
                print(f"[WARN] Skip: last mask not found under {sam_masks_dir}")
                return None, None, None
            pred_u8 = np.array(_PIL.open(to_abs(last_mask_path)), dtype=np.uint8)
        else:
            pred_u8 = np.zeros_like(gt_u8)
    except Exception as e:
        print(f"[WARN] Skip: failed to read masks ({e})")
        return None, None, None

    # 若形状不同，将pred按最近邻缩放到gt尺寸
    h_g, w_g = gt_u8.shape[:2]
    h_p, w_p = pred_u8.shape[:2]
    if (h_g, w_g) != (h_p, w_p):
        try:
            # from PIL import Image as _PIL
            pred_u8 = np.array(_PIL.fromarray(pred_u8).resize((w_g, h_g), resample=_PIL.NEAREST), dtype=np.uint8)
        except Exception:
            pass

    ### For Debug Only:
    # if stem(gt_mask_path) == "BRATS_001_z0088":
    #     print("#Debug#")
    ### :For Debug Only

    # 计算差异的最大连通分量代表点
    dbg_path = None
    if dbg_out_dir:
        try:
            subdir = os.path.join(dbg_out_dir, stem(gt_mask_path))
            os.makedirs(subdir, exist_ok=True)
            dbg_path = os.path.join(subdir, f"dbg_{max(0, int(current_points_len))}.png")
        except Exception:
            dbg_path = None
    rep = largest_diff_component_representative(gt_u8, pred_u8, connectivity=2, min_area=0, dbg_out_path=dbg_path)
    if rep is None:
        print(f"[INFO] no diff component; skip append")
        return None, None, None
    x_next = int(rep["x"])  # 列
    y_next = int(rep["y"])  # 行
    label_next = 1 if int(rep["sign"]) > 0 else 0
    return x_next, y_next, label_next

def main() -> None:
    global _PIL
    args = parse_args()

    # Debug-only mode: read JSON, generate debug images but do not write JSON
    if args.debug_json:
        if not args.debug_output_dir:
            raise RuntimeError("--debug_output_dir is required when --debug_json is set")
        json_path = args.debug_json
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Debug JSON not found: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        try:
            arr = json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Debug JSON must be a JSON array: {e}")
        if not isinstance(arr, list):
            raise RuntimeError("Debug JSON root must be a JSON array")

        num_generated = 0
        for rec in arr:
            if not isinstance(rec, dict):
                continue

            image_path = rec.get("image")
            gt_mask_path = rec.get("gt_mask")
            sam_masks_dir = rec.get("sam_masks_dir")
            if not (image_path and sam_masks_dir):
                continue
            sample_stem = stem(image_path)
            ### For Debug Only:
            # if not sample_stem == "BRATS_001_z0088":
            #     continue
            ### :For Debug Only
            try:
                subdir = os.path.join(args.debug_output_dir, sample_stem)
                os.makedirs(subdir, exist_ok=True)
            except Exception:
                continue

            points = rec.get("points", [])
            labels = rec.get("labels", [])
            if not (isinstance(points, list) and isinstance(labels, list)):
                continue
            if len(points) != len(labels):
                continue

            gt_mask_u8 = np.array(_PIL.open(to_abs(gt_mask_path)), dtype=np.uint8)
            # Load base image if exists
            base_img_bgr = None
            try:
                img_arr = np.array(_PIL.open(to_abs(image_path)).convert("RGB"))
                base_img_bgr = img_arr[:, :, ::-1]
            except Exception:
                base_img_bgr = None
            # Load gt as all-zero array to preserve panel structure; we visualize differences of predicted masks only
            for i, (pt, lb) in enumerate(zip(points, labels)):
                # For step i, compare masks of ground truth and step i-1 (i==0 uses empty prev)
                if i == 0:
                    prev = np.zeros_like(gt_mask_u8, dtype=np.uint8)
                else:
                    try:
                        prev_path = os.path.join(sam_masks_dir, sample_stem, f"{i-1}.png")
                        prev = np.array(_PIL.open(to_abs(prev_path)).convert("L"), dtype=np.uint8) if os.path.isfile(prev_path) else np.zeros_like(gt_mask_u8, dtype=np.uint8)
                    except Exception:
                        prev = np.zeros_like(gt_mask_u8, dtype=np.uint8)
                # Render panels with provided point
                dbg_path = os.path.join(subdir, f"dbg_{i}.png")
                _render_diff_panels_with_point(
                    gt_mask_u8,
                    prev,
                    (float(pt[0]), float(pt[1])),
                    int(lb),
                    dbg_path,
                    base_image_bgr=base_img_bgr,
                )
                num_generated += 1
        print(f"Generated debug images for {num_generated} steps into {args.debug_output_dir}")
        return

    # 追加模式：读取已有JSON数组，计算下一提示点并写回
    if args.appendto_jsonl:
        json_path = args.appendto_jsonl
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Append target not found: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        try:
            arr = json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Append target must be a JSON array: {e}")
        if not isinstance(arr, list):
            raise RuntimeError("Append target root must be a JSON array")

        updated = 0
        for idx, rec in enumerate(arr):
            if not isinstance(rec, dict):
                continue
            image_path = rec.get("image")
            gt_mask_path = rec.get("gt_mask")
            sam_masks_dir = rec.get("sam_masks_dir")
            points_list = rec.get("points", [])
            labels_list = rec.get("labels", [])

            if not (image_path and gt_mask_path and sam_masks_dir):
                print(f"[WARN] Skip idx {idx}: missing fields")
                continue

            x_next, y_next, label_next = calculate_next_point(
                gt_mask_path,
                sam_masks_dir,
                len(points_list),
                dbg_out_dir=args.debug_output_dir,
            )
            if x_next is None or y_next is None or label_next is None:
                continue

            # 追加points与labels
            if not isinstance(points_list, list):
                points_list = []
            if not isinstance(labels_list, list):
                labels_list = []
            points_list.append([x_next, y_next])
            labels_list.append(int(label_next))
            rec["points"] = points_list
            rec["labels"] = labels_list
            updated += 1

        # 回写同一文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(arr, f, indent=2, ensure_ascii=False)
        print(f"Appended next prompts to {json_path} (updated {updated} items)")
        return

    # 否则：首写模式，生成首个提示点（与之前逻辑一致）
    if not args.images_dir or not args.masks_dir or not args.output_jsonl:
        raise RuntimeError("images_dir, masks_dir and output_jsonl are required for initial generation mode")

    images_dir = args.images_dir
    masks_dir = args.masks_dir
    output_jsonl = args.output_jsonl
    allowed_image_exts = tuple(e.strip().lower() for e in args.image_exts.split(",") if e.strip())

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    # Visualization output disabled
    # if args.viz_dir:
    #     os.makedirs(args.viz_dir, exist_ok=True)

    # Gather masks as the source of truth
    mask_files = list_files_with_exts(masks_dir, allowed_exts=(".png",))
    if len(mask_files) == 0:
        raise RuntimeError(f"No mask PNG files found under {masks_dir}")

    num_written = 0
    num_skipped_empty = 0
    num_missing_images = 0
    num_size_mismatch = 0

    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for mask_name in mask_files:
            sample_stem = stem(mask_name)
            mask_path = os.path.join(masks_dir, mask_name)
            
            # Load mask
            try:
                mask_img = _PIL.open(to_abs(mask_path))
                mask_u8 = np.array(mask_img, dtype=np.uint8)
            except Exception as e:
                print(f"[WARN] Skipping unreadable mask: {mask_path} ({e})")
                continue
            
            x_next, y_next, label_next = calculate_next_point(
                mask_path,
                None,
                0,
                dbg_out_dir=args.debug_output_dir,
            )
            if x_next is None or y_next is None or label_next is None:
                if args.skip_empty:
                    num_skipped_empty += 1
                    continue
                else:
                    # For empty masks and skip_empty=False, place centroid at image center
                    h, w = mask_u8.shape[:2]
                    centroid = (float(w - 1) / 2.0, float(h - 1) / 2.0)
            centroid = (x_next, y_next)

            # Find corresponding image
            img_path = find_corresponding_image(images_dir, sample_stem, allowed_image_exts)
            if img_path is None:
                num_missing_images += 1
                print(f"[WARN] Missing image for mask stem '{sample_stem}' in {images_dir}")
                continue
           
            image_field = os.path.abspath(img_path) if args.abs_paths else img_path
            mask_field = os.path.abspath(mask_path) if args.abs_paths else mask_path
            record = {"image": image_field, "gt_mask": mask_field, "points": [[float(centroid[0]), float(centroid[1])]], "labels": [1]}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1

            # Visualization
            if False and args.viz_dir:
                try:
                    # Load image RGB
                    with _PIL.open(to_abs(img_path)) as im_rgb:
                        im_rgb = im_rgb.convert("RGB")
                        w_im, h_im = im_rgb.size
                        
                        # Ensure mask size matches visualization image
                        m = mask_u8
                        if (m.shape[1], m.shape[0]) != (w_im, h_im):
                            # resize nearest for mask
                            m_img = Image.fromarray(m)
                            m_img = m_img.resize((w_im, h_im), resample=Image.NEAREST)
                            m = np.array(m_img, dtype=np.uint8)
                        
                        # Create colored mask overlay
                        overlay = Image.new("RGBA", (w_im, h_im), (0, 0, 0, 0))
                        mask_alpha = int(max(0.0, min(1.0, args.viz_alpha)) * 255)
                        color = (255, 0, 0, mask_alpha)
                        # Build alpha mask where foreground>0
                        fg = (m > 0).astype(np.uint8) * mask_alpha
                        alpha_img = Image.fromarray(fg, mode="L")
                        color_img = Image.new("RGBA", (w_im, h_im), color)
                        overlay.paste(color_img, (0, 0), mask=alpha_img)
                        
                        # Compose overlay on image
                        comp = im_rgb.convert("RGBA")
                        comp = Image.alpha_composite(comp, overlay)
                        
                        # Draw the deepest point
                        draw = ImageDraw.Draw(comp)
                        px, py = float(centroid[0]), float(centroid[1])
                        r = max(1, int(args.viz_radius))
                        draw.ellipse((px - r, py - r, px + r, py + r), outline=(0, 255, 0, 255), width=2)
                        
                        # Save
                        out_name = f"{sample_stem}.png"
                        comp.convert("RGB").save(os.path.join(args.viz_dir, out_name))
                except Exception as e:
                    print(f"[WARN] Visualization failed for {img_path}: {e}")

    print(f"Wrote {num_written} samples to {output_jsonl}")
    if num_skipped_empty:
        print(f"Skipped empty masks: {num_skipped_empty}")
    if num_missing_images:
        print(f"Masks with missing images: {num_missing_images}")
    if num_size_mismatch:
        print(f"Image/Mask size mismatches adjusted: {num_size_mismatch}")


if __name__ == "__main__":
    main()



```

```python
#!/usr/bin/env python3

"""
Generate JSONL supervision (image,x,y) by computing a "deep interior" point of
binary masks using the Euclidean Distance Transform (EDT) maximum.

Each output line is a JSON object like:
  {"image": "/abs/path/to/image.jpg", "gt_mask": "/abs/path/to/mask.png", "points": [[123.4,456.7]], "labels": [1]}

Coordinates are in pixel space of the original image/mask pair.
Designed for Seg-R1 heatmap-classification pretraining pipelines.

Example usage:
  # 1) 首次生成（计算首个最佳提示点）
  python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --images_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir  /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour.jsonl

  /opt/anaconda3/envs/seg-r1/bin/python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --images_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/images \
    --masks_dir datasets/seg_r1_md/Task01_BrainTumour/canonical/masks \
    --output_jsonl datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251002.jsonl

  # 2) 追加模式（为每条记录计算后续的第2个及以后提示点）
  python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --appendto_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour.jsonl

  /opt/anaconda3/envs/seg-r1/bin/python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --appendto_jsonl datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour-251003.jsonl

  # 3) 调试模式（生成调试图像）
  # 若 sam2_segment_from_points.py 已生成 heatmap-{i}.png，本脚本会在调试图中追加第4幅：
  # “概率热力图 + 当前提示点”，用于核对提示点与热区的一致性。
  /opt/anaconda3/envs/seg-r1/bin/python seg-rl/annotator/gen_point_jsonl_from_masks.py \
    --debug_json outputs/braintumour/pred_251023.jsonl \
    --debug_output_dir outputs/braintumour/dbg_gen_points

Notes:
- Foreground is defined as any non-zero pixel in the mask.
- Target point is the pixel farthest from the background (EDT argmax), which
  is guaranteed to lie inside the foreground, robust to holes and thin parts.
- Samples with empty masks are skipped by default (override with --skip_empty false).
"""

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
# from PIL import Image, ImageDraw
from PIL import Image as _PIL
from scipy.ndimage import label, distance_transform_edt
import cv2  # type: ignore

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create/Append deepest-point prompts from masks using EDT")
    # 互斥：首写输出 vs 追加到已存在的JSON数组
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--output_jsonl", type=str, help="Path to write JSONL lines for first prompts")
    group.add_argument("--appendto_jsonl", type=str, help="Path to existing JSON array to append next prompts")
    group.add_argument("--debug_json", type=str, help="Path to JSON array to read for debug visualization only")

    # 首次生成模式需要的参数（在追加模式下将被忽略）
    p.add_argument("--images_dir", type=str, default=None, help="Directory containing images (e.g., JPG)")
    p.add_argument("--masks_dir", type=str, default=None, help="Directory containing PNG masks aligned to images")
    p.add_argument("--abs_paths", type=lambda s: s.lower() in {"1","true","yes","y"}, default=True,
                   help="Write absolute image paths (default: True)")
    p.add_argument("--skip_empty", type=lambda s: s.lower() in {"1","true","yes","y"}, default=True,
                   help="Skip samples where mask is empty (default: True)")
    p.add_argument("--image_exts", type=str, default=".jpg,.jpeg,.png",
                   help="Comma-separated acceptable image extensions (default: .jpg,.jpeg,.png)")
    p.add_argument("--viz_dir", type=str, default=None, help="Optional directory to save overlay visualizations")
    p.add_argument("--viz_alpha", type=float, default=0.35, help="Alpha for mask overlay (0-1)")
    p.add_argument("--viz_radius", type=int, default=4, help="Radius (pixels) for deepest-point circle")
    p.add_argument("--debug_output_dir", type=str, default=None,
                   help="If set, write per-sample debug images for diff component selection")
    return p.parse_args()


def list_files_with_exts(directory: str, allowed_exts: Tuple[str, ...]) -> List[str]:
    directory = to_abs(directory) or directory
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    out: List[str] = []
    for name in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, name)):
            continue
        if any(name.lower().endswith(ext) for ext in allowed_exts):
            out.append(name)
    return sorted(out)


def stem(path: str) -> str:
    name = os.path.basename(path)
    # Remove common two-part extensions like .nii.gz by iterative splitext
    base, ext = os.path.splitext(name)
    return base


def to_abs(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.abspath(path) if not os.path.isabs(path) else path


def _find_latest_mask_path(sam_masks_dir: str, image_stem: str, current_points_len: int) -> Optional[str]:
    """返回该图像在sam_masks_dir下的最新预测mask路径。
    优先尝试 sam_masks_dir/<stem>/<current_points_len-1>.png；若不存在，
    则在该目录下寻找最大数字文件名的png。
    """
    candidate_dir = os.path.join(sam_masks_dir, image_stem)
    if not os.path.isdir(candidate_dir):
        return None
    k = max(0, int(current_points_len - 1))
    preferred = os.path.join(candidate_dir, f"{k}.png")
    if os.path.isfile(preferred):
        return preferred
    # fallback: find max numeric png
    best_idx = -1
    best_path = None
    for name in os.listdir(candidate_dir):
        if not name.lower().endswith(".png"):
            continue
        s = os.path.splitext(name)[0]
        try:
            idx = int(s)
        except Exception:
            continue
        if idx > best_idx:
            best_idx = idx
            best_path = os.path.join(candidate_dir, name)
    return best_path

from scipy import ndimage as ndi

def farthest_point_from_boundary(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    # mask: 二值数组，区域为 True/1，背景为 False/0
    mask = (mask > 0)
    dist = ndi.distance_transform_edt(mask)  # 区域内每点到背景的欧氏距离
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    # r = float(dist[y, x])  # 到边界的最大最小距离
    # return (y, x), r
    return (float(x), float(y))

def compute_centroid(mask_u8: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Compute a robust interior point (x,y) for the foreground mask as the
    Euclidean Distance Transform (EDT) maximum (i.e., the pixel farthest from
    the background). Returns None if the mask has no foreground.
    """
    if mask_u8.ndim == 3:
        # If mask has channels, convert to single channel by any non-zero across channels
        mask_u8 = (mask_u8.any(axis=2)).astype(np.uint8) * 255
    m = (mask_u8 > 0).astype(np.uint8)
    # Preprocess: keep largest connected component and fill holes for robustness
    try:
        from scipy import ndimage as ndi  # type: ignore
        if m.sum() > 0:
            labels, nlab = ndi.label(m, structure=np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.uint8))
            if nlab > 1:
                sizes = ndi.sum(m, labels, index=range(1, nlab + 1))
                largest_idx = int(np.argmax(sizes)) + 1
                m = (labels == largest_idx).astype(np.uint8)
            # Fill holes
            m = ndi.binary_fill_holes(m > 0).astype(np.uint8)
    except Exception:
        try:
            import cv2  # type: ignore
            if m.sum() > 0:
                # Connected components with stats; background is label 0
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
                if num_labels > 2:
                    # Find largest non-background component by area (stats[:, cv2.CC_STAT_AREA])
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    largest_idx = 1 + int(np.argmax(areas))
                    m = (labels == largest_idx).astype(np.uint8)
                elif num_labels == 2:
                    # Only one foreground component
                    m = (labels == 1).astype(np.uint8)
                # Fill holes via flood fill on background
                h, w = m.shape
                inv = (1 - m) * 255
                ff = inv.copy()
                mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
                cv2.floodFill(ff, mask, seedPoint=(0, 0), newVal=128)
                holes = (ff != 128)  # interior background
                m = ((1 - inv/255).astype(np.uint8) | holes.astype(np.uint8))
        except Exception:
            # If no morphology libs, proceed without preprocessing
            pass
    if m.sum() == 0:
        return None
    # Prefer SciPy's true Euclidean distance; fallback to OpenCV if SciPy unavailable
    dist = None
    try:
        from scipy import ndimage as ndi  # type: ignore
        dist = ndi.distance_transform_edt(m)
    except Exception:
        try:
            import cv2  # type: ignore
            dist = cv2.distanceTransform(m, distanceType=cv2.DIST_L2, maskSize=5)
        except Exception:
            # As a last resort, use cityblock (L1) via iterative distance on uint8
            # This is a coarse fallback and less accurate than EDT.
            # Here we approximate by growing distance in four-neighborhood.
            h, w = m.shape
            dist = np.zeros_like(m, dtype=np.float32)
            # Forward pass
            for y in range(h):
                for x in range(w):
                    if m[y, x] == 0:
                        continue
                    v = dist[y, x]
                    if y > 0:
                        v = max(v, dist[y - 1, x] + 1.0)
                    if x > 0:
                        v = max(v, dist[y, x - 1] + 1.0)
                    dist[y, x] = v
            # Backward pass
            for y in range(h - 1, -1, -1):
                for x in range(w - 1, -1, -1):
                    if m[y, x] == 0:
                        continue
                    v = dist[y, x]
                    if y + 1 < h:
                        v = max(v, dist[y + 1, x] + 1.0)
                    if x + 1 < w:
                        v = max(v, dist[y, x + 1] + 1.0)
                    dist[y, x] = v
    # Argmax returns (y, x)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return (float(x), float(y))

def largest_diff_component_representative(A, B, connectivity=2, min_area=0, dbg_out_path: Optional[str] = None):
    # A, B: 2D uint8 arrays, values {0,1} 或 {0,255}
    A = (A > 0).astype(np.uint8)
    B = (B > 0).astype(np.uint8)
    ### C = A.astype(np.int16) - B.astype(np.int16)  # {-1, 0, +1}
    C = A.astype(np.int8) - B.astype(np.int8)  # {-1, 0, +1}

    regions = []
    # 8 邻域结构元（connectivity=2 表示 8 邻域；=1 表示 4 邻域）
    st = np.ones((3,3), dtype=np.uint8) if connectivity == 2 else np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)

    for sign, mask in ((1, C > 0), (-1, C < 0)):
        if not mask.any():
            continue
        lab, n = ndi.label(mask, structure=st)
        if n == 0:
            continue
        areas = np.bincount(lab.ravel())[1:]  # 忽略背景0
        if min_area > 0:
            keep_ids = np.where(areas >= min_area)[0] + 1
            if keep_ids.size == 0:
                continue
            mask2 = np.isin(lab, keep_ids)
            lab, n = ndi.label(mask2, structure=st)
            if n == 0:
                continue
            areas = np.bincount(lab.ravel())[1:]

        k = np.argmax(areas) + 1
        comp = (lab == k)
        regions.append((sign, int(areas[k-1]), comp))

    if not regions:
        return None  # 无非零差异

    # 选面积最大的分量；若需以“最深”优先，可在 key 中加入 max EDT 作为次序
    sign, area, comp = max(regions, key=lambda t: t[1])

    # centroid = farthest_point_from_boundary(comp)
    # # centroid = compute_centroid(comp)
    # if centroid is None:
    #     return None
    D = distance_transform_edt(comp)
    (centroidY, centroidX) = np.unravel_index(np.argmax(D), D.shape)

    # 调试可视化（改用统一的渲染函数，保持风格一致）
    if dbg_out_path:
        try:
            _render_diff_panels_with_point(
                A=A,
                B=B,
                point_xy=(float(centroidX), float(centroidY)),
                label=1 if int(sign) > 0 else 0,
                dbg_out_path=dbg_out_path,
                connectivity=connectivity,
            )
        except Exception:
            pass

    return dict(sign=sign, area=area, y=int(centroidY), x=int(centroidX), max_radius=float(D[centroidY, centroidX]))


def _render_diff_panels_with_point(
    A: np.ndarray,
    B: np.ndarray,
    point_xy: Tuple[float, float],
    label: int,
    dbg_out_path: Optional[str],
    connectivity: int = 2,
    base_image_bgr: Optional[np.ndarray] = None,
    heatmap_rgb: Optional[np.ndarray] = None,
) -> None:
    """Render 3-panel debug image reusing the visualization style in
    largest_diff_component_representative, but mark a provided point instead of
    the recomputed centroid.
    - Panel 1: A (green) vs B (red) overlay
    - Panel 2: signed diff C (positive green, negative red)
    - Panel 3: connected components colored, with marker at point_xy
    - Panel 4 (optional): probability heatmap (RGB) + current point marker
    """
    rng = np.random.default_rng(12345)
    try:
        if dbg_out_path is None:
            return
        h, w = A.shape[:2]
        # Panel 1: A/B overlay (A=green, B=red), optionally blended on base image
        overlay1 = np.zeros((h, w, 3), dtype=np.uint8)
        overlay1[..., 1] = (A > 0).astype(np.uint8) * 255  # G
        overlay1[..., 2] = (B > 0).astype(np.uint8) * 255  # R
        if base_image_bgr is not None:
            base = base_image_bgr
            if base.ndim == 2:
                base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            if base.shape[:2] != (h, w):
                base = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
            panel1 = cv2.addWeighted(base, 1.0, overlay1, 0.6, 0.0)
        else:
            bg = np.zeros_like(overlay1)
            panel1 = cv2.addWeighted(bg, 1.0, overlay1, 0.6, 0.0)

        # Panel 2: signed diff C (positive green, negative red)
        C = A.astype(np.int16) - B.astype(np.int16)
        panel2 = np.zeros((h, w, 3), dtype=np.uint8)
        pos = (C > 0)
        neg = (C < 0)
        panel2[pos, 1] = 255
        panel2[neg, 2] = 255

        # Panel 3: connected components colored + provided point marker
        panel3 = np.zeros((h, w, 3), dtype=np.uint8)
        st = np.ones((3,3), dtype=np.uint8) if connectivity == 2 else np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
        lab, n = ndi.label(C, structure=st)
        if n != 0:
            colors = (rng.integers(0, 256, size=(n, 3))).astype(np.uint8)
            colors = np.clip(colors + 100, 0, 255)
            for i in range(1, n + 1):
                panel3[lab == i] = colors[i - 1]

        # draw marker at provided (x,y): foreground -> caret '^', background -> 'X'
        cx = int(round(float(point_xy[0])))
        cy = int(round(float(point_xy[1])))
        def _draw_cross(img, x, y, size=6, color=(255,255,255), thickness=2):
            cv2.line(img, (x - size, y - size), (x + size, y + size), (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, (x - size, y + size), (x + size, y - size), (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, (x - size, y - size), (x + size, y + size), color, thickness, lineType=cv2.LINE_AA)
            cv2.line(img, (x - size, y + size), (x + size, y - size), color, thickness, lineType=cv2.LINE_AA)
        def _draw_caret(img, x, y, size=6, color=(255,255,255), thickness=2):
            p_top = (int(x), int(y - size))
            p_left = (int(x - size), int(y + size))
            p_right = (int(x + size), int(y + size))
            cv2.line(img, p_left, p_top, (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, p_right, p_top, (0,0,0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.line(img, p_left, p_top, color, thickness, lineType=cv2.LINE_AA)
            cv2.line(img, p_right, p_top, color, thickness, lineType=cv2.LINE_AA)
        if int(label) > 0:
            _draw_caret(panel3, cx, cy, size=6, color=(255,255,255), thickness=2)
        else:
            _draw_cross(panel3, cx, cy, size=6, color=(255,255,255), thickness=2)

        # Compute and draw metrics comparing B (pred) vs A (gt) on panel1
        def _compute_metrics(gt_u8: np.ndarray, pr_u8: np.ndarray) -> dict:
            gt = (gt_u8 > 0)
            pr = (pr_u8 > 0)
            tp = int(np.logical_and(gt, pr).sum())
            fp = int(np.logical_and(~gt, pr).sum())
            fn = int(np.logical_and(gt, ~pr).sum())
            denom_dice = 2 * tp + fp + fn
            dice = (2 * tp / denom_dice) if denom_dice > 0 else 1.0
            denom_iou = tp + fp + fn
            iou = (tp / denom_iou) if denom_iou > 0 else 1.0
            precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
            recall = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            # Lightweight S-measure implementation adapted for binary masks
            def _s_measure_binary(pred: np.ndarray, gt: np.ndarray) -> float:
                pred_f = pred.astype(np.float64)
                gt_f = gt.astype(np.float64)
                y_mean = gt_f.mean()
                if y_mean == 0:
                    return float(1.0 - pred_f.mean())
                if y_mean == 1:
                    return float(pred_f.mean())

                def s_object(p: np.ndarray, g: np.ndarray) -> float:
                    vals = p[g == 1]
                    if vals.size == 0:
                        return 0.0
                    mx = float(vals.mean())
                    sx = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
                    return float(2 * mx / (mx * mx + 1 + sx + np.spacing(1)))

                def ssim(a: np.ndarray, b: np.ndarray) -> float:
                    h_, w_ = a.shape
                    N = h_ * w_
                    if N <= 1:
                        return 1.0
                    mx = float(a.mean()); my = float(b.mean())
                    sx = float(((a - mx) ** 2).sum() / (N - 1))
                    sy = float(((b - my) ** 2).sum() / (N - 1))
                    sxy = float(((a - mx) * (b - my)).sum() / (N - 1))
                    alpha = 4 * mx * my * sxy
                    beta = (mx * mx + my * my) * (sx + sy)
                    if alpha != 0:
                        return float(alpha / (beta + np.spacing(1)))
                    if alpha == 0 and beta == 0:
                        return 1.0
                    return 0.0

                coords = np.argwhere(gt_f == 1)
                h_, w_ = gt_f.shape
                if coords.size == 0:
                    cx, cy = int(round(w_ / 2)), int(round(h_ / 2))
                else:
                    cy, cx = coords.mean(axis=0).round().astype(int).tolist()
                cx = int(cx) + 1; cy = int(cy) + 1
                gt_LT = gt_f[0:cy, 0:cx]; gt_RT = gt_f[0:cy, cx:w_]
                gt_LB = gt_f[cy:h_, 0:cx]; gt_RB = gt_f[cy:h_, cx:w_]
                pr_LT = pred_f[0:cy, 0:cx]; pr_RT = pred_f[0:cy, cx:w_]
                pr_LB = pred_f[cy:h_, 0:cx]; pr_RB = pred_f[cy:h_, cx:w_]
                area = float(h_ * w_)
                w1 = (cx * cy) / area
                w2 = (cy * (w_ - cx)) / area
                w3 = ((h_ - cy) * cx) / area
                w4 = 1.0 - w1 - w2 - w3
                region_score = (
                    w1 * ssim(pr_LT, gt_LT) +
                    w2 * ssim(pr_RT, gt_RT) +
                    w3 * ssim(pr_LB, gt_LB) +
                    w4 * ssim(pr_RB, gt_RB)
                )
                alpha = 0.5
                return float(max(0.0, alpha * s_object(pred_f, gt_f) + (1 - alpha) * region_score))

            s_measure = _s_measure_binary(pr, gt)
            return {
                "DICE": dice,
                "IOU": iou,
                "PRECISION": precision,
                "RECALL": recall,
                "S_MEASURE": s_measure,
                "F1_SCORE": f1,
            }

        def _draw_metrics_footer(img: np.ndarray, metrics: dict, keys: List[str]) -> None:
            parts: List[str] = []
            for k in keys:
                if k in metrics:
                    parts.append(f"{k}: {metrics[k]:.3f}")
            if not parts:
                return
            text = "  ".join(parts)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 1
            max_width = img.shape[1] - 16
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            if tw > max_width and tw > 0:
                scale = max(0.3, scale * (max_width / tw))
                (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            x = 8
            y = img.shape[0] - 6
            # stroke for readability
            cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), 3, lineType=cv2.LINE_AA)
            cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        try:
            m = _compute_metrics(A, B)
            _draw_metrics_footer(panel1, m, ["DICE", "IOU"])
            _draw_metrics_footer(panel2, m, ["PRECISION", "RECALL"])
            _draw_metrics_footer(panel3, m, ["S_MEASURE", "F1_SCORE"])
        except Exception:
            pass

        # Panel 4: probability heatmap + point (若提供 heatmap_rgb)
        if heatmap_rgb is not None:
            hm = heatmap_rgb
            if hm.ndim == 2:
                hm = cv2.cvtColor(hm, cv2.COLOR_GRAY2BGR)
            if hm.shape[:2] != (h, w):
                hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
            panel4 = hm.copy()
            # draw current point on heatmap
            if int(label) > 0:
                _draw_caret(panel4, cx, cy, size=6, color=(255,255,255), thickness=2)
            else:
                _draw_cross(panel4, cx, cy, size=6, color=(255,255,255), thickness=2)

        # Titles
        def _add_title(img: np.ndarray, text: str) -> None:
            band_h = max(20, min(48, img.shape[0] // 20))
            roi = img[0:band_h, :, :]
            overlay = roi.copy()
            cv2.rectangle(overlay, (0, 0), (img.shape[1] - 1, band_h - 1), (0, 0, 0), thickness=-1)
            cv2.addWeighted(overlay, 0.5, roi, 0.5, 0, dst=roi)
            org = (8, band_h - 6)
            font_scale = 0.4
            font_thickness = 1
            # Truncate text to fit within image width
            max_width = img.shape[1] - 16  # Leave some padding
            (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            if text_width > max_width:
                # Binary search to find the right truncation point
                ellipsis = "..."
                for i in range(len(text), 0, -1):
                    truncated = text[:i] + ellipsis
                    (trunc_width, _), _ = cv2.getTextSize(truncated, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    if trunc_width <= max_width:
                        text = truncated
                        break
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, lineType=cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)
        _add_title(panel1, "Overlay A (green) vs B (red)")
        _add_title(panel2, "Signed diff C: + (green), - (red)")
        _add_title(panel3, "Components + selected point")
        if heatmap_rgb is not None:
            _add_title(panel4, "Probability heatmap + selected point")

        panels = [panel1, panel2, panel3]
        if heatmap_rgb is not None:
            panels.append(panel4)
        canvas = np.concatenate(panels, axis=1)
        os.makedirs(os.path.dirname(dbg_out_path) or ".", exist_ok=True)
        _PIL.fromarray(canvas[..., ::-1]).save(dbg_out_path)
    except Exception:
        pass

def find_corresponding_image(
    images_dir: str,
    sample_stem: str,
    allowed_exts: Tuple[str, ...],
) -> Optional[str]:
    images_dir_abs = to_abs(images_dir)
    for ext in allowed_exts:
        candidate = os.path.join(images_dir_abs, sample_stem + ext)
        if os.path.isfile(candidate):
            return candidate
    # Fallback: scan directory and match by stem (handles mixed-case extensions or uncommon extensions)
    for name in os.listdir(images_dir_abs):
        if stem(name).lower() == sample_stem.lower():
            return os.path.join(images_dir_abs, name)
    return None

def calculate_next_point(gt_mask_path, sam_masks_dir, current_points_len, dbg_out_dir: Optional[str] = None):
    # 读取gt与pred
    try:
        gt_u8 = np.array(_PIL.open(to_abs(gt_mask_path)), dtype=np.uint8)
        if current_points_len > 0:
            last_mask_path = _find_latest_mask_path(sam_masks_dir, stem(gt_mask_path), current_points_len)
            if not last_mask_path or not os.path.isfile(last_mask_path):
                print(f"[WARN] Skip: last mask not found under {sam_masks_dir}")
                return None, None, None
            pred_u8 = np.array(_PIL.open(to_abs(last_mask_path)), dtype=np.uint8)
        else:
            pred_u8 = np.zeros_like(gt_u8)
    except Exception as e:
        print(f"[WARN] Skip: failed to read masks ({e})")
        return None, None, None

    # 若形状不同，将pred按最近邻缩放到gt尺寸
    h_g, w_g = gt_u8.shape[:2]
    h_p, w_p = pred_u8.shape[:2]
    if (h_g, w_g) != (h_p, w_p):
        try:
            # from PIL import Image as _PIL
            pred_u8 = np.array(_PIL.fromarray(pred_u8).resize((w_g, h_g), resample=_PIL.NEAREST), dtype=np.uint8)
        except Exception:
            pass

    ### For Debug Only:
    # if stem(gt_mask_path) == "BRATS_001_z0088":
    #     print("#Debug#")
    ### :For Debug Only

    # 计算差异的最大连通分量代表点
    dbg_path = None
    if dbg_out_dir:
        try:
            subdir = os.path.join(dbg_out_dir, stem(gt_mask_path))
            os.makedirs(subdir, exist_ok=True)
            dbg_path = os.path.join(subdir, f"dbg_{max(0, int(current_points_len))}.png")
        except Exception:
            dbg_path = None
    rep = largest_diff_component_representative(gt_u8, pred_u8, connectivity=2, min_area=0, dbg_out_path=dbg_path)
    if rep is None:
        print(f"[INFO] no diff component; skip append")
        return None, None, None
    x_next = int(rep["x"])  # 列
    y_next = int(rep["y"])  # 行
    label_next = 1 if int(rep["sign"]) > 0 else 0
    return x_next, y_next, label_next

def main() -> None:
    global _PIL
    args = parse_args()

    # Debug-only mode: read JSON, generate debug images but do not write JSON
    if args.debug_json:
        if not args.debug_output_dir:
            raise RuntimeError("--debug_output_dir is required when --debug_json is set")
        json_path = args.debug_json
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Debug JSON not found: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        try:
            arr = json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Debug JSON must be a JSON array: {e}")
        if not isinstance(arr, list):
            raise RuntimeError("Debug JSON root must be a JSON array")

        num_generated = 0
        for rec in arr:
            if not isinstance(rec, dict):
                continue

            image_path = rec.get("image")
            gt_mask_path = rec.get("gt_mask")
            sam_masks_dir = rec.get("sam_masks_dir")
            if not (image_path and sam_masks_dir):
                continue
            sample_stem = stem(image_path)
            ### For Debug Only:
            # if not sample_stem == "BRATS_001_z0088":
            #     continue
            ### :For Debug Only
            try:
                subdir = os.path.join(args.debug_output_dir, sample_stem)
                os.makedirs(subdir, exist_ok=True)
            except Exception:
                continue

            points = rec.get("points", [])
            labels = rec.get("labels", [])
            if not (isinstance(points, list) and isinstance(labels, list)):
                continue
            if len(points) != len(labels):
                continue

            gt_mask_u8 = np.array(_PIL.open(to_abs(gt_mask_path)), dtype=np.uint8)
            # Load base image if exists
            base_img_bgr = None
            try:
                img_arr = np.array(_PIL.open(to_abs(image_path)).convert("RGB"))
                base_img_bgr = img_arr[:, :, ::-1]
            except Exception:
                base_img_bgr = None
            # Load heatmap-{i}.png if exists
            heatmap_rgb = None
            try:
                hm_path = os.path.join(sam_masks_dir, sample_stem, f"heatmap-{i}.png")
                if os.path.isfile(hm_path):
                    hm = np.array(_PIL.open(to_abs(hm_path)).convert("RGB"))
                    heatmap_rgb = hm[:, :, ::-1]  # to BGR
            except Exception:
                heatmap_rgb = None
            # Load gt as all-zero array to preserve panel structure; we visualize differences of predicted masks only
            for i, (pt, lb) in enumerate(zip(points, labels)):
                # For step i, compare masks of ground truth and step i-1 (i==0 uses empty prev)
                if i == 0:
                    prev = np.zeros_like(gt_mask_u8, dtype=np.uint8)
                else:
                    try:
                        prev_path = os.path.join(sam_masks_dir, sample_stem, f"{i-1}.png")
                        prev = np.array(_PIL.open(to_abs(prev_path)).convert("L"), dtype=np.uint8) if os.path.isfile(prev_path) else np.zeros_like(gt_mask_u8, dtype=np.uint8)
                    except Exception:
                        prev = np.zeros_like(gt_mask_u8, dtype=np.uint8)
                # Render panels with provided point
                dbg_path = os.path.join(subdir, f"dbg_{i}.png")
                _render_diff_panels_with_point(
                    gt_mask_u8,
                    prev,
                    (float(pt[0]), float(pt[1])),
                    int(lb),
                    dbg_path,
                    base_image_bgr=base_img_bgr,
                    heatmap_rgb=heatmap_rgb,
                )
                num_generated += 1
        print(f"Generated debug images for {num_generated} steps into {args.debug_output_dir}")
        return

    # 追加模式：读取已有JSON数组，计算下一提示点并写回
    if args.appendto_jsonl:
        json_path = args.appendto_jsonl
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Append target not found: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        try:
            arr = json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Append target must be a JSON array: {e}")
        if not isinstance(arr, list):
            raise RuntimeError("Append target root must be a JSON array")

        updated = 0
        for idx, rec in enumerate(arr):
            if not isinstance(rec, dict):
                continue
            image_path = rec.get("image")
            gt_mask_path = rec.get("gt_mask")
            sam_masks_dir = rec.get("sam_masks_dir")
            points_list = rec.get("points", [])
            labels_list = rec.get("labels", [])

            if not (image_path and gt_mask_path and sam_masks_dir):
                print(f"[WARN] Skip idx {idx}: missing fields")
                continue

            x_next, y_next, label_next = calculate_next_point(
                gt_mask_path,
                sam_masks_dir,
                len(points_list),
                dbg_out_dir=args.debug_output_dir,
            )
            if x_next is None or y_next is None or label_next is None:
                continue

            # 追加points与labels
            if not isinstance(points_list, list):
                points_list = []
            if not isinstance(labels_list, list):
                labels_list = []
            points_list.append([x_next, y_next])
            labels_list.append(int(label_next))
            rec["points"] = points_list
            rec["labels"] = labels_list
            updated += 1

        # 回写同一文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(arr, f, indent=2, ensure_ascii=False)
        print(f"Appended next prompts to {json_path} (updated {updated} items)")
        return

    # 否则：首写模式，生成首个提示点（与之前逻辑一致）
    if not args.images_dir or not args.masks_dir or not args.output_jsonl:
        raise RuntimeError("images_dir, masks_dir and output_jsonl are required for initial generation mode")

    images_dir = args.images_dir
    masks_dir = args.masks_dir
    output_jsonl = args.output_jsonl
    allowed_image_exts = tuple(e.strip().lower() for e in args.image_exts.split(",") if e.strip())

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    # Visualization output disabled
    # if args.viz_dir:
    #     os.makedirs(args.viz_dir, exist_ok=True)

    # Gather masks as the source of truth
    mask_files = list_files_with_exts(masks_dir, allowed_exts=(".png",))
    if len(mask_files) == 0:
        raise RuntimeError(f"No mask PNG files found under {masks_dir}")

    num_written = 0
    num_skipped_empty = 0
    num_missing_images = 0
    num_size_mismatch = 0

    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for mask_name in mask_files:
            sample_stem = stem(mask_name)
            mask_path = os.path.join(masks_dir, mask_name)
            
            # Load mask
            try:
                mask_img = _PIL.open(to_abs(mask_path))
                mask_u8 = np.array(mask_img, dtype=np.uint8)
            except Exception as e:
                print(f"[WARN] Skipping unreadable mask: {mask_path} ({e})")
                continue
            
            x_next, y_next, label_next = calculate_next_point(
                mask_path,
                None,
                0,
                dbg_out_dir=args.debug_output_dir,
            )
            if x_next is None or y_next is None or label_next is None:
                if args.skip_empty:
                    num_skipped_empty += 1
                    continue
                else:
                    # For empty masks and skip_empty=False, place centroid at image center
                    h, w = mask_u8.shape[:2]
                    centroid = (float(w - 1) / 2.0, float(h - 1) / 2.0)
            centroid = (x_next, y_next)

            # Find corresponding image
            img_path = find_corresponding_image(images_dir, sample_stem, allowed_image_exts)
            if img_path is None:
                num_missing_images += 1
                print(f"[WARN] Missing image for mask stem '{sample_stem}' in {images_dir}")
                continue
           
            image_field = os.path.abspath(img_path) if args.abs_paths else img_path
            mask_field = os.path.abspath(mask_path) if args.abs_paths else mask_path
            record = {"image": image_field, "gt_mask": mask_field, "points": [[float(centroid[0]), float(centroid[1])]], "labels": [1]}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1

            # Visualization
            if False and args.viz_dir:
                try:
                    # Load image RGB
                    with _PIL.open(to_abs(img_path)) as im_rgb:
                        im_rgb = im_rgb.convert("RGB")
                        w_im, h_im = im_rgb.size
                        
                        # Ensure mask size matches visualization image
                        m = mask_u8
                        if (m.shape[1], m.shape[0]) != (w_im, h_im):
                            # resize nearest for mask
                            m_img = Image.fromarray(m)
                            m_img = m_img.resize((w_im, h_im), resample=Image.NEAREST)
                            m = np.array(m_img, dtype=np.uint8)
                        
                        # Create colored mask overlay
                        overlay = Image.new("RGBA", (w_im, h_im), (0, 0, 0, 0))
                        mask_alpha = int(max(0.0, min(1.0, args.viz_alpha)) * 255)
                        color = (255, 0, 0, mask_alpha)
                        # Build alpha mask where foreground>0
                        fg = (m > 0).astype(np.uint8) * mask_alpha
                        alpha_img = Image.fromarray(fg, mode="L")
                        color_img = Image.new("RGBA", (w_im, h_im), color)
                        overlay.paste(color_img, (0, 0), mask=alpha_img)
                        
                        # Compose overlay on image
                        comp = im_rgb.convert("RGBA")
                        comp = Image.alpha_composite(comp, overlay)
                        
                        # Draw the deepest point
                        draw = ImageDraw.Draw(comp)
                        px, py = float(centroid[0]), float(centroid[1])
                        r = max(1, int(args.viz_radius))
                        draw.ellipse((px - r, py - r, px + r, py + r), outline=(0, 255, 0, 255), width=2)
                        
                        # Save
                        out_name = f"{sample_stem}.png"
                        comp.convert("RGB").save(os.path.join(args.viz_dir, out_name))
                except Exception as e:
                    print(f"[WARN] Visualization failed for {img_path}: {e}")

    print(f"Wrote {num_written} samples to {output_jsonl}")
    if num_skipped_empty:
        print(f"Skipped empty masks: {num_skipped_empty}")
    if num_missing_images:
        print(f"Masks with missing images: {num_missing_images}")
    if num_size_mismatch:
        print(f"Image/Mask size mismatches adjusted: {num_size_mismatch}")


if __name__ == "__main__":
    main()



```

已更新调试可视化，新增第4幅面板“概率热力图 + 当前提示点”，便于核对提示点是否落在热区上。逻辑如下：
- 在 `--debug_json` 模式中，遍历记录的每一步 i：
  - 读取 `sam_masks_dir/<stem>/{i-1}.png` 作为上一步掩膜（i=0 时用全零）。
  - 若存在 `sam_masks_dir/<stem>/heatmap-{i}.png`，加载为 `heatmap_rgb`。
  - 调用 `_render_diff_panels_with_point(..., heatmap_rgb=...)`，将热力图叠加为第4幅面板并在其上绘制当前提示点。

关键改动点：
- 函数签名扩展：`_render_diff_panels_with_point(..., heatmap_rgb: Optional[np.ndarray] = None)`。
- 若提供 `heatmap_rgb`：
  - 统一 resize 到与 A/B 相同尺寸；
  - 将当前提示点绘制到热力图上；
  - 追加标题“Probability heatmap + selected point”，并拼接为4联图。
- 在 debug 主循环中，尝试加载 `heatmap-{i}.png` 并传入上述函数。

真实可用的调用示例（与你现有流程兼容）：
1) 先用脚本生成掩膜与热力图（脚本已更新加入 `--heatmap_model` 参数）：
```bash
/root/autodl-tmp/envs/seg-r1/bin/python seg-rl/predict_points_and_sam.sh ptr_251023 12 pretrained/points_predictor-251001-160epochs.pt cuda
```
这将生成：
- JSON: outputs/braintumour/ptr_251023.jsonl
- 掩膜: outputs/braintumour/ptr_251023_sam_masks/<stem>/{i}.png
- 热力图: outputs/braintumour/ptr_251023_sam_masks/<stem>/heatmap-{i}.png

2) 运行本调试可视化，生成四联图：
```bash
/opt/anaconda3/envs/seg-r1/bin/python seg-rl/annotator/gen_point_jsonl_from_masks.py \
  --debug_json outputs/braintumour/ptr_251023.jsonl \
  --debug_output_dir outputs/braintumour/dbg_gen_points
```
输出目录示例：
- outputs/braintumour/dbg_gen_points/BRATS_001_z0032/dbg_0.png
- 其中包含四幅并排面板：A/B叠加、差异图、连通域+点、概率热力图+点

如需热力图叠加到底图一起显示，可后续扩展一个 overlay 版本；目前保持彩色热力图单独面板，避免与 A/B 叠加信息冲突。

```shellscript
/root/autodl-tmp/envs/seg-r1/bin/python seg-rl/predict_points_and_sam.sh ptr_251023 12 pretrained/points_predictor-251001-160epochs.pt cuda
```

```shellscript
/opt/anaconda3/envs/seg-r1/bin/python seg-rl/annotator/gen_point_jsonl_from_masks.py \
  --debug_json outputs/braintumour/ptr_251023.jsonl \
  --debug_output_dir outputs/braintumour/dbg_gen_points
```

---

**User**

不对，我执行以下命令，看到在输出目录产生的可视化图像都还是只有原先的三列，没有四列。检查一下是怎么回事：

---

**Cursor**

I'll quickly inspect a couple of records in outputs/braintumour/ptr_251023.jsonl to verify the 'sam_masks_dir' and file naming, and confirm the expected 'heatmap-{i}.png' paths.

We noticed your debug images still show only 3 panels because the segmenter skips processing when masks already exist (due to --skip_existing), so the new heatmap files aren't generated for those steps. I'll update the segmenter to still generate missing heatmaps even when skipping existing masks, then you can rerun the segment step (no deletion needed) to backfill heatmaps. Now I'll apply this change.

```python
#!/usr/bin/env python3

"""
使用SAM2从点提示进行图像分割

读取由点提示组成的输入（支持JSON数组或JSONL逐行），使用SAM2进行分割，
将预测mask保存为灰度图像，并按要求生成/追加每幅图像对应的bbox日志。

功能:
- 从点提示生成图像分割mask
- 自动计算每个mask的最小包围盒
- 可选择输出包含mask路径和包围盒坐标的JSON文件

输入JSON格式（数组）或JSONL（逐行）:
    新格式: {"image": "/path/img.jpg", "points": [[x1,y1], [x2,y2]], "labels": [1, 0]}
    旧格式: {"image": "/path/img.jpg", "x": x1, "y": y1}

输出:
    - 预测mask按图像stem创建子目录保存为灰度图：output_dir/<stem>/<k>.png
      其中k为提示点序列长度-1；像素值：0=背景，255=前景
    - 在output_dir生成<stem>.jsonl（JSON对象，非数组）。该文件内容保持为：
      {"count": N, "bboxes": [[x1,y1,x2,y2], ...]}。每次新增一个mask，仅向
      bboxes追加一个新的bbox并将count递增，无需重写历史bbox。
    - 将输入JSON数组拷贝到 --json_output，并为每条加入/覆写
      "sam_masks_dir": output_dir

依赖项:
    - SAM2 (Segment Anything Model 2)
    - torch, PIL, opencv-python, numpy

安装SAM2:
    git clone https://github.com/facebookresearch/segment-anything-2.git third_party/sam2
    cd third_party/sam2 && pip install -e .

下载模型:
    # 下载SAM2.1 Hiera Large模型 (~900MB)
    wget -O third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

基础用法:
    python seg-rl/sam2_segment_from_points.py \
      --input_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour.jsonl \
      --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/pretrain_gt_masks \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda \

带JSON输出:
    python seg-rl/sam2_segment_from_points.py \
      --input_jsonl /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour.jsonl \
      --json_output /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/segrl_pretrain_braintumour.jsonl \
      --output_dir /root/autodl-tmp/datasets/seg_r1_md/Task01_BrainTumour/pretrain_gt_masks \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda \
      --resize 512 512 \
      --skip_existing

    python seg-rl/sam2_segment_from_points.py \
      --input_jsonl outputs/braintumour/pred_points-e160-251019.jsonl \
      --json_output outputs/braintumour/pred_points-e160-251019.jsonl \
      --output_dir outputs/braintumour/pred_masks-e160-251019 \
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda \
      --resize 512 512 \
      --skip_existing

    python seg-rl/sam2_segment_from_points.py \
      --input_jsonl outputs/braintumour/pred_points-sft_e185-251018.jsonl \
      --json_output outputs/braintumour/pred_points-sft_e185-251018.jsonl \
      --output_dir outputs/braintumour/pred_masks-sft_e185-251018 \
      --sam_checkpoint /root/autodl-tmp/works/Seg-R0/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device cuda \
      --resize 512 512 \
      --skip_existing
 
    /opt/anaconda3/envs/seg-r1/bin/python seg-rl/sam2_segment_from_points.py \
      --input_jsonl datasets/seg_r1_md/Task01_BrainTumour/pred_points-251001.jsonl \
      --json_output datasets/seg_r1_md/Task01_BrainTumour/pred_points-251001.jsonl \
      --output_dir outputs/braintumour/pretrain_pred_masks-251001 \
      --sam_checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --device mps \
      --resize 512 512 \
      --skip_existing

详细参数及输出格式说明请参考：docs/cursor_pretrain_flow_all_2025092801.md）：
    

"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# 导入热力图点预测模型与可视化工具（用于生成 heatmap-{i}.png）
try:
    from heatmap.model import ModelConfig as _HMModelConfig, PointHeatmapModel as _PointHeatmapModel  # type: ignore
    import torch.nn.functional as _F  # type: ignore
    from heatmap.utils import heatmap_to_pil as _heatmap_to_pil  # type: ignore
    _HEATMAP_AVAILABLE = True
except Exception:
    # 若用户未提供热力图权重或未安装依赖，则仅跳过热力图生成功能
    _HEATMAP_AVAILABLE = False

# 添加sam2路径
sys.path.append(str(Path(__file__).parent.parent / "third_party" / "sam2"))

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"Error importing SAM2: {e}")
    print("Please ensure SAM2 is properly installed and the path is correct.")
    sys.exit(1)


def _print_progress(num_processed: int, num_skipped: int, num_errors: int, note: str) -> None:
    """在同一行输出进度信息"""
    msg = f"[Progress] processed={num_processed} skipped={num_skipped} errors={num_errors} | {note}"
    print(f"\r{msg}", end="", flush=True)


def to_abs(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.abspath(path) if not os.path.isabs(path) else path


class SAMWrapper:
    """SAM2包装器，用于图像分割预测"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """初始化SAM2模型和预测器
        
        Args:
            model_path: SAM2模型检查点路径
            device: 运行设备 (e.g. "cuda", "cuda:0", "cpu")
                   如果为None，将自动检测可用设备
        """
        # 自动检测设备
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        # SAM2配置文件路径
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam_model = build_sam2(model_cfg, model_path, device=self.device) 
        # # 检查配置文件是否存在
        # config_path = Path(__file__).parent.parent / "third_party" / "sam2" / "sam2" / model_cfg
        # if not config_path.exists():
        #     # 尝试相对于当前目录
        #     config_path = Path(model_cfg)
        #     if not config_path.exists():
        #         raise FileNotFoundError(f"SAM2 config file not found: {model_cfg}")
        # sam_model = build_sam2(str(config_path), model_path)
        
        # # 移动到指定设备
        # sam_model = sam_model.to(self.device)
        
        # 初始化预测器
        self.predictor = SAM2ImagePredictor(sam_model)
        self.last_mask = None
        
    def predict(self, 
                image: PILImage.Image, 
                points: Optional[List[Tuple[int, int]]] = None, 
                labels: Optional[List[int]] = None,
                bbox: Optional[List[int]] = None) -> Tuple[np.ndarray, float]:
        """使用给定提示运行SAM2预测
        
        Args:
            image: 输入PIL图像
            points: 点坐标列表 [(x,y), ...]
            labels: 点标签列表 (1=前景, 0=背景)
            bbox: 可选边界框 [x1,y1,x2,y2]
            
        Returns:
            (predicted_mask, confidence_score)的元组
        """
        input_points = np.array(points) if points else None
        input_labels = np.array(labels) if labels else None
        input_bboxes = np.array([bbox]) if bbox else None

        # 转换为numpy数组
        image_np = np.array(image)
        
        # 确保是RGB格式
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image_np
        
        # 设置图像
        self.predictor.set_image(rgb_image)
        
        # 预测
        mask_pred, score, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_bboxes,
            multimask_output=False,
        )
        
        self.last_mask = mask_pred[0]
        return mask_pred[0], score[0]


def validate_and_extract_points(obj: Dict[str, Any], line_num: int) -> Tuple[bool, Optional[List[Tuple[float, float]]], Optional[List[int]]]:
    """验证并提取JSON条目中的点和标签
    
    Args:
        obj: JSON对象
        line_num: 行号（用于错误报告）
        
    Returns:
        (is_valid, points, labels)的元组
    """
    if "image" not in obj:
        print(f"[WARN] Line {line_num}: Missing 'image' field")
        return False, None, None
        
    # 检查格式：新格式有"points"和"labels"，旧格式有"x"和"y"
    if "points" in obj and "labels" in obj:
        # 新格式
        points = obj["points"]
        labels = obj["labels"]
        
        if not isinstance(points, list) or not isinstance(labels, list):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must be lists")
            return False, None, None
            
        if len(points) != len(labels):
            print(f"[WARN] Line {line_num}: 'points' and 'labels' must have same length")
            return False, None, None
        
        # 验证点格式
        valid_points = []
        valid_labels = []
        for point, label in zip(points, labels):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                print(f"[WARN] Line {line_num}: Invalid point format {point}")
                continue
            valid_points.append((float(point[0]), float(point[1])))
            valid_labels.append(int(label))
            
        if len(valid_points) == 0:
            print(f"[WARN] Line {line_num}: No valid points found")
            return False, None, None
            
        return True, valid_points, valid_labels
        
    elif "x" in obj and "y" in obj:
        # 旧格式 - 假设为正类点
        x = float(obj["x"])
        y = float(obj["y"])
        return True, [(x, y)], [1]
        
    else:
        print(f"[WARN] Line {line_num}: Missing coordinate data (need 'points'+'labels' or 'x'+'y')")
        return False, None, None


def get_output_path(image_path: str, output_dir: str) -> str:
    """生成输出mask文件路径（旧：单文件路径；新：保留但未使用）
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        
    Returns:
        输出mask文件路径
    """
    image_name = Path(image_path).stem
    return os.path.join(output_dir, f"{image_name}.png")


def get_mask_dir_for_image(image_path: str, output_dir: str) -> str:
    """返回该图像对应的mask子目录路径 output_dir/<stem>"""
    image_name = Path(image_path).stem
    return os.path.join(output_dir, image_name)


def get_mask_index_from_points(points: List[Tuple[float, float]]) -> int:
    """根据点序列长度返回mask索引：len(points)-1"""
    return max(0, int(len(points) - 1))


def get_mask_path_for_points(image_path: str, output_dir: str, points: List[Tuple[float, float]]) -> str:
    """输出mask路径：output_dir/<stem>/<k>.png，k=len(points)-1"""
    mask_dir = get_mask_dir_for_image(image_path, output_dir)
    os.makedirs(mask_dir, exist_ok=True)
    k = get_mask_index_from_points(points)
    return os.path.join(mask_dir, f"{k}.png")


def append_state_log(output_dir: str, image_path: str, bbox: Tuple[int, int, int, int]) -> None:
    """在 <output_dir>/<stem>.jsonl 中维护单个JSON对象：
    {"count": N, "bboxes": [[x1,y1,x2,y2], ...]}
    仅追加当前bbox并将count递增。不会重新读取或计算历史bbox。
    """
    stem = Path(image_path).stem
    log_path = os.path.join(output_dir, f"{stem}.jsonl")

    record: Dict[str, Any] = {"count": 0, "bboxes": []}
    if os.path.isfile(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "count" in parsed and "bboxes" in parsed:
                        record = parsed
        except Exception:
            pass

    # 追加当前bbox并递增count
    record.setdefault("bboxes", [])
    record["bboxes"].append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
    record["count"] = int(record.get("count", 0)) + 1

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False)


def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """计算mask的最小包围盒
    
    Args:
        mask: 二值mask数组
        
    Returns:
        (x_min, y_min, x_max, y_max) 包围盒坐标
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask_binary)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        # 如果没有前景像素，返回空包围盒
        return 0, 0, 0, 0
    
    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))
    
    return x_min, y_min, x_max, y_max


def save_mask_as_grayscale(mask: np.ndarray, output_path: str) -> None:
    """将mask保存为灰度图像
    
    Args:
        mask: 二值mask数组
        output_path: 输出文件路径
    """
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 转换为灰度值 (0=背景, 255=前景)
    mask_gray = mask_binary * 255
    
    # 保存
    cv2.imwrite(output_path, mask_gray)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Use SAM2 to segment images from point prompts")
    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Input JSON array (preferred) or JSONL file containing image paths and point coordinates")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory to save mask images")
    p.add_argument("--sam_checkpoint", type=str, required=True,
                   help="Path to SAM2 model checkpoint")
    p.add_argument("--device", type=str, default=None,
                   help="Device to run on (cuda/cpu). Auto-detect if not specified")
    p.add_argument("--config_path", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml",
                   help="Path to SAM2 config file")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip processing if output file already exists")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                   help="Resize input images to specified size [width height]")
    p.add_argument("--json_output", type=str, default=None,
                   help="Path to write the copied input JSON array with added 'sam_masks_dir'")
    # 额外：生成提示点概率热力图（heatmap-{i}.png）相关参数
    p.add_argument("--heatmap_model", type=str, default=None,
                   help="Path to heatmap predictor checkpoint (.pt); if set, also save heatmap-{i}.png")
    p.add_argument("--heatmap_tau", type=float, default=1.0,
                   help="Softmax temperature for probability heatmap generation")
    p.add_argument("--heatmap_size", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                   help="Inference size [width height] for heatmap model; default = use --resize or native size")
    return p.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.isfile(to_abs(args.input_jsonl)):
        print(f"Error: Input JSONL file not found: {args.input_jsonl}")
        return 1
    
    # 检查SAM2检查点
    if not os.path.isfile(to_abs(args.sam_checkpoint)):
        print(f"Error: SAM2 checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    # 创建输出目录
    args.output_dir = to_abs(args.output_dir) or args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化SAM2
    print(f"Initializing SAM2 with checkpoint: {to_abs(args.sam_checkpoint)}")
    print(f"Using device: {args.device or 'auto-detect'}")
    
    try:
        sam_wrapper = SAMWrapper(to_abs(args.sam_checkpoint), args.device)
        print("SAM2 initialized successfully")
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return 1
    
    # 可选：初始化热力图点预测模型
    hm_model = None
    hm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if (args.device is None) else torch.device(args.device)
    if _HEATMAP_AVAILABLE and args.heatmap_model:
        try:
            cfg = _HMModelConfig(backbone="unet_s", pretrained=False, main_in_channels=3, cond_in_channels=1)
            hm_model = _PointHeatmapModel(cfg).to(hm_device)
            ckpt = torch.load(to_abs(args.heatmap_model), map_location="cpu")
            sd = ckpt.get("model", ckpt)
            hm_model.load_state_dict(sd, strict=False)
            hm_model.eval()
            print(f"Heatmap model loaded: {to_abs(args.heatmap_model)}")
        except Exception as e:
            print(f"[WARN] Failed to load heatmap model: {e}. Skip heatmap generation.")
            hm_model = None

    # 读取输入（优先按JSON数组解析；失败则按JSONL逐行解析）
    print(f"Processing input: {to_abs(args.input_jsonl)}")
    
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    
    # 解析输入记录列表 records: List[Dict]
    records: List[Dict[str, Any]] = []
    try:
        with open(to_abs(args.input_jsonl), 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    records = parsed
                else:
                    print("[ERROR] Input JSON root must be an array")
                    return 1
            else:
                # JSONL逐行
                for line_num, line in enumerate(content.splitlines(), 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                        num_errors += 1
                        continue
    except Exception as e:
        print(f"Error reading input: {e}")
        return 1

    for idx, obj in enumerate(records, 1):
        line_num = idx

        # 验证并提取点信息
        is_valid, points, labels = validate_and_extract_points(obj, line_num)
        if not is_valid:
            num_errors += 1
            continue

        image_path = to_abs(obj["image"]) or obj["image"]
        ### For Debug Only:
        # stem = Path(image_path).stem
        # if not stem == "BRATS_001_z0088":
        #     continue
        ### :For Debug Only

        # 检查图像文件是否存在
        if not os.path.isfile(image_path):
            print(f"[ERROR] Line {line_num}: Image file not found: {image_path}")
            num_errors += 1
            continue

        # 生成输出路径（子目录/<k>.png）
        output_path = get_mask_path_for_points(image_path, args.output_dir, points)

        # 检查是否跳过已存在的文件
        mask_existed = os.path.isfile(output_path)
        if args.skip_existing and mask_existed:
            # 若请求生成热力图且当前步的 heatmap 不存在，仍然补齐保存 heatmap-{k}.png
            try:
                if hm_model is not None:
                    k = get_mask_index_from_points(points)
                    hm_out_path = os.path.join(get_mask_dir_for_image(image_path, args.output_dir), f"heatmap-{k}.png")
                    if not os.path.isfile(hm_out_path):
                        # 载入图像与尺寸（与常规路径一致）
                        image = PILImage.open(to_abs(image_path)).convert("RGB")
                        orig_w, orig_h = image.size
                        if args.resize:
                            resize_w, resize_h = int(args.resize[0]), int(args.resize[1])
                            image_for_pred = image.resize((resize_w, resize_h), PILImage.BILINEAR)
                            points_resized = [(px * (resize_w / float(orig_w)), py * (resize_h / float(orig_h))) for (px, py) in points]
                        else:
                            image_for_pred = image
                            points_resized = points

                        # 选择热力图推理尺寸
                        if args.heatmap_size is not None:
                            hm_w, hm_h = int(args.heatmap_size[0]), int(args.heatmap_size[1])
                        elif args.resize is not None:
                            hm_w, hm_h = int(args.resize[0]), int(args.resize[1])
                        else:
                            hm_w, hm_h = orig_w, orig_h

                        # 构造上一时刻掩膜
                        if k <= 0:
                            prev_gray_pil = PILImage.new("L", (hm_w, hm_h), 0)
                        else:
                            prev_path = os.path.join(get_mask_dir_for_image(image_path, args.output_dir), f"{k-1}.png")
                            if os.path.isfile(prev_path):
                                prev_gray_pil = PILImage.open(prev_path).convert("L").resize((hm_w, hm_h), PILImage.NEAREST)
                            else:
                                prev_pts = points_resized[:-1]
                                prev_labs = labels[:-1] if labels else None
                                prev_mask_np, _ = sam_wrapper.predict(
                                    image_for_pred.resize((hm_w, hm_h), PILImage.BILINEAR), prev_pts, prev_labs
                                )
                                prev_gray_pil = PILImage.fromarray((prev_mask_np.astype(np.uint8) * 255), mode="L")

                        # 模型前处理与推理
                        from torchvision.transforms import functional as TF  # type: ignore
                        rgb_for_hm = image.resize((hm_w, hm_h), PILImage.BILINEAR)
                        rgb_t = TF.to_tensor(rgb_for_hm)
                        rgb_t = TF.normalize(rgb_t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)).unsqueeze(0)
                        g_t = TF.to_tensor(prev_gray_pil)
                        g_t = ((g_t - 0.5) / 0.5).unsqueeze(0)
                        with torch.no_grad():
                            logits, _ = hm_model(rgb_t.to(hm_device), g_t.to(hm_device))
                            b, c, Hh, Wh = logits.shape
                            p = _F.softmax(logits.view(b, -1) / max(float(args.heatmap_tau), 1e-6), dim=1)
                            prob = p.view(Hh, Wh).detach().cpu().float().numpy()
                        hm_img = _heatmap_to_pil(prob)
                        if (hm_w, hm_h) != (orig_w, orig_h):
                            hm_img = hm_img.resize((orig_w, orig_h), PILImage.BILINEAR)
                        os.makedirs(os.path.dirname(hm_out_path), exist_ok=True)
                        hm_img.save(hm_out_path)
            except Exception as _:
                pass

            _print_progress(num_processed, num_skipped + 1, num_errors,
                            f"skip line {line_num}: {Path(output_path).name} exists")
            num_skipped += 1
            continue

        try:
            # 加载图像
            image = PILImage.open(to_abs(image_path)).convert("RGB")
            orig_w, orig_h = image.size

            # 若指定resize，则按比例缩放点坐标并对图像进行resize
            if args.resize:
                resize_w, resize_h = int(args.resize[0]), int(args.resize[1])
                scale_x = float(resize_w) / float(orig_w)
                scale_y = float(resize_h) / float(orig_h)
                points_resized = [(px * scale_x, py * scale_y) for (px, py) in points]
                image_for_pred = image.resize((resize_w, resize_h), PILImage.BILINEAR)
            else:
                points_resized = points
                image_for_pred = image
                scale_x = 1.0
                scale_y = 1.0

            # 运行SAM2预测（在处理后的分辨率上）
            mask, confidence = sam_wrapper.predict(image_for_pred, points_resized, labels)

            # 如果做了resize，则将mask缩放回原图尺寸，并在原图尺寸上计算bbox
            if args.resize:
                mask_binary = (mask > 0).astype(np.uint8)
                mask_orig_size = cv2.resize(mask_binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                x_min, y_min, x_max, y_max = calculate_bounding_box(mask_orig_size)
                # 保存缩放回原尺寸的mask
                save_mask_as_grayscale(mask_orig_size, output_path)
            else:
                # 未resize，直接使用原mask与bbox
                x_min, y_min, x_max, y_max = calculate_bounding_box(mask)
                save_mask_as_grayscale(mask, output_path)

            # 追加/更新该图像的日志（单一JSON对象，追加bbox并递增count）
            try:
                append_state_log(args.output_dir, image_path, (x_min, y_min, x_max, y_max))
            except Exception as e:
                print(f"[WARN] Failed to append state log for {image_path}: {e}")

            # 可选：生成并保存提示点概率热力图 heatmap-{k}.png
            # 语义：对于第 i 步（k=i），热力图来源于“图像 + 上一步的预测掩膜（i-1步；若i=0则为全零掩膜）”。
            if hm_model is not None:
                try:
                    # 选择热力图推理尺寸（优先 heatmap_size，其次使用 --resize，最后使用原始尺寸）
                    if args.heatmap_size is not None:
                        hm_w, hm_h = int(args.heatmap_size[0]), int(args.heatmap_size[1])
                    elif args.resize is not None:
                        hm_w, hm_h = int(args.resize[0]), int(args.resize[1])
                    else:
                        hm_w, hm_h = orig_w, orig_h

                    # 构造“上一时刻”的灰度掩膜（若 i>0 且存在文件，则直接读；否则为了稳健性直接用上一时刻的SAM预测）
                    k = get_mask_index_from_points(points)
                    prev_gray_pil: Optional[PILImage.Image]
                    if k <= 0:
                        prev_gray_pil = PILImage.new("L", (hm_w, hm_h), 0)
                    else:
                        mask_dir = get_mask_dir_for_image(image_path, args.output_dir)
                        prev_path = os.path.join(mask_dir, f"{k-1}.png")
                        if os.path.isfile(prev_path):
                            prev_gray_pil = PILImage.open(prev_path).convert("L").resize((hm_w, hm_h), PILImage.NEAREST)
                        else:
                            # 若上一步掩膜文件缺失，则用上一时刻点再次运行一次SAM以得到上一时刻掩膜
                            prev_pts = points_resized[:-1]
                            prev_labs = labels[:-1] if labels else None
                            prev_mask_np, _ = sam_wrapper.predict(
                                image_for_pred.resize((hm_w, hm_h), PILImage.BILINEAR), prev_pts, prev_labs
                            )
                            prev_gray = (prev_mask_np.astype(np.uint8) * 255)
                            prev_gray_pil = PILImage.fromarray(prev_gray, mode="L")

                    # 构造模型输入张量：RGB 与 灰度条件（范围与归一化与训练一致）
                    # 注意：这里独立于 SAM2 的 resize，确保热力图与指定 hm_w/h 对齐
                    from torchvision.transforms import functional as TF  # type: ignore
                    rgb_for_hm = image.resize((hm_w, hm_h), PILImage.BILINEAR)
                    rgb_t = TF.to_tensor(rgb_for_hm)
                    rgb_t = TF.normalize(rgb_t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)).unsqueeze(0)
                    gray_pil = prev_gray_pil if prev_gray_pil is not None else PILImage.new("L", (hm_w, hm_h), 0)
                    g_t = TF.to_tensor(gray_pil)
                    g_t = ((g_t - 0.5) / 0.5).unsqueeze(0)

                    with torch.no_grad():
                        logits, _ = hm_model(rgb_t.to(hm_device), g_t.to(hm_device))  # [1,1,H,W]
                        # 将 logits 转为概率热力图（按像素 softmax），与训练/可视化一致
                        b, c, Hh, Wh = logits.shape
                        p = _F.softmax(logits.view(b, -1) / max(float(args.heatmap_tau), 1e-6), dim=1)
                        prob = p.view(Hh, Wh).detach().cpu().float().numpy()

                    # 映射为彩色图并按原图尺寸保存，便于与 {k}.png 对齐查看
                    hm_img = _heatmap_to_pil(prob)
                    if (hm_w, hm_h) != (orig_w, orig_h):
                        hm_img = hm_img.resize((orig_w, orig_h), PILImage.BILINEAR)
                    heatmap_path = os.path.join(get_mask_dir_for_image(image_path, args.output_dir), f"heatmap-{k}.png")
                    os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
                    hm_img.save(heatmap_path)
                except Exception as e:
                    print(f"[WARN] Failed to save heatmap for {image_path}: {e}")

            _print_progress(num_processed + 1, num_skipped, num_errors,
                            f"ok line {line_num}: {Path(output_path).name} conf={confidence:.3f}")
            num_processed += 1

        except Exception as e:
            _print_progress(num_processed, num_skipped, num_errors + 1,
                            f"error line {line_num}: {Path(image_path).name}")
            num_errors += 1
            continue
    
    # 将输入数组复制到json_output并添加/覆盖 sam_masks_dir
    if args.json_output:
        try:
            # 若前面用JSONL解析得到records，则以records为准写出数组
            out_records = []
            for obj in records:
                if isinstance(obj, dict):
                    obj2 = dict(obj)
                    obj2["sam_masks_dir"] = args.output_dir
                    out_records.append(obj2)
            with open(args.json_output, 'w', encoding='utf-8') as jf:
                json.dump(out_records, jf, indent=2, ensure_ascii=False)
            print(f"Copied input to json_output with 'sam_masks_dir': {args.json_output}")
        except Exception as e:
            print(f"[ERROR] Failed to save json_output: {e}")
            return 1
    
    # 打印总结（先补换行清空进度行）
    print()
    print(f"Processing completed:")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Errors: {num_errors}")
    print(f"  Output directory: {args.output_dir}")
    if args.json_output:
        print(f"  JSON output: {args.json_output}")
    
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

```

- 原因：你此前已经生成过掩膜，运行分割脚本时启用了 `--skip_existing`，因此脚本此前会直接跳过整步，未生成 `heatmap-{i}.png`。调试器在找不到热力图时只会输出三列。
- 修复：我已更新 `seg-rl/sam2_segment_from_points.py`，即使 `--skip_existing` 跳过已存在的 `{i}.png` 掩膜时，也会检测并补齐缺失的 `heatmap-{i}.png`（在不重算掩膜的情况下只补热力图）。

下一步：
- 直接再次运行你之前的分割步骤（无需删除已有输出），脚本会为缺失的步数补齐 `heatmap-{i}.png`。
- 然后再运行你的调试可视化命令（gen_point_jsonl_from_masks.py 的 --debug_json 模式），就会看到四列面板。

