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
- **PEFT（参数高效微调）**: `seg-rl/peft/` **[NEW 2025-11-08]**
  - Late LoRA 与 SAM2: `lora_sam2.py`（`LoRALinear`；`LoRASAM2Wrapper`；LoRA注入、特征提取、掩模预测）
  - 基于SAM特征的点预测网络: `point_predictor_peft.py`（`MaskEncoder`；`FiLMFusion`/`ConcatFusion`；`Decoder`；`PointPredictorFromSAMFeatures`）
  - 数据加载: `datasets_peft.py`（`PEFTPointDataset` 支持k=0全零掩模；`PEFTPointDatasetForEval`）
  - 监督预训练: `train_supervised_peft.py`（双优化器、AMP、自动续传）
  - GRPO强化学习: `train_grpo_peft.py`（`PolicyNetwork`；rollout；GRPO更新）
  - 评估: `eval_peft_model.py`（完整rollout评估）
  - 工具: `utils_peft.py`（metrics、checkpoint管理、可视化辅助）
  - 文档: `README.md`（技术细节）；`../README_PEFT_EXPERIMENT_GUIDE.md`（完整实验指南）；`../PEFT_COMMANDS_CHEATSHEET.md`（命令速查）
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
    - 主题：训练中出现的 `ValueError` 的定位与修复；文档内多次出现"SAM2 integration"小节，涉及依赖与导入稳健性。

- 2025-11-08 **[NEW PEFT Implementation]**
  - **PEFT（参数高效微调）模块完整实现**
    - 主题：集成Late LoRA到SAM2，实现基于SAM特征的点预测网络，包含监督预训练和GRPO强化学习两阶段训练。
    - **快速开始**：见 `README_PEFT_EXPERIMENT_GUIDE.md`（完整实验指南，包含环境准备、数据生成、训练、评估的详细命令）。
    - **命令速查**：见 `PEFT_COMMANDS_CHEATSHEET.md`（常用命令快速参考）。
    - 技术文档：见 `seg-rl/peft/README.md`（架构设计、模块说明、超参数建议）。
    - 理论参考：Late LoRA方法来自论文 `docs/Parameter Efficient Fine-Tuning of Segment Anything Model for Biomedical Imaging.pdf`。

- 其他专题/需求文档（数据与可视化等）
  - `docs/rl_data_requirements.md`，`docs/prerl_data_requirements.md`，`docs/sft_data_requirements.md`，`docs/sod_finetune_data_requirements.md`
    - 主题：不同训练阶段（RL/Pre-RL/SFT/SOD）的数据字段与文件组织要求。
  - `docs/cursor_convert_dataset_for_seg_rl_train.md`
    - 主题：将现有数据转为适配本项目训练/评估的数据格式与脚本流程。
  - `docs/cursor_mask_comparison.md`，`docs/cursor_opt_pretrain_vis.md`，`docs/cursor_code_understand-1.md`，`docs/cursor_understanding_maybe_apply_chat_t.md`
    - 主题：可视化优化、掩膜对比、代码理解与沟通方式等辅助材料。

### 四、常见问题到代码/文档的快速跳转
- 想"端到端"跑通：看 `seg-rl/predict_points_and_sam.py` 的 `main` 行172；配合 `docs/cursor_sam2_segment-20250924.md` 的使用说明与参数。
- 想训练"监督热力图模型"（baseline，无PEFT）：
  - 配置/入口见 `seg-rl/heatmap/train.py`（`parse_args` 行92；`main` 行131）。
  - 模型定义与采样见 `seg-rl/heatmap/model.py`（详见上文行号）。
  - 损失函数/目标生成见 `seg-rl/heatmap/losses.py`；数据集见 `datasets.py`。
- 想做"GRPO 点策略微调"（baseline）：看 `seg-rl/heatmap/train_grpo_points.py`（`validate` 行364；`train` 行413）。
- **想用"PEFT + SAM2微调"训练点预测模型**：
  - **完整实验指南**：看 `README_PEFT_EXPERIMENT_GUIDE.md`（端到端流程、所有命令、参数说明）。
  - **命令速查表**：看 `PEFT_COMMANDS_CHEATSHEET.md`（常用命令快速参考）。
  - **技术细节**：看 `seg-rl/peft/README.md`（架构说明、模块文档）。
  - 监督训练入口：`seg-rl/peft/train_supervised_peft.py`
  - GRPO训练入口：`seg-rl/peft/train_grpo_peft.py`
  - LoRA实现：`seg-rl/peft/lora_sam2.py`（`LoRALinear`；`LoRASAM2Wrapper`）
  - 点预测网络：`seg-rl/peft/point_predictor_peft.py`（`PointPredictorFromSAMFeatures`）
- 想只做"点预测推理/调试"：看 `seg-rl/heatmap/predict_next_point_from_model.py`（`run_initial` 行209；`run_append` 行263；`main` 行399）。
- 想"用点喂给 SAM2 出掩膜"：看 `seg-rl/sam2_segment_from_points.py`（`SAMWrapper` 行125；`calculate_bounding_box` 行324；`main` 行389）。
- 想"评估掩膜质量"：看 `seg-rl/evaluation/eval_sam_masks.py`（`_compute_metrics` 行161；`main` 行206）。
- 想"调试可视化（点更清晰）"：看 `seg-rl/visualization/viz_heuristic_sam_points.py`（`_draw_caret` 行94；`_draw_cross` 行108）。
- 想理解"层级/联合策略"与数学推导：优先看 `docs/cursor_heatmap_model.md` 对应小节（行962/2274/3032/10077；3137/10182 等），再对照 `model.py` 相应函数实现。

### 五、数据与产物路径要点
- 数据集与 JSONL：参考 `docs/*_data_requirements.md` 与 `seg-rl/annotator/gen_point_jsonl_from_masks.py`。
- 典型输出目录：`outputs/heuristic_points*`、`outputs/sam_everything/*`、`outputs/seg_r1_md/*`、`output/braintumour/*` 等。
- 预训练权重示例：`pretrained/points_predictor-251001-160epochs.pt`。

### 六、建议的阅读顺序

**对于baseline（无PEFT）**:
1) 本索引（本文）→ 2) `cursor_heatmap_model.md` 的"核心思路/层级与联合策略"→ 3) `seg-rl/heatmap/model.py` 实现 → 4) `train.py` 与 `train_grpo_points.py` 训练入口 → 5) `sam2_segment_from_points.py` → 6) 评估与可视化脚本。

**对于PEFT（SAM2微调）**:
1) 本索引（本文）→ 2) `README_PEFT_EXPERIMENT_GUIDE.md`（完整实验流程）→ 3) `seg-rl/peft/README.md`（技术细节）→ 4) `seg-rl/peft/lora_sam2.py`（LoRA实现）→ 5) `seg-rl/peft/point_predictor_peft.py`（新点预测网络）→ 6) `train_supervised_peft.py` 与 `train_grpo_peft.py`（训练入口）。

---
维护说明：新增文档或改动核心函数后，请同步更新本文的行号锚点与条目简述。


