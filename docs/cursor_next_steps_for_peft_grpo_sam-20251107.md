# Next steps for SAM model modification
_Exported on 11/7/2025 at 18:19:01 PST from Cursor (2.0.64)_

---

**User**

阅读@docs  下的所有*.md和@Seg_RL-design.pdf ，以及seg-rl下的所有代码以了解上下文。尤其是docs/Seg_RL-design.pdf和@knowledge_map_index.md 包含了简要的总结和索引，可以用于快速了解和查询。接下来，我需要你基于我们现有工作的基础，进行下一步的修改。



具体地，之前的网络模型和训练方案，在最终强化学习时没有获得较理想的性能。因此我计划引入PEFT（Parameter Efficient Fine-Tuning）来同时微调SAM本身，而不仅仅把SAM作为黑盒使用，请参考这篇论文@Parameter Efficient Fine-Tuning of Segment Anything Model for Biomedical Imaging.pdf 。我初步设想的改造方案是：一方面，对于SAM的微调，借用该论文中提出的Late LoRA方法，只在SAM的image encoder中最后一个Transformer块上加入LoRA；另一方面，原有的“下一提示点预测模型”的输入是<待分割图像+真值掩模>，但由于我们要让SAM的图像编码器的一部分也参与微调，为了使该被微调部分能够直接与“下一提示点预测模型”的推理产生关联，需要让该预测模型转而接受<SAM图像编码器的输出+真值掩模>作为输入。在做这样的改变后，预测模型的网络结构应该如何设计，也需要你深入研究决定。



除了上述区别以外，模型训练的框架设计与之前基本相似，都包含一个通过启发式方法自动生成数据集的有监督预训练过程，和一个基于GRPO的强化学习训练过程。

为了便于研究和调试，我希望你能尽量遵循之前工程实现里一些值得借鉴的模式，例如：用多个脚本实现整个训练流程中不同的环节，以写入和读取文件的方式存储和传递中间结果，以利于模块之间的解耦合和调试，用专门的脚本读取中间结果文件生成可视化图示。当然你也可以使用你认为更好的模式来实现。无论采用什么方法，最终想要达成的效果就是可以使研究人员方便地评估效果和调试问题。还有几点要求：1. 要考虑到断点保存/加载，以使得实验可以方便地暂停和接续；2. 要有足够清晰的代码注释，除非代码本身已经很明显到足以自解释；3. 每个脚本应该有关于如何调用的范例，放在脚本开头的说明部分

请认真理解全部上下文和任务要求。自行研究和提出你认为最合理的设计方案，向我讲述一遍。接下来如果我表示认可，你就可以开始着手实现你的代码。

---

**Cursor**

I'll scan the repo to understand the current SAM2 integration, heatmap predictor, and RL trainer structure, then draft a concrete PEFT-Late LoRA integration plan and the new predictor design aligned with your patterns.

1. LoRA target SAM variant?
   - a) SAM2.1 Hiera-L (default)
   - b) SAM2.1 Hiera-B+
   - c) SAM2 Hiera-L
   - d) Plain SAM (ViT)

2. LoRA implementation?
   - a) Custom lightweight LoRA (default)
   - b) HuggingFace PEFT (best-effort wrap)
   - c) Other (specify)

Once you pick, I’ll finalize a concrete plan and start implementing after your approval.

— 

Planned high-level design (preview)

- PEFT-Late LoRA on SAM image encoder
  - Inject LoRA only into the last transformer block of the image encoder trunk (attn q,k,v and MLP projections).
  - Expose save/load of LoRA state separately; freeze base weights by default; only LoRA params require_grad.

- Predictor redesign: consume SAM encoder features + GT mask
  - Precompute SAM feature map at stride 16 with 256 channels (FPN fused or select level).
  - New model `PointHeatmapFromSamFeatModel`: input [B, 257, H/16, W/16] (concat downsampled mask); small UNet/FPN head to heatmap logits (+ label head), reuse existing hierarchical policy utilities.
  - Support soft heatmap targets and existing KL/MSE/CE losses.

- Training flows (match prior patterns, file-based intermediates)
  - Supervised pretrain
    - Mode A (fast): freeze SAM, cache features to disk, train predictor.
    - Mode B (joint-PEFT): enable LoRA, compute features on-the-fly (or periodic recache), train predictor+LoRA with small LR on LoRA.
    - Scripts write checkpoints, JSONL of sample metadata, and visualizations.
  - RL with GRPO
    - Precompute per-image SAM features each episode (reused across steps) while LoRA is updating.
    - Train predictor + LoRA jointly; SAM2 uses current LoRA to produce masks for reward.
    - Maintain resumeable checkpoints and periodic qualitative viz.

- Modular scripts (each with header usage examples)
  - seg-rl/sam_peft/late_lora.py: LoRA injection, save/load helpers.
  - seg-rl/sam_peft/extract_features.py: cache SAM features to .pt/.npz with index JSONL.
  - seg-rl/heatmap_peft/model.py: new predictor.
  - seg-rl/heatmap_peft/datasets.py: read feature files + masks + targets.
  - seg-rl/heatmap_peft/train.py: supervised training (freeze/joint).
  - seg-rl/heatmap_peft/train_grpo_points.py: RL training using SAM+LoRA.
  - seg-rl/sam2_segment_from_points.py: add flags to load LoRA and to output features optionally.
  - seg-rl/visualization/viz_heatmap_from_feats.py: overlays and debugging panels.

- Checkpointing and resumes
  - Separate artifacts: predictor.pt, lora.safetensors, optimizer+scheduler states, rng seeds, args.json.
  - Resume restores both and SAM feature cache metadata.

- Evaluation and debugging
  - Offline eval scripts to compute IoU/Dice/SM; step-wise rollout viz; ablations (freeze vs LoRA; which FPN level).

If this aligns, I’ll produce a concise implementation plan with exact file paths and staged todos for you to approve before coding.

