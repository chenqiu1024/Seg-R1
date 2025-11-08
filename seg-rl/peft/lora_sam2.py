#!/usr/bin/env python3
"""
LoRA集成模块 - 为SAM2注入Late LoRA并提供特征提取接口

该模块实现了Late LoRA策略：只在SAM2 image encoder的最后一个Transformer块中注入LoRA适配器。
LoRA (Low-Rank Adaptation) 通过低秩矩阵分解实现参数高效微调。

核心功能:
- LoRALinear: LoRA层的基础实现
- LoRASAM2Wrapper: 封装SAM2模型，注入LoRA，提供特征提取和掩模预测接口

调用示例:
    # 初始化带LoRA的SAM2
    sam2_lora = LoRASAM2Wrapper(
        sam_checkpoint="third_party/sam2/checkpoints/sam2.1_hiera_large.pt",
        lora_rank=16,
        lora_alpha=32,
        device="cuda"
    )
    
    # 提取图像特征（用于点预测网络）
    features = sam2_lora.get_image_features(image_tensor, feature_scale=8)
    
    # 预测分割掩模（用于环境反馈）
    mask = sam2_lora.predict_mask(image_tensor, points, labels)
    
    # 保存/加载LoRA权重
    sam2_lora.save_lora_checkpoint("outputs/lora_weights.pt")
    sam2_lora.load_lora_checkpoint("outputs/lora_weights.pt")

参考论文:
    Late LoRA方法参考: Parameter Efficient Fine-Tuning of Segment Anything Model
    for Biomedical Imaging
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage

# 添加sam2路径
_SAM2_PATH = Path(__file__).parent.parent.parent / "third_party" / "sam2"
if str(_SAM2_PATH) not in sys.path:
    sys.path.insert(0, str(_SAM2_PATH))

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"Error importing SAM2: {e}")
    print("Please ensure SAM2 is properly installed in third_party/sam2")
    raise


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 线性层
    
    实现公式: output = frozen_linear(x) + (alpha/rank) * lora_B @ lora_A @ x
    
    Args:
        base_linear: 被冻结的基础线性层
        rank: LoRA的秩（低秩矩阵的维度）
        alpha: 缩放因子，控制LoRA对输出的贡献
        
    属性:
        base_linear: 冻结的原始权重（不可训练）
        lora_A: 下投影矩阵 [rank, in_features]
        lora_B: 上投影矩阵 [out_features, rank]
        scaling: alpha / rank
    """
    
    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base_linear = base_linear
        self.base_linear.requires_grad_(False)  # 冻结基础权重
        
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        
        # 获取base_linear的设备，确保LoRA参数在同一设备上
        device = base_linear.weight.device
        
        # LoRA矩阵初始化：A用小随机值，B用零（保证初始时LoRA不影响输出）
        self.lora_A = nn.Parameter(torch.randn(rank, in_features, device=device) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device))
        self.scaling = alpha / rank
        
        self.rank = rank
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 基础输出 + LoRA增量
        
        Args:
            x: 输入张量 [..., in_features]
            
        Returns:
            输出张量 [..., out_features]
        """
        # 基础线性层输出（冻结，无梯度）
        base_out = self.base_linear(x)
        
        # LoRA路径: x @ A^T @ B^T，带缩放
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        
        return base_out + self.scaling * lora_out
    
    def extra_repr(self) -> str:
        return f'in_features={self.base_linear.in_features}, out_features={self.base_linear.out_features}, rank={self.rank}, alpha={self.alpha}'


class LoRASAM2Wrapper:
    """
    SAM2 + Late LoRA 封装器
    
    实现Late LoRA策略：只在SAM2 image encoder的最后一个Transformer块注入LoRA。
    根据SAM2.1 Hiera Large配置，Hiera有4个stage [2, 6, 36, 4]，我们在最后一个
    stage的最后一个block中的attention层注入LoRA。
    
    主要功能:
    1. 加载SAM2模型并注入LoRA到指定位置
    2. 提取多尺度图像特征（用于点预测网络）
    3. 生成分割掩模（用于环境反馈和训练）
    4. 管理LoRA权重的保存和加载
    
    Args:
        sam_checkpoint: SAM2模型checkpoint路径
        lora_rank: LoRA秩，默认16
        lora_alpha: LoRA缩放因子，默认32
        device: 运行设备 ("cuda", "cpu", "mps")
        config_file: SAM2配置文件路径
    """
    
    def __init__(
        self,
        sam_checkpoint: str,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        device: str = "cuda",
        config_file: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    ):
        self.device = torch.device(device)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # 加载SAM2模型
        print(f"Loading SAM2 from {sam_checkpoint}")
        sam_model = build_sam2(config_file, sam_checkpoint, device=self.device)
        
        # 初始化预测器
        self.predictor = SAM2ImagePredictor(sam_model)
        self.model = sam_model
        
        # 注入LoRA到最后一个Transformer块
        self._inject_lora()
        
        print(f"LoRA injected: rank={lora_rank}, alpha={lora_alpha}")
        print(f"Trainable LoRA parameters: {sum(p.numel() for p in self.get_lora_parameters()):,}")
        
    def _inject_lora(self):
        """
        注入LoRA到SAM2 image encoder的最后一个Transformer块
        
        对于Hiera模型，结构为:
        - trunk (Hiera backbone)
          - blocks: nn.ModuleList[MultiScaleBlock]  <- 所有块的列表
          - stage_ends: List[int]  <- 每个阶段结束的块索引
            - blocks[i].attn: MultiScaleAttention
              - qkv: Linear  <- 注入LoRA的目标
        """
        # 获取image encoder的trunk (Hiera)
        image_encoder = self.model.image_encoder
        trunk = image_encoder.trunk
        
        # 访问blocks列表（Hiera使用blocks而不是stages）
        if not hasattr(trunk, 'blocks') or len(trunk.blocks) == 0:
            raise RuntimeError("Cannot find blocks in Hiera trunk")
        
        # 获取最后一个block（最后一个阶段的最后一个块）
        last_block = trunk.blocks[-1]
        
        # 找到attention层的qkv投影
        if not hasattr(last_block, 'attn'):
            raise RuntimeError("Cannot find attention in last block")
        
        attn = last_block.attn
        
        # 注入LoRA到qkv层
        # 注意: 不同SAM2版本可能有不同的注意力实现，需要适配
        if hasattr(attn, 'qkv') and isinstance(attn.qkv, nn.Linear):
            print(f"Injecting LoRA into last block's attn.qkv")
            original_qkv = attn.qkv
            attn.qkv = LoRALinear(original_qkv, self.lora_rank, self.lora_alpha)
        elif hasattr(attn, 'q_proj') and hasattr(attn, 'k_proj') and hasattr(attn, 'v_proj'):
            # 如果qkv是分开的，分别注入
            print(f"Injecting LoRA into last block's attn.q_proj, k_proj, v_proj")
            if isinstance(attn.q_proj, nn.Linear):
                original_q = attn.q_proj
                attn.q_proj = LoRALinear(original_q, self.lora_rank, self.lora_alpha)
            if isinstance(attn.k_proj, nn.Linear):
                original_k = attn.k_proj
                attn.k_proj = LoRALinear(original_k, self.lora_rank, self.lora_alpha)
            if isinstance(attn.v_proj, nn.Linear):
                original_v = attn.v_proj
                attn.v_proj = LoRALinear(original_v, self.lora_rank, self.lora_alpha)
        else:
            raise RuntimeError("Cannot find qkv or q/k/v projection layers in attention")
        
        # 可选: 也可以注入到MLP层
        # if hasattr(last_block, 'mlp'):
        #     ...
    
    def get_lora_parameters(self) -> Iterator[nn.Parameter]:
        """
        返回所有LoRA参数的迭代器（用于优化器）
        
        Returns:
            LoRA参数迭代器
        """
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                yield module.lora_A
                yield module.lora_B
    
    def freeze_sam_base(self):
        """
        冻结SAM2的所有基础参数，只保留LoRA可训练
        
        注意: LoRA注入时已经冻结了被wrap的Linear层，此方法确保其他层也被冻结
        """
        for name, param in self.model.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
        
        # 确保LoRA参数可训练
        for param in self.get_lora_parameters():
            param.requires_grad = True
    
    def get_image_features(
        self,
        image: torch.Tensor,
        feature_scale: int = 8
    ) -> torch.Tensor:
        """
        提取SAM2图像特征（用于点预测网络）
        
        SAM2的FPN输出多个尺度的特征:
        - Level 0: stride 4 (高分辨率)
        - Level 1: stride 8 (中等，推荐)
        - Level 2: stride 16 (低分辨率)
        
        Args:
            image: 输入图像 [B, 3, H, W]，已归一化
            feature_scale: 特征尺度，8表示H/8 x W/8（推荐）
            
        Returns:
            特征张量 [B, 256, H/scale, W/scale]
        """
        # 通过image encoder提取特征
        backbone_out = self.model.forward_image(image)
        
        # SAM2内部方法: 准备backbone特征
        # 返回: (backbone_out, vision_feats, vision_pos_embeds, feat_sizes)
        # vision_feats格式: [HW, B, C] (已展平)
        _, vision_feats, _, feat_sizes = self.model._prepare_backbone_features(backbone_out)
        
        # vision_feats是一个list，包含多个尺度的特征
        # 根据feature_scale选择对应的level
        if feature_scale == 4:
            feat_idx = 0
        elif feature_scale == 8:
            feat_idx = 1 if len(vision_feats) > 1 else 0
        elif feature_scale == 16:
            feat_idx = 2 if len(vision_feats) > 2 else len(vision_feats) - 1
        else:
            raise ValueError(f"Unsupported feature_scale: {feature_scale}. Use 4, 8, or 16.")
        
        # 获取选中的特征和尺寸
        feature = vision_feats[feat_idx]  # [HW, B, C]
        feat_size = feat_sizes[feat_idx]  # (H, W)
        batch_size = image.shape[0]
        
        # 将特征从 [HW, B, C] 转换为 [B, C, H, W]
        # 参考 sam2_image_predictor.py 中的转换方式
        feature = feature.permute(1, 2, 0)  # [B, C, HW]
        feature = feature.view(batch_size, -1, *feat_size)  # [B, C, H, W]
        
        return feature
    
    def predict_mask(
        self,
        image: Union[torch.Tensor, np.ndarray, PILImage.Image],
        points: List[Tuple[float, float]],
        labels: List[int],
        return_logits: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        使用SAM2生成分割掩模（完整前向传播）
        
        Args:
            image: 输入图像，可以是:
                - torch.Tensor [B, 3, H, W] 或 [3, H, W]
                - np.ndarray [H, W, 3]
                - PIL.Image
            points: 点提示列表 [(x, y), ...]
            labels: 点标签列表 [1, 0, ...]，1=前景，0=背景
            return_logits: 是否返回logits（用于某些loss计算）
            
        Returns:
            如果return_logits=False: 二值掩模 [H, W] 或 [B, H, W]
            如果return_logits=True: (掩模, logits)
        """
        # 转换输入为numpy格式（SAM2 predictor需要）
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]  # 取第一张图
            image = image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            image = (image * 255).astype(np.uint8)
        elif isinstance(image, PILImage.Image):
            image = np.array(image)
        
        # 设置图像
        self.predictor.set_image(image)
        
        # 转换点和标签为numpy数组
        input_points = np.array(points, dtype=np.float32) if points else None
        input_labels = np.array(labels, dtype=np.int32) if labels else None
        
        # 预测
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False,
        )
        
        # 转换为torch tensor
        mask = torch.from_numpy(masks[0]).float().to(self.device)
        
        if return_logits:
            logits_tensor = torch.from_numpy(logits[0]).float().to(self.device)
            return mask, logits_tensor
        return mask
    
    def save_lora_checkpoint(self, path: str):
        """
        只保存LoRA权重（不包含SAM基础权重）
        
        Args:
            path: 保存路径
        """
        lora_state = {}
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear):
                lora_state[name + '.lora_A'] = module.lora_A.data.cpu()
                lora_state[name + '.lora_B'] = module.lora_B.data.cpu()
        
        checkpoint = {
            'lora_state_dict': lora_state,
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        print(f"LoRA checkpoint saved to {path}")
    
    def load_lora_checkpoint(self, path: str):
        """
        加载LoRA权重
        
        Args:
            path: checkpoint路径
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        lora_state = checkpoint['lora_state_dict']
        
        # 验证配置一致性
        if checkpoint['lora_rank'] != self.lora_rank:
            print(f"Warning: LoRA rank mismatch: checkpoint={checkpoint['lora_rank']}, current={self.lora_rank}")
        if checkpoint['lora_alpha'] != self.lora_alpha:
            print(f"Warning: LoRA alpha mismatch: checkpoint={checkpoint['lora_alpha']}, current={self.lora_alpha}")
        
        # 加载LoRA参数
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear):
                lora_A_key = name + '.lora_A'
                lora_B_key = name + '.lora_B'
                if lora_A_key in lora_state:
                    module.lora_A.data = lora_state[lora_A_key].to(self.device)
                if lora_B_key in lora_state:
                    module.lora_B.data = lora_state[lora_B_key].to(self.device)
        
        print(f"LoRA checkpoint loaded from {path}")
    
    def get_lora_state_dict(self) -> dict:
        """
        获取LoRA状态字典（用于完整checkpoint）
        
        Returns:
            LoRA状态字典
        """
        lora_state = {}
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear):
                lora_state[name + '.lora_A'] = module.lora_A.data
                lora_state[name + '.lora_B'] = module.lora_B.data
        return lora_state
    
    def load_lora_state_dict(self, lora_state: dict):
        """
        加载LoRA状态字典
        
        Args:
            lora_state: LoRA状态字典
        """
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear):
                lora_A_key = name + '.lora_A'
                lora_B_key = name + '.lora_B'
                if lora_A_key in lora_state:
                    module.lora_A.data = lora_state[lora_A_key].to(self.device)
                if lora_B_key in lora_state:
                    module.lora_B.data = lora_state[lora_B_key].to(self.device)


# 便捷函数
def load_sam2_with_lora(
    sam_checkpoint: str,
    lora_checkpoint: Optional[str] = None,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    device: str = "cuda"
) -> LoRASAM2Wrapper:
    """
    加载带LoRA的SAM2模型（便捷函数）
    
    Args:
        sam_checkpoint: SAM2基础模型路径
        lora_checkpoint: LoRA权重路径（可选）
        lora_rank: LoRA秩
        lora_alpha: LoRA缩放因子
        device: 设备
        
    Returns:
        LoRASAM2Wrapper实例
    """
    wrapper = LoRASAM2Wrapper(
        sam_checkpoint=sam_checkpoint,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        device=device
    )
    
    if lora_checkpoint is not None and os.path.exists(lora_checkpoint):
        wrapper.load_lora_checkpoint(lora_checkpoint)
    
    return wrapper

