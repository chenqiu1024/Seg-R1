"""
SAM Conv-LoRA Implementation

基于论文 "Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model"
实现 Conv-LoRA 方法，在 LoRA 的基础上引入卷积操作来更好地保持空间结构信息。

Conv-LoRA 的核心思想：
    W = W_0 + Conv(B @ A) * scale
    
相比标准 LoRA，Conv-LoRA 在低秩分解后添加卷积层，使得：
1. 能够捕获空间局部相关性
2. 更适合视觉任务
3. 参数效率仍然很高

与 Late LoRA 的区别：
- Late LoRA: 只在最后一个 Transformer 块
- Conv-LoRA: 可以在多个位置，使用卷积增强

注意：Conv-LoRA 与 Late LoRA 互斥，只能选择其中一种。
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLoRALayer(nn.Module):
    """Conv-LoRA 层实现
    
    在 LoRA 的基础上添加卷积操作：
    output = W_0 @ x + Conv(B @ A @ x) * scale
    
    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        rank: LoRA 秩（越小参数越少）
        alpha: LoRA 缩放因子
        kernel_size: 卷积核大小（1 表示退化为标准 LoRA）
        dropout: Dropout 概率
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.kernel_size = kernel_size
        self.scaling = alpha / rank
        
        # LoRA 矩阵：A 和 B
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # 卷积层（如果 kernel_size > 1）
        if kernel_size > 1:
            padding = kernel_size // 2
            self.conv = nn.Conv2d(
                out_features, 
                out_features, 
                kernel_size=kernel_size, 
                padding=padding, 
                groups=out_features,  # Depthwise convolution
                bias=False
            )
            # 初始化为单位卷积（接近恒等变换）
            nn.init.zeros_(self.conv.weight)
            # 中心位置设为 1
            center = kernel_size // 2
            for i in range(out_features):
                self.conv.weight.data[i, 0, center, center] = 1.0
        else:
            self.conv = nn.Identity()
        
        # 初始化 LoRA 矩阵
        nn.init.kaiming_uniform_(self.lora_A.weight, a=torch.math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor, spatial_shape: Optional[tuple] = None) -> torch.Tensor:
        """前向传播：计算 Conv-LoRA 增量
        
        Args:
            x: 输入张量 [..., in_features] 或 [B, L, in_features]
            spatial_shape: 空间形状 (H, W)，用于 reshape 到 2D 进行卷积
            
        Returns:
            Conv-LoRA 输出 [..., out_features]
        """
        orig_shape = x.shape
        x = self.dropout(x)
        
        # LoRA 低秩分解
        result = self.lora_B(self.lora_A(x))  # [..., out_features]
        
        # 如果有卷积且提供了空间形状
        if self.kernel_size > 1 and spatial_shape is not None:
            H, W = spatial_shape
            # Reshape 到 2D: [B, L, C] -> [B, H, W, C] -> [B, C, H, W]
            if len(result.shape) == 3:  # [B, L, C]
                B, L, C = result.shape
                assert L == H * W, f"Spatial size mismatch: L={L}, H*W={H*W}"
                result = result.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
                result = self.conv(result)  # [B, C, H, W]
                result = result.permute(0, 2, 3, 1).reshape(B, L, C)  # [B, L, C]
            elif len(result.shape) == 4:  # [B, C, H, W]
                result = self.conv(result)
        
        return result * self.scaling


class ConvLoRALinear(nn.Module):
    """带 Conv-LoRA 适配器的线性层
    
    组合原始线性层和 Conv-LoRA 层：
    output = linear(x) + conv_lora(x, spatial_shape)
    
    Args:
        linear: 原始的 nn.Linear 层（将被冻结）
        rank: LoRA 秩
        alpha: LoRA 缩放因子
        kernel_size: 卷积核大小
        dropout: Dropout 概率
    """
    
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = linear
        self.lora = ConvLoRALayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        
        # 冻结原始层
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor, spatial_shape: Optional[tuple] = None) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量
            spatial_shape: 空间形状 (H, W)，用于卷积操作
            
        Returns:
            输出 = 原始层输出 + Conv-LoRA 输出
        """
        return self.linear(x) + self.lora(x, spatial_shape)


def apply_conv_lora_to_sam_encoder(
    sam_model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    kernel_size: int = 3,
    dropout: float = 0.0,
    target_blocks: Optional[List[int]] = None,
    target_modules: Optional[List[str]] = None,
) -> Dict[str, nn.Module]:
    """为 SAM image encoder 应用 Conv-LoRA
    
    Conv-LoRA 可以应用到多个 Transformer 块中，不仅限于最后一个块。
    
    Args:
        sam_model: SAM2 模型实例
        rank: LoRA 秩
        alpha: LoRA 缩放因子
        kernel_size: 卷积核大小（3 推荐，1 退化为标准 LoRA）
        dropout: Dropout 概率
        target_blocks: 要应用 LoRA 的 block 索引列表，None 表示只用最后一个
        target_modules: 要添加 LoRA 的模块名称列表
    
    Returns:
        应用了 Conv-LoRA 的模块字典
    """
    if target_modules is None:
        # 默认为 Q 和 V 投影
        target_modules = ["qkv", "proj"]
    
    # 冻结整个 SAM 模型
    for param in sam_model.parameters():
        param.requires_grad = False
    
    replaced_modules = {}
    
    # 获取 image encoder
    image_encoder = sam_model.image_encoder
    
    # SAM2 Hiera 架构：image_encoder.trunk.blocks
    if hasattr(image_encoder, "trunk") and hasattr(image_encoder.trunk, "blocks"):
        blocks = image_encoder.trunk.blocks
        total_blocks = len(blocks)
        
        if total_blocks == 0:
            print("[Conv-LoRA] Warning: No blocks found in trunk")
            return replaced_modules
        
        # 确定要应用 LoRA 的 block 索引
        if target_blocks is None:
            # 默认：只用最后一个 block
            target_block_indices = [total_blocks - 1]
        else:
            # 用户指定的 blocks
            target_block_indices = [idx if idx >= 0 else total_blocks + idx 
                                   for idx in target_blocks]
            # 过滤无效索引
            target_block_indices = [idx for idx in target_block_indices 
                                   if 0 <= idx < total_blocks]
        
        print(f"[Conv-LoRA] Total blocks: {total_blocks}, applying to blocks: {target_block_indices}")
        print(f"[Conv-LoRA] Conv kernel size: {kernel_size}")
        
        # 在指定的 blocks 中应用 Conv-LoRA
        for block_idx in target_block_indices:
            block = blocks[block_idx]
            
            # 查找注意力模块
            lora_applied = False
            for name, module in block.named_modules():
                if name == "attn" or name.endswith(".attn"):
                    print(f"[Conv-LoRA] Found attention module in block {block_idx}: {name}")
                    
                    # 为指定模块添加 Conv-LoRA
                    for subname in target_modules:
                        if hasattr(module, subname):
                            submodule = getattr(module, subname)
                            if isinstance(submodule, nn.Linear):
                                # 获取原始模块的设备
                                device = next(submodule.parameters()).device
                                
                                conv_lora_linear = ConvLoRALinear(
                                    submodule, 
                                    rank=rank, 
                                    alpha=alpha,
                                    kernel_size=kernel_size,
                                    dropout=dropout
                                )
                                
                                # 将 Conv-LoRA 层移动到相同设备
                                conv_lora_linear = conv_lora_linear.to(device)
                                
                                setattr(module, subname, conv_lora_linear)
                                full_name = f"image_encoder.trunk.blocks[{block_idx}].{name}.{subname}"
                                replaced_modules[full_name] = conv_lora_linear
                                print(f"[Conv-LoRA] Applied to {full_name}: "
                                      f"in={submodule.in_features}, out={submodule.out_features}, "
                                      f"rank={rank}, kernel={kernel_size}x{kernel_size}, device={device}")
                                lora_applied = True
            
            if not lora_applied:
                print(f"[Conv-LoRA] Warning: No attention modules found in block {block_idx}")
    else:
        print("[Conv-LoRA] Warning: SAM model structure not recognized")
        print(f"[Conv-LoRA] Available encoder attributes: {[attr for attr in dir(image_encoder) if not attr.startswith('_')]}")
    
    if len(replaced_modules) == 0:
        print("[Conv-LoRA] Warning: No modules were replaced.")
    
    return replaced_modules


def get_conv_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """获取模型中所有 Conv-LoRA 层的参数
    
    Args:
        model: 包含 Conv-LoRA 层的模型
        
    Returns:
        Conv-LoRA 参数列表
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, ConvLoRALinear):
            lora_params.extend(module.lora.parameters())
        elif isinstance(module, ConvLoRALayer):
            lora_params.extend(module.parameters())
    return lora_params


def count_conv_lora_parameters(model: nn.Module) -> Dict[str, int]:
    """统计 Conv-LoRA 参数数量
    
    Args:
        model: 包含 Conv-LoRA 层的模型
        
    Returns:
        参数统计字典
    """
    conv_lora_params = 0
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    for module in model.modules():
        if isinstance(module, ConvLoRALinear):
            for param in module.lora.parameters():
                conv_lora_params += param.numel()
        elif isinstance(module, ConvLoRALayer):
            for param in module.parameters():
                conv_lora_params += param.numel()
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "conv_lora": conv_lora_params,
        "trainable_ratio": trainable_params / max(total_params, 1) * 100,
    }


def print_conv_lora_info(model: nn.Module) -> None:
    """打印 Conv-LoRA 配置信息
    
    Args:
        model: 包含 Conv-LoRA 层的模型
    """
    stats = count_conv_lora_parameters(model)
    print("\n" + "="*60)
    print("Conv-LoRA Configuration Summary")
    print("="*60)
    print(f"Total parameters:     {stats['total']:>12,}")
    print(f"Trainable parameters: {stats['trainable']:>12,}")
    print(f"Conv-LoRA parameters: {stats['conv_lora']:>12,}")
    print(f"Trainable ratio:      {stats['trainable_ratio']:>11.2f}%")
    print("="*60 + "\n")

