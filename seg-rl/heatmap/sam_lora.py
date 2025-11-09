"""
SAM Late LoRA Implementation

基于论文 "Parameter Efficient Fine-Tuning of Segment Anything Model for Biomedical Imaging"
实现 Late LoRA 方法，在 SAM 的 image encoder 的最后一个 Transformer 块中添加 LoRA 层。

LoRA (Low-Rank Adaptation) 通过低秩分解减少可训练参数数量：
    W = W_0 + BA
其中 W_0 是冻结的预训练权重，B 和 A 是可训练的低秩矩阵。

Reference: https://arxiv.org/abs/2502.00418
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """LoRA 层实现
    
    在原始线性层的基础上添加低秩适配器：
    output = W_0 @ x + (B @ A) @ x * scale
    
    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        rank: LoRA 秩（越小参数越少，通常为 4-16）
        alpha: LoRA 缩放因子（通常设为 rank 的 1-2 倍）
        dropout: Dropout 概率
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA 矩阵：A 使用 Kaiming 初始化，B 初始化为 0
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A.weight, a=torch.math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：计算 LoRA 增量
        
        Args:
            x: 输入张量 [..., in_features]
            
        Returns:
            LoRA 输出 [..., out_features]
        """
        x = self.dropout(x)
        result = self.lora_B(self.lora_A(x))
        return result * self.scaling


class LoRALinear(nn.Module):
    """带 LoRA 适配器的线性层
    
    组合原始线性层和 LoRA 层：
    output = linear(x) + lora(x)
    
    Args:
        linear: 原始的 nn.Linear 层（将被冻结）
        rank: LoRA 秩
        alpha: LoRA 缩放因子
        dropout: Dropout 概率
    """
    
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        # 冻结原始层
        for param in self.linear.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            输出 = 原始层输出 + LoRA 输出
        """
        return self.linear(x) + self.lora(x)
    
    def merge_weights(self) -> nn.Linear:
        """将 LoRA 权重合并到原始层中
        
        用于推理时加速，将 W = W_0 + BA 合并为单个矩阵
        
        Returns:
            合并后的 nn.Linear 层
        """
        merged = nn.Linear(
            self.linear.in_features,
            self.linear.out_features,
            bias=self.linear.bias is not None
        )
        
        # 计算合并后的权重
        with torch.no_grad():
            lora_weight = self.lora.lora_B.weight @ self.lora.lora_A.weight
            lora_weight = lora_weight * self.lora.scaling
            merged.weight.copy_(self.linear.weight + lora_weight)
            if self.linear.bias is not None and merged.bias is not None:
                merged.bias.copy_(self.linear.bias)
                
        return merged


def apply_lora_to_attention(
    attn_module: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> Dict[str, nn.Module]:
    """为注意力模块的指定层添加 LoRA
    
    Args:
        attn_module: 注意力模块（如 nn.MultiheadAttention 或自定义 Attention）
        rank: LoRA 秩
        alpha: LoRA 缩放因子
        dropout: Dropout 概率
        target_modules: 要添加 LoRA 的模块名称列表，如 ["q_proj", "v_proj"]
                       如果为 None，则添加到 ["q_proj", "k_proj", "v_proj", "out_proj"]
    
    Returns:
        替换的 LoRA 模块字典 {name: LoRALinear}
    """
    if target_modules is None:
        # 默认为 Q, K, V, O 投影层
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
    
    replaced_modules = {}
    
    for name, module in attn_module.named_modules():
        # 检查是否是目标模块
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                # 获取父模块和属性名
                *parents, attr_name = name.split(".")
                parent = attn_module
                for p in parents:
                    parent = getattr(parent, p)
                
                # 替换为 LoRALinear
                lora_linear = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, attr_name, lora_linear)
                replaced_modules[name] = lora_linear
                print(f"[LoRA] Applied to {name}: in={module.in_features}, out={module.out_features}, rank={rank}")
                break
    
    return replaced_modules


def apply_late_lora_to_sam_encoder(
    sam_model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> Dict[str, nn.Module]:
    """为 SAM image encoder 的最后一个 Transformer 块添加 Late LoRA
    
    根据论文建议，只在最后一个 Transformer 块中添加 LoRA，
    这样既保持了效率，又能有效地进行任务适配。
    
    Args:
        sam_model: SAM2 模型实例
        rank: LoRA 秩
        alpha: LoRA 缩放因子
        dropout: Dropout 概率
        target_modules: 要添加 LoRA 的模块名称列表
    
    Returns:
        应用了 LoRA 的模块字典
    """
    if target_modules is None:
        # 默认为 Q 和 V 投影（论文中推荐的配置）
        target_modules = ["q_proj", "v_proj"]
    
    # 冻结整个 SAM 模型
    for param in sam_model.parameters():
        param.requires_grad = False
    
    replaced_modules = {}
    
    # 获取 image encoder
    image_encoder = sam_model.image_encoder
    
    # SAM2 使用 Hiera 架构，blocks 在 image_encoder.trunk.stages 中
    # 最后一个 stage 包含最后的 transformer blocks
    if hasattr(image_encoder, "trunk") and hasattr(image_encoder.trunk, "stages"):
        stages = image_encoder.trunk.stages
        if len(stages) > 0:
            last_stage = stages[-1]
            
            # 在最后一个 stage 的所有 blocks 中添加 LoRA
            # 或者只在最后一个 block 中添加（取决于论文的具体实现）
            if hasattr(last_stage, "blocks") and len(last_stage.blocks) > 0:
                # 只在最后一个 block 中添加 LoRA（Late LoRA）
                last_block = last_stage.blocks[-1]
                
                # 查找注意力模块
                for name, module in last_block.named_modules():
                    # SAM2 Hiera 的注意力模块通常叫 attn
                    if "attn" in name and hasattr(module, "qkv"):
                        # Hiera 使用融合的 QKV 投影
                        print(f"[Late LoRA] Found attention module: {name}")
                        
                        # 为 QKV 和 proj 添加 LoRA
                        for subname, submodule in module.named_children():
                            if isinstance(submodule, nn.Linear) and subname in ["qkv", "proj"]:
                                lora_linear = LoRALinear(
                                    submodule, 
                                    rank=rank, 
                                    alpha=alpha, 
                                    dropout=dropout
                                )
                                setattr(module, subname, lora_linear)
                                full_name = f"image_encoder.trunk.stages[-1].blocks[-1].{name}.{subname}"
                                replaced_modules[full_name] = lora_linear
                                print(f"[Late LoRA] Applied to {full_name}: "
                                      f"in={submodule.in_features}, out={submodule.out_features}, rank={rank}")
            else:
                print("[Late LoRA] Warning: No blocks found in last stage")
    else:
        print("[Late LoRA] Warning: SAM model structure not recognized, trying alternative path")
        
        # 备用方案：直接查找 encoder 中的 blocks
        if hasattr(image_encoder, "blocks"):
            blocks = image_encoder.blocks
            if len(blocks) > 0:
                last_block = blocks[-1]
                for name, module in last_block.named_modules():
                    if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
                        lora_linear = LoRALinear(
                            module.qkv,
                            rank=rank,
                            alpha=alpha,
                            dropout=dropout
                        )
                        module.qkv = lora_linear
                        replaced_modules[f"image_encoder.blocks[-1].{name}.qkv"] = lora_linear
                        print(f"[Late LoRA] Applied to blocks[-1].{name}.qkv")
    
    if len(replaced_modules) == 0:
        print("[Late LoRA] Warning: No modules were replaced. SAM model may have unexpected structure.")
        print("[Late LoRA] Available attributes:", dir(image_encoder))
        if hasattr(image_encoder, "trunk"):
            print("[Late LoRA] Trunk attributes:", dir(image_encoder.trunk))
    
    return replaced_modules


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """获取模型中所有 LoRA 层的参数
    
    Args:
        model: 包含 LoRA 层的模型
        
    Returns:
        LoRA 参数列表
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, (LoRALayer, LoRALinear)):
            lora_params.extend(module.lora.parameters())
    return lora_params


def count_lora_parameters(model: nn.Module) -> Dict[str, int]:
    """统计 LoRA 参数数量
    
    Args:
        model: 包含 LoRA 层的模型
        
    Returns:
        参数统计字典
    """
    lora_params = 0
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    for module in model.modules():
        if isinstance(module, (LoRALayer, LoRALinear)):
            for param in module.lora.parameters():
                lora_params += param.numel()
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "lora": lora_params,
        "trainable_ratio": trainable_params / max(total_params, 1) * 100,
    }


def print_lora_info(model: nn.Module) -> None:
    """打印 LoRA 配置信息
    
    Args:
        model: 包含 LoRA 层的模型
    """
    stats = count_lora_parameters(model)
    print("\n" + "="*60)
    print("LoRA Configuration Summary")
    print("="*60)
    print(f"Total parameters:     {stats['total']:>12,}")
    print(f"Trainable parameters: {stats['trainable']:>12,}")
    print(f"LoRA parameters:      {stats['lora']:>12,}")
    print(f"Trainable ratio:      {stats['trainable_ratio']:>11.2f}%")
    print("="*60 + "\n")

