#!/usr/bin/env python3
"""
测试 SAM Late LoRA 实现的兼容性

验证以下场景：
1. 标准模型（不使用 SAM）可以正常创建和前向传播
2. 使用 SAM encoder（冻结）的模型可以正常工作
3. 使用 SAM encoder + LoRA 的模型可以正常工作
4. Checkpoint 保存和加载的兼容性

注意：此脚本需要 SAM2 checkpoint 才能运行完整测试
"""

import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 直接导入（因为我们在 heatmap 目录下）
try:
    from model import ModelConfig, PointHeatmapModel, PointHeatmapModelWithSAM
    from utils import save_checkpoint, load_checkpoint
except ImportError:
    # 备用：通过相对路径导入
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from model import ModelConfig, PointHeatmapModel, PointHeatmapModelWithSAM
    from utils import save_checkpoint, load_checkpoint


def print_section(title: str):
    """打印章节标题"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_standard_model():
    """测试标准模型"""
    print_section("测试 1: 标准模型（不使用 SAM）")
    
    cfg = ModelConfig(
        backbone="unet_s",
        pretrained=False,
        main_in_channels=3,
        cond_in_channels=1,
    )
    
    model = PointHeatmapModel(cfg)
    print(f"✓ 模型创建成功")
    
    # 测试前向传播
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    batch_size = 2
    height, width = 240, 240
    x = torch.randn(batch_size, 3, height, width, device=device)
    cond = torch.randn(batch_size, 1, height, width, device=device)
    
    with torch.no_grad():
        logits, label_logits = model(x, cond)
    
    assert logits.shape == (batch_size, 1, height, width), f"Logits shape mismatch: {logits.shape}"
    assert label_logits.shape == (batch_size, 2), f"Label logits shape mismatch: {label_logits.shape}"
    print(f"✓ 前向传播成功: logits={logits.shape}, label_logits={label_logits.shape}")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ 参数统计: total={total_params:,}, trainable={trainable_params:,}")
    
    return model


def test_sam_frozen_model(sam_checkpoint: str):
    """测试 SAM encoder 冻结模型"""
    print_section("测试 2: SAM Encoder（冻结，不使用 LoRA）")
    
    if not os.path.exists(sam_checkpoint):
        print(f"⚠ 跳过测试：SAM checkpoint 不存在: {sam_checkpoint}")
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = ModelConfig(
        backbone="unet_s",
        pretrained=False,
        main_in_channels=3,
        cond_in_channels=1,
        use_sam_encoder=True,
        sam_checkpoint=sam_checkpoint,
        sam_lora_enabled=False,
        sam_freeze_encoder=True,
    )
    
    model = PointHeatmapModelWithSAM(cfg).to(device)
    print(f"✓ 模型创建成功")
    
    # 测试前向传播
    batch_size = 2
    height, width = 240, 240
    x = torch.randn(batch_size, 3, height, width, device=device)
    cond = torch.randn(batch_size, 1, height, width, device=device)
    
    with torch.no_grad():
        logits, label_logits = model(x, cond)
    
    assert logits.shape == (batch_size, 1, height, width), f"Logits shape mismatch: {logits.shape}"
    assert label_logits.shape == (batch_size, 2), f"Label logits shape mismatch: {label_logits.shape}"
    print(f"✓ 前向传播成功: logits={logits.shape}, label_logits={label_logits.shape}")
    
    # 验证 SAM encoder 是否被冻结
    sam_params_trainable = sum(p.numel() for p in model.sam_encoder.parameters() if p.requires_grad)
    assert sam_params_trainable == 0, f"SAM encoder should be frozen but has {sam_params_trainable} trainable params"
    print(f"✓ SAM encoder 已正确冻结")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ 参数统计: total={total_params:,}, trainable={trainable_params:,}")
    
    return model


def test_sam_lora_model(sam_checkpoint: str):
    """测试 SAM encoder + LoRA 模型"""
    print_section("测试 3: SAM Encoder + Late LoRA")
    
    if not os.path.exists(sam_checkpoint):
        print(f"⚠ 跳过测试：SAM checkpoint 不存在: {sam_checkpoint}")
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = ModelConfig(
        backbone="unet_s",
        pretrained=False,
        main_in_channels=3,
        cond_in_channels=1,
        use_sam_encoder=True,
        sam_checkpoint=sam_checkpoint,
        sam_lora_enabled=True,
        sam_lora_rank=8,
        sam_lora_alpha=16.0,
        sam_lora_dropout=0.0,
        sam_freeze_encoder=False,
    )
    
    model = PointHeatmapModelWithSAM(cfg).to(device)
    print(f"✓ 模型创建成功")
    
    # 测试前向传播
    batch_size = 2
    height, width = 240, 240
    x = torch.randn(batch_size, 3, height, width, device=device)
    cond = torch.randn(batch_size, 1, height, width, device=device)
    
    with torch.no_grad():
        logits, label_logits = model(x, cond)
    
    assert logits.shape == (batch_size, 1, height, width), f"Logits shape mismatch: {logits.shape}"
    assert label_logits.shape == (batch_size, 2), f"Label logits shape mismatch: {label_logits.shape}"
    print(f"✓ 前向传播成功: logits={logits.shape}, label_logits={label_logits.shape}")
    
    # 验证 LoRA 参数是否可训练
    lora_params = 0
    for name, param in model.named_parameters():
        if "lora" in name.lower() and param.requires_grad:
            lora_params += param.numel()
    
    assert lora_params > 0, "LoRA parameters should be trainable"
    print(f"✓ LoRA 参数可训练: {lora_params:,} 个参数")
    
    # 验证 SAM encoder 的非 LoRA 部分是否被冻结
    sam_non_lora_trainable = 0
    for name, param in model.sam_encoder.named_parameters():
        if "lora" not in name.lower() and param.requires_grad:
            sam_non_lora_trainable += param.numel()
    
    assert sam_non_lora_trainable == 0, f"Non-LoRA SAM params should be frozen but {sam_non_lora_trainable} are trainable"
    print(f"✓ SAM encoder 非 LoRA 部分已正确冻结")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = trainable_params / max(total_params, 1) * 100
    print(f"✓ 参数统计: total={total_params:,}, trainable={trainable_params:,} ({trainable_ratio:.2f}%)")
    print(f"✓ LoRA 参数占可训练参数: {lora_params / max(trainable_params, 1) * 100:.2f}%")
    
    return model


def test_checkpoint_compatibility():
    """测试 checkpoint 兼容性"""
    print_section("测试 4: Checkpoint 兼容性")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        # 测试标准模型 checkpoint
        print("\n[4.1] 测试标准模型 checkpoint")
        cfg1 = ModelConfig(backbone="unet_s", pretrained=False, main_in_channels=3, cond_in_channels=1)
        model1 = PointHeatmapModel(cfg1).to(device)
        
        ckpt_path = os.path.join(tmpdir, "standard_model.pt")
        optimizer = torch.optim.AdamW(model1.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler()
        
        save_checkpoint(ckpt_path, model1, optimizer, scaler, epoch=1, step=100)
        print(f"✓ 标准模型 checkpoint 保存成功")
        
        # 加载到相同类型的模型
        model1_loaded = PointHeatmapModel(cfg1).to(device)
        ckpt = load_checkpoint(ckpt_path, model1_loaded)
        print(f"✓ 标准模型 checkpoint 加载成功 (epoch={ckpt.epoch}, step={ckpt.step})")
        assert ckpt.metadata.get("use_sam_encoder") == False
        print(f"✓ Metadata 正确: use_sam_encoder={ckpt.metadata.get('use_sam_encoder')}")
        
        print("\n[4.2] 测试 checkpoint 保存的元数据")
        state = torch.load(ckpt_path, map_location="cpu")
        assert "metadata" in state, "Checkpoint should contain metadata"
        assert "version" in state, "Checkpoint should contain version"
        print(f"✓ Checkpoint 包含完整元数据")
        print(f"  - version: {state['version']}")
        print(f"  - metadata: {state['metadata']}")


def main():
    """主测试函数"""
    print("="*60)
    print("SAM Late LoRA 兼容性测试")
    print("="*60)
    
    # SAM checkpoint 路径（可以根据实际情况修改）
    sam_checkpoint = "third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
    
    # 测试 1: 标准模型
    try:
        model1 = test_standard_model()
        print("✅ 测试 1 通过")
    except Exception as e:
        print(f"❌ 测试 1 失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 测试 2: SAM 冻结模型
    try:
        model2 = test_sam_frozen_model(sam_checkpoint)
        if model2 is not None:
            print("✅ 测试 2 通过")
        else:
            print("⚠️  测试 2 跳过（SAM checkpoint 不存在）")
    except Exception as e:
        print(f"❌ 测试 2 失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 测试 3: SAM + LoRA 模型
    try:
        model3 = test_sam_lora_model(sam_checkpoint)
        if model3 is not None:
            print("✅ 测试 3 通过")
        else:
            print("⚠️  测试 3 跳过（SAM checkpoint 不存在）")
    except Exception as e:
        print(f"❌ 测试 3 失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 测试 4: Checkpoint 兼容性
    try:
        test_checkpoint_compatibility()
        print("✅ 测试 4 通过")
    except Exception as e:
        print(f"❌ 测试 4 失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print_section("所有测试完成")
    print("✅ 兼容性测试通过！")
    print("\n注意事项：")
    print("1. 当不使用 --use_sam_encoder 时，程序行为与原版完全一致")
    print("2. 当使用 --use_sam_encoder 但不使用 --sam_lora_enabled 时，SAM encoder 被冻结")
    print("3. 当同时使用 --use_sam_encoder 和 --sam_lora_enabled 时，启用 Late LoRA 微调")
    print("4. Checkpoint 包含模型类型元数据，支持自动兼容性检测")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

