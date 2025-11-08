#!/bin/bash
# PEFT快速开始脚本
# 用于验证环境和运行小规模测试

set -e

echo "=================================="
echo "PEFT Quick Start Script"
echo "=================================="

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# 步骤1: 检查Python环境
# ============================================================================
echo -e "\n${YELLOW}[1/6] Checking Python environment...${NC}"

if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python ${PYTHON_VERSION}${NC}"

# ============================================================================
# 步骤2: 检查PyTorch
# ============================================================================
echo -e "\n${YELLOW}[2/6] Checking PyTorch...${NC}"

python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null || {
    echo -e "${RED}Error: PyTorch not installed${NC}"
    echo "Install with: pip install torch torchvision"
    exit 1
}

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
echo -e "${GREEN}✓ PyTorch OK${NC}"

# ============================================================================
# 步骤3: 检查SAM2
# ============================================================================
echo -e "\n${YELLOW}[3/6] Checking SAM2...${NC}"

if [ ! -d "third_party/sam2" ]; then
    echo -e "${RED}Error: SAM2 not found in third_party/sam2${NC}"
    echo "Clone with: git clone https://github.com/facebookresearch/segment-anything-2.git third_party/sam2"
    exit 1
fi

python -c "import sys; sys.path.insert(0, 'third_party/sam2'); from sam2.build_sam import build_sam2" 2>/dev/null || {
    echo -e "${RED}Error: SAM2 not properly installed${NC}"
    echo "Install with: cd third_party/sam2 && pip install -e ."
    exit 1
}

echo -e "${GREEN}✓ SAM2 OK${NC}"

# ============================================================================
# 步骤4: 检查SAM2权重
# ============================================================================
echo -e "\n${YELLOW}[4/6] Checking SAM2 checkpoint...${NC}"

SAM_CKPT="third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
if [ ! -f "${SAM_CKPT}" ]; then
    echo -e "${YELLOW}Warning: SAM2 checkpoint not found${NC}"
    echo "Download with:"
    echo "  wget -O ${SAM_CKPT} https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    SAM_CKPT_OK=false
else
    echo -e "${GREEN}✓ SAM2 checkpoint found${NC}"
    SAM_CKPT_OK=true
fi

# ============================================================================
# 步骤5: 测试PEFT模块
# ============================================================================
echo -e "\n${YELLOW}[5/6] Testing PEFT modules...${NC}"

python -m seg-rl.peft.test_modules --device cpu 2>&1 | grep -q "All tests passed" && {
    echo -e "${GREEN}✓ PEFT modules OK${NC}"
} || {
    echo -e "${RED}Error: PEFT module tests failed${NC}"
    echo "Run: python -m seg-rl.peft.test_modules --device cpu"
    exit 1
}

# ============================================================================
# 步骤6: 运行小规模演示（可选）
# ============================================================================
if [ "$SAM_CKPT_OK" = true ]; then
    echo -e "\n${YELLOW}[6/6] Running small demo (optional)...${NC}"
    echo "This will test the complete pipeline with dummy data."
    echo -n "Continue? (y/n) "
    read -r response
    
    if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
        # 创建临时目录
        DEMO_DIR="outputs/demo_peft"
        mkdir -p ${DEMO_DIR}
        
        echo "Creating dummy data..."
        python -c "
import numpy as np
from PIL import Image
import os

# 创建临时目录
os.makedirs('${DEMO_DIR}/images', exist_ok=True)
os.makedirs('${DEMO_DIR}/masks', exist_ok=True)

# 生成2个虚拟样本
for i in range(2):
    # 随机图像
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    Image.fromarray(img).save(f'${DEMO_DIR}/images/sample_{i:03d}.jpg')
    
    # 随机掩模（中心有一个圆形前景）
    mask = np.zeros((256, 256), dtype=np.uint8)
    y, x = np.ogrid[:256, :256]
    center_y, center_x = 128, 128
    radius = 50
    circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    mask[circle_mask] = 255
    Image.fromarray(mask).save(f'${DEMO_DIR}/masks/sample_{i:03d}.png')

print('Dummy data created!')
"
        
        echo "Generating initial points..."
        python seg-rl/annotator/gen_point_jsonl_from_masks.py \
            --images_dir ${DEMO_DIR}/images \
            --masks_dir ${DEMO_DIR}/masks \
            --output_jsonl ${DEMO_DIR}/demo.jsonl
        
        echo "Running SAM2 segmentation..."
        python seg-rl/sam2_segment_from_points.py \
            --input_jsonl ${DEMO_DIR}/demo.jsonl \
            --json_output ${DEMO_DIR}/demo.jsonl \
            --output_dir ${DEMO_DIR}/sam_masks \
            --sam_checkpoint ${SAM_CKPT} \
            --device cpu \
            --resize 256 256
        
        echo "Training for 2 epochs (demo)..."
        python -m seg-rl.peft.train_supervised_peft \
            --jsonl ${DEMO_DIR}/demo.jsonl \
            --sam_checkpoint ${SAM_CKPT} \
            --lora_rank 8 --lora_alpha 16 \
            --image_size 256 256 \
            --batch_size 2 --epochs 2 \
            --lr_sam 1e-5 --lr_point 1e-4 \
            --out_dir ${DEMO_DIR}/model \
            --device cpu
        
        echo -e "${GREEN}✓ Demo completed successfully!${NC}"
        echo "Results saved in: ${DEMO_DIR}"
    else
        echo "Skipping demo."
    fi
else
    echo -e "\n${YELLOW}[6/6] Skipping demo (SAM2 checkpoint not found)${NC}"
fi

# ============================================================================
# 总结
# ============================================================================
echo ""
echo "=================================="
echo -e "${GREEN}Quick Start Check Complete!${NC}"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Prepare your dataset (see README_PEFT_EXPERIMENT_GUIDE.md)"
echo "  2. Generate training data with gen_point_jsonl_from_masks.py"
echo "  3. Run supervised training with train_supervised_peft.py"
echo "  4. Run GRPO training with train_grpo_peft.py"
echo ""
echo "For detailed instructions, see:"
echo "  - README_PEFT_EXPERIMENT_GUIDE.md"
echo "  - seg-rl/peft/README.md"
echo ""

