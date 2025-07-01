#!/bin/bash

# Prepare datasets for Seg-R1 training
# This script converts images to JPEG and removes NonCAM samples from COD10K

set -e  # Exit on any error

python utils/convert_image2jpeg.py datasets/COD10K-v3/Train/Image --delete_original

python utils/convert_image2jpeg.py datasets/DIS5K/DIS-TR/im --delete_original

python utils/convert_image2jpeg.py datasets/DIS5K/DIS-TR/gt --delete_original

python utils/convert_image2jpeg.py datasets/CAMO-V.1.0-CVIU2019/Images/Train --delete_original

python utils/prepare_cod.py

python utils/convert_image2jpeg.py datasets/DUTS/DUTS-TR/DUTS-TR-Image --delete_original