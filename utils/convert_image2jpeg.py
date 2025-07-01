import os
from PIL import Image
import argparse
from tqdm import tqdm

MAX_PIXELS = 89478485
MAX_SIZE = 4096

def convert_to_jpeg(directory, recursive=True, delete_original=False):
    supported_exts = ['.png', '.bmp', '.tiff', '.webp', '.jpeg', '.jpg']
    Image.MAX_IMAGE_PIXELS = None

    image_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in supported_exts:
                image_files.append(os.path.join(root, filename))
        if not recursive:
            break

    for input_path in tqdm(image_files, desc="Processing images"):
        root, filename = os.path.split(input_path)
        name, ext = os.path.splitext(filename)
        ext = ext.lower()
        output_path = os.path.join(root, f"{name}{ext}")

        try:
            with Image.open(input_path) as img:
                width, height = img.size
                num_pixels = width * height

                # Resize if image too large
                if num_pixels > MAX_PIXELS:
                    scale = min(MAX_SIZE / width, MAX_SIZE / height)
                    new_size = (int(width * scale), int(height * scale))
                    img = img.resize(new_size, Image.LANCZOS)

                if img.mode != 'RGB':
                    img = img.convert('RGB')

                img.save(output_path, "JPEG", quality=95)

            if delete_original and input_path != output_path:
                os.remove(input_path)

        except Exception as e:
            print(f"Failed to process {input_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images to JPEG and resize oversized images.")
    parser.add_argument("directory", type=str, help="Path to the directory containing images")
    parser.add_argument("--delete_original", action="store_true", help="Delete original images after conversion")
    args = parser.parse_args()
    convert_to_jpeg(args.directory, recursive=True, delete_original=args.delete_original)
