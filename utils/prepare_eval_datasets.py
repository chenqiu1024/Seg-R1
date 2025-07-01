import os
import json
import argparse

def process(image_dir, gt_dir, output_file, root_dir):
    if not os.path.exists(image_dir) or not os.path.exists(gt_dir):
        print("Error: Image or ground truth directory does not exist")
        exit(1)

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])
    if not image_files:
        print("Error: No JPG images found in image directory")
        exit(1)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        count = 0
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            gt_path = os.path.join(gt_dir, img_file.replace('.jpg', '.png'))

            if not os.path.exists(gt_path):
                print(f"Warning: Missing ground truth for {img_file}, skipping")
                continue

            img_rel_path = os.path.relpath(img_path, root_dir)
            gt_rel_path = os.path.relpath(gt_path, root_dir)

            entry = {
                "image_path": img_rel_path,
                "ground_truth": gt_rel_path,
            }
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
            count += 1

    print(f"Successfully created dataset with {count} entries in {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSONL dataset entries for image and ground truth masks.")
    parser.add_argument('--image_dir', type=str, required=True, help='Path to directory containing input images (.jpg)')
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to directory containing ground truth masks (.png)')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output JSONL file')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory to make paths relative to')

    args = parser.parse_args()
    process(args.image_dir, args.gt_dir, args.output_file, args.root_dir)
