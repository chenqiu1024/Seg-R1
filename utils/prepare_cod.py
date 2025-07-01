import os
import shutil

def delete_noncams(root_dir):
    deleted_count = 0
    for current_dir, _, files in os.walk(root_dir):
        for file in files:
            if "NonCAM" in file:
                file_path = os.path.join(current_dir, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

def split_gt_by_image(base_dir):
    image_dir = os.path.join(base_dir, 'Images')
    gt_dir = os.path.join(base_dir, 'GT')

    gt_train_dir = os.path.join(gt_dir, 'Train')
    gt_test_dir = os.path.join(gt_dir, 'Test')

    os.makedirs(gt_train_dir, exist_ok=True)
    os.makedirs(gt_test_dir, exist_ok=True)

    train_images = os.listdir(os.path.join(image_dir, 'Train'))
    test_images = os.listdir(os.path.join(image_dir, 'Test'))

    train_names = [os.path.splitext(f)[0] for f in train_images]
    test_names = [os.path.splitext(f)[0] for f in test_images]

    moved_train, moved_test = 0, 0
    for name in train_names:
        src = os.path.join(gt_dir, name + '.png')
        dst = os.path.join(gt_train_dir, name + '.png')
        if os.path.exists(src):
            shutil.move(src, dst)
            moved_train += 1

    for name in test_names:
        src = os.path.join(gt_dir, name + '.png')
        dst = os.path.join(gt_test_dir, name + '.png')
        if os.path.exists(src):
            shutil.move(src, dst)
            moved_test += 1


if __name__ == "__main__":
    camo_base_dir = "datasets/CAMO-V.1.0-CVIU2019"
    cod10k_base_dir = "datasets/COD10K-v3"
    delete_noncams(cod10k_base_dir)
    split_gt_by_image(camo_base_dir)
