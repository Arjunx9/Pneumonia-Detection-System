"""
Dataset Preparation Script
Helps organize and split the dataset into train/val/test sets
"""

import os
import shutil
import numpy as np


def prepare_dataset(source_dir, output_dir='dataset', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Prepare dataset by splitting into train/val/test sets
    
    Args:
        source_dir: Source directory containing class folders (NORMAL, PNEUMONIA)
        output_dir: Output directory for organized dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory not found: {source_dir}")
        print("Use the real path to your chest X-ray folder, e.g.:")
        print("  python utils/prepare_dataset.py --source C:\\Users\\Dell\\Downloads\\chest_xray")
        return
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    for split_dir in [train_dir, val_dir, test_dir]:
        os.makedirs(split_dir, exist_ok=True)
    
    # Get all class directories
    classes = [d for d in os.listdir(source_dir) 
               if os.path.isdir(os.path.join(source_dir, d)) and d.upper() in ['NORMAL', 'PNEUMONIA']]
    
    if not classes:
        print("Error: No class directories found (NORMAL, PNEUMONIA)")
        print(f"Found directories: {os.listdir(source_dir)}")
        return
    
    print(f"Found classes: {classes}")
    
    total_images = 0
    
    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        
        # Get all image files
        images = [f for f in os.listdir(class_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        
        if not images:
            print(f"Warning: No images found in {class_path}")
            continue
        
        # Shuffle
        np.random.seed(42)
        np.random.shuffle(images)
        
        # Split indices
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Create class directories in each split
        for split_dir in [train_dir, val_dir, test_dir]:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
        
        # Copy files
        print(f"\nProcessing {class_name}:")
        print(f"  Total images: {n_total}")
        
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_dir, class_name, img)
            shutil.copy2(src, dst)
        print(f"  Train: {len(train_images)} images")
        
        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_dir, class_name, img)
            shutil.copy2(src, dst)
        print(f"  Val: {len(val_images)} images")
        
        for img in test_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(test_dir, class_name, img)
            shutil.copy2(src, dst)
        print(f"  Test: {len(test_images)} images")
        
        total_images += n_total
    
    print(f"\nDataset preparation complete!")
    print(f"Total images processed: {total_images}")
    print(f"\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"    ├── train/")
    print(f"    ├── val/")
    print(f"    └── test/")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('--source', type=str, required=True,
                        help='Source directory containing class folders')
    parser.add_argument('--output', type=str, default='dataset',
                        help='Output directory (default: dataset)')
    parser.add_argument('--train', type=float, default=0.7,
                        help='Training ratio (default: 0.7)')
    parser.add_argument('--val', type=float, default=0.15,
                        help='Validation ratio (default: 0.15)')
    parser.add_argument('--test', type=float, default=0.15,
                        help='Test ratio (default: 0.15)')
    parser.add_argument('--max-images', type=int, default=1000,
                        help='Max total images to use (default: 1000). Set to 0 to use all images.')
    
    args = parser.parse_args()
    
    prepare_dataset(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        max_total_images=args.max_images if args.max_images > 0 else None
    )
