import argparse
import yaml
import os
import shutil
import random
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# Mapping ID Linemod -> ID YOLO
CLASS_MAPPING = {
    1: 0,   
    2: 1,    
    4: 2, 
    5: 3, 
    6: 4,
    8: 5,
    9: 6,
    10: 7, 
    11: 8, 
    12: 9, 
    13: 10,
    14: 11,
    15: 12
}

def convert_bbox_to_yolo(size, box):
    """
        We convert from [xmin, ymin, w, h] to [x_center, y_center, w, h] 
        and then we normalize it.

        Args:
            - size: image size (width, height) 
            - box: [xmin, ymin, w, h]
        
        Returns:
            - tuple (x_center, y_center, width, height) 

    """
    
    img_w, img_h = size
    dw = 1.0 / img_w
    dh = 1.0 / img_h

    x_min, y_min, w_pixels, h_pixels = map(float, box)

    x_center = (x_min + w_pixels / 2.0) * dw
    y_center = (y_min + h_pixels / 2.0) * dh
    width = w_pixels * dw
    height = h_pixels * dh

    # clamp centers (optional safety)
    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)

    return (x_center, y_center, width, height)

def setup_directories(output_dir, clean_output=False):
    """
        Creates the structure. Optionally wipe output_dir first to avoid leakage across runs.
    """
    output_dir = Path(output_dir)
    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'depths', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'camera_intrinsics', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'pose_labels', split), exist_ok=True)

    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

def write_data_yaml(output_dir):
    data_yml = (
        "path: data/linemod_yolo\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  - ape\n"
        "  - benchvise\n"
        "  - camera\n"
        "  - can\n"
        "  - cat\n"
        "  - driller\n"
        "  - duck\n"
        "  - eggbox\n"
        "  - glue\n"
        "  - holepuncher\n"
        "  - iron\n"
        "  - lamp\n"
        "  - phone\n"
    )

    with open(os.path.join(output_dir, "data.yml"), "w") as f:
        f.write(data_yml)

def split_frame_ids(ground_truth, train_ratio, seed):
    rng = random.Random(seed)
    frame_ids = sorted([int(k) for k in ground_truth.keys()])
    rng.shuffle(frame_ids)

    n_train = int(len(frame_ids) * train_ratio)
    train_ids = set(frame_ids[:n_train])
    val_ids = set(frame_ids[n_train:])

    return train_ids, val_ids

def process_dataset(dataset_root, output_dir, training_ratio=0.8, seed=42, clean_output=True):
    """
        Convert raw LINEMOD dataset into YOLO detection format.

        Args:
            - dataset_root: data/linemod/ -> linemod directory path
            - output_dir:  data/linemod_yolo/ -> output directory path
            - training_ratio: ratio between training data and validation data
        
        Returns:
            - new data (preprocessed according to what YOLO needs)
    """

    setup_directories(output_dir, clean_output=clean_output)
    root_path = Path(dataset_root)
    data_path = Path(os.path.join(root_path, 'data'))
    model_path = os.path.join(root_path, 'models')

    # subfolders contains all the subfolders (e.g: "01", "02" etc)
    subfolders = [f for f in data_path.iterdir() if f.is_dir()]

    total_images = 0

    for folder in subfolders:
        print(f"Processing folder: {folder.name}")

        gt_file = folder / 'gt.yml'
        rgb_dir = folder / 'rgb'
        depth_dir = folder / 'depth'
        info_file = folder / 'info.yml'

        if not info_file.exists():
            print(f"ALERT: info.yml NOT FOUND in {folder.name}. I'll skip this folder")
            continue

        if not gt_file.exists():
            print(f"ALERT: gt.yml NOT FOUND in {folder.name}. I'll skip this folder")
            continue

        with open(info_file, 'r') as f:
            infos = yaml.load(f, Loader=yaml.SafeLoader)

        with open(gt_file, 'r') as f:
            # Safe open for YAML file
            ground_truth = yaml.load(f, Loader=yaml.SafeLoader)

        if not ground_truth:
            continue

        # Deterministic split per object folder
        folder_seed = seed + int(folder.name)
        train_ids, val_ids = split_frame_ids(ground_truth, training_ratio, folder_seed)

        loop = tqdm(ground_truth.items(), desc=f"Proc. {folder.name}", unit="img")
    
        # We iterate over each frame in gt.yml
        for img_id, objects in loop:
            # The filename of the image inside rgb folder is 0000.png, 0001.png ...
            # So we can get the img_id, that is simply an integer (e.g: 0, 1), from the YAML and then 
            # extend it to create the filename of the image by extending the img_id
            img_filename = f"{img_id:04d}.png"
            src_img_path = rgb_dir / img_filename
            src_depth_path = depth_dir / img_filename

            if not src_img_path.exists():
                continue #skip

            with Image.open(src_img_path) as img:
                img_w, img_h = img.size

            # We prepare the label for the current image
            yolo_labels = []
            pose_labels = []  # Pose ground truth: rotation + translation (separate from YOLO labels)
            has_valid_object = False

            # The folder name (e.g., "02") indicates the target object for this folder
            # We only want to save the pose for THIS specific object, not all objects in the scene
            target_object_id = int(folder.name)
            
            for obj in objects:
                original_id = obj['obj_id']
            
                # ONLY process the target object for this folder
                # (e.g., folder "02" should only contain data for obj_id 2)
                # This ensures bbox and pose labels are always aligned
                if original_id == target_object_id and original_id in CLASS_MAPPING:
                    class_id = CLASS_MAPPING[original_id]
                    bbox = obj['obj_bb']
                    
                    # Convert bbox to YOLO format
                    yolo_bbox = convert_bbox_to_yolo((img_w, img_h), bbox)
                    
                    # Formatted string --> class_id x_center y_center width height
                    label_str = f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
                    yolo_labels.append(label_str)
                    
                    # Pose ground truth: class_id R11 R12 R13 R21 R22 R23 R31 R32 R33 t1 t2 t3
                    # cam_R_m2c is a 3x3 rotation matrix stored as a flat list (row-major)
                    # cam_t_m2c is a 3D translation vector
                    rot_matrix = obj['cam_R_m2c']
                    translation = obj['cam_t_m2c']
                    pose_str = f"{class_id} " + " ".join(f"{r:.8f}" for r in rot_matrix) + " " + " ".join(f"{t:.8f}" for t in translation)
                    pose_labels.append(pose_str)
                    
                    has_valid_object = True
            
            # Camera instrinsics --> 
            # 0. fx
            # 1. always 0
            # 2. cx
            # 3. always 0
            # 4. fy
            # 5. cy
            # 6. always 0
            # 7. always 0
            # 8. always 1
            camera_str = f"{infos[img_id]['cam_K'][0]} {infos[img_id]['cam_K'][1]} {infos[img_id]['cam_K'][2]} {infos[img_id]['cam_K'][3]} {infos[img_id]['cam_K'][4]} {infos[img_id]['cam_K'][5]} {infos[img_id]['cam_K'][6]} {infos[img_id]['cam_K'][7]} {infos[img_id]['cam_K'][8]}"
            

            # If the image contains at least one object of our interest, we save it
            if has_valid_object:

                split = 'train' if img_id in train_ids else 'val'
                
                # We create a unique name: foldername_imgname.png
                # Es: 01_0000.png
                unique_name = f"{folder.name}_{img_filename}"
                unique_txt = unique_camera = f"{folder.name}_{img_id:04d}.txt"
                
                dst_img_path = os.path.join(output_dir, 'images', split, unique_name)
                dst_depth_path = os.path.join(output_dir, 'depths', split, unique_name)
                dst_label_path = os.path.join(output_dir, 'labels', split, unique_txt)
                dst_camera_intrinsics_path = os.path.join(output_dir, 'camera_intrinsics', split, unique_camera)
                dst_pose_path = os.path.join(output_dir, 'pose_labels', split, unique_txt)
                
                # Image (RGB)
                shutil.copy(src_img_path, dst_img_path)
                
                # Depth image
                if src_depth_path.exists():
                    shutil.copy(src_depth_path, dst_depth_path)
                
                # Labeling (YOLO format for detection)
                with open(dst_label_path, 'w') as f_out:
                    f_out.write('\n'.join(yolo_labels))

                # Camera
                with open(dst_camera_intrinsics_path, 'w') as f_out:
                    f_out.write(camera_str)
                
                # Pose labels (rotation matrix + translation)
                with open(dst_pose_path, 'w') as f_out:
                    f_out.write('\n'.join(pose_labels))
                
                total_images += 1
    
    dst_models_path = os.path.join(output_dir, 'models')
    models_info_path = os.path.join(model_path, 'models_info.yml')
    shutil.copy(models_info_path, dst_models_path)

    # Copy 3D object models (obj_01.ply to obj_15.ply)
    for obj_id in range(1, 16):
        src_model = os.path.join(model_path, f"obj_{obj_id:02d}.ply")
        dst_model = os.path.join(dst_models_path, f"obj_{obj_id:02d}.ply")
        if os.path.exists(src_model):
            shutil.copy(src_model, dst_model)
        else:
            print(f"WARNING: Model file {src_model} not found.")

    write_data_yaml(output_dir)

    print(f"\nDone! Dataset generated in '{output_dir}'.")
    print(f"All {total_images} images are in 'images/train'.")

def main():
    parser = argparse.ArgumentParser(description='Preprocessing Linemod dataset to YOLO input format.')
    parser.add_argument('--no_clean_output', action="store_true", help='Do not clean the output folder before preprocessing.')
    args = parser.parse_args()

    dataset_path = "data/linemod/Linemod_preprocessed"
    output_dir = "data/linemod_yolo"
    process_dataset(dataset_path, output_dir, training_ratio=0.8, clean_output=not args.no_clean_output)

if __name__ == "__main__":
    main()

