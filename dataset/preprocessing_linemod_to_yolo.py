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
        We convert from [xmin, ymin, w, h] to [x_center, y_center, w, h] and then normalized

        => box = [xmin, ymin, w, h]


        Parameters
            - size = image size (width, height) 
            - box = [xmin, ymin, w, h]
    """
    
    dw = 1. / size[0] # size[0] = width
    dh = 1. / size[1] # size[1] = height

    x_min = box[0]
    y_min = box[1]
    w_pixels = box[2]
    h_pixels = box[3]


    # Center coordinates (x, y)
    x_center = x_min + w_pixels / 2.0
    y_center = y_min + h_pixels / 2.0

    # Normalization
    x_center *= dw
    width = w_pixels * dw
    y_center *= dh
    height = h_pixels * dh

    return (x_center, y_center, width, height)



def setup_directories(output_dir):
    """ Creates the structure"""
    output_dir = Path(output_dir)
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'camera_intrinsics', split), exist_ok=True)

    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

def process_dataset(dataset_root, output_dir, training_ratio):
    """
        Convert raw LINEMOD dataset into YOLO detection format.

        Parameters:
            - dataset_root: data/linemod/ -> linemod directory path
            - output_dir:  data/linemod_yolo/ -> output directory path
            - training_ratio: ratio between training data and validation data
    """

    setup_directories(output_dir)
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
        info_file = folder / 'info.yml'

        if not info_file.exists():
            print(f"ALERT: info.yml NOT FOUND in {folder.name}. I'll skip this folder")
            continue

        with open(info_file, 'r') as f:
            infos = yaml.load(f, Loader=yaml.SafeLoader)

        if not gt_file.exists():
            print(f"ALERT: gt.yml NOT FOUND in {folder.name}. I'll skip this folder")
            continue

        with open(gt_file, 'r') as f:
            # Safe open for YAML file
            ground_truth = yaml.load(f, Loader=yaml.SafeLoader)

        if not ground_truth:
            continue

        loop = tqdm(ground_truth.items(), desc=f"Proc. {folder.name}", unit="img")
    
        # We iterate over each frame in gt.yml
        for img_id, objects in loop:
            # The filename of the image inside rgb folder is 0000.png, 0001.png ...
            # So we can get the img_id, that is simply an integer (e.g: 0, 1), from the YAML and then 
            # extend it to create the filename of the image by extending the img_id
            img_filename = f"{img_id:04d}.png"
            src_img_path = rgb_dir / img_filename

            if not src_img_path.exists():
                continue #skip

            with Image.open(src_img_path) as img:
                img_w, img_h = img.size

            # We prepare the label for the current image
            yolo_labels = []
            has_valid_object = False

            for obj in objects:
                original_id = obj['obj_id']
            
                # If the object of interest is in our list
                if original_id in CLASS_MAPPING:
                    class_id = CLASS_MAPPING[original_id]
                    bbox = obj['obj_bb']
                    
                    # Convert
                    yolo_bbox = convert_bbox_to_yolo((img_w, img_h), bbox)
                    
                    # Formatted string --> class_id x_center y_center width height
                    label_str = f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
                    yolo_labels.append(label_str)
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

                split = 'train' if random.random() < training_ratio else 'val'
                
                # We create a unique name: foldername_imgname.png
                # Es: 01_0000.png
                unique_name = f"{folder.name}_{img_filename}"
                unique_txt = unique_camera = f"{folder.name}_{img_id:04d}.txt"
                
                
                dst_img_path = os.path.join(output_dir, 'images', split, unique_name)
                dst_label_path = os.path.join(output_dir, 'labels', split, unique_txt)
                dst_camera_intrinsics_path = os.path.join(output_dir, 'camera_intrinsics', split, unique_camera)
                
                # Image
                shutil.copy(src_img_path, dst_img_path)
                
                # Labeling
                with open(dst_label_path, 'w') as f_out:
                    f_out.write('\n'.join(yolo_labels))

                # Camera
                with open(dst_camera_intrinsics_path, 'w') as f_out:
                    f_out.write(camera_str)
                
                
                total_images += 1
    
    dst_models_path = os.path.join(output_dir, 'models')
    models_info_path = os.path.join(model_path, 'models_info.yml')
    shutil.copy(models_info_path, dst_models_path)
        
    print(f"\Done! Dataset generated in '{output_dir}'.")
    print(f"All {total_images} images are in 'images/train'.")


if __name__ == "__main__":
    dataset_path = "data/linemod/Linemod_preprocessed"
    output_dir = "data/linemod_yolo"
    process_dataset(dataset_path, output_dir, 0.8)

