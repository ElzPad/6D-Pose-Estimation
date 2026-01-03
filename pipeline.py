import argparse
import os
import yaml
from tqdm import tqdm
import torch
import json
import cv2
import numpy as np

from ultralytics import YOLO
from geometry.pinhole_camera_model import pinhole_translation
from geometry.quaternion_to_rotation_matrix import quaternion_to_matrix
from dataset.linemod_dataset import LinemodDataset
from torch.utils.data import DataLoader
from augmentations import get_val_transforms
from models.yolo.load import load_yolo
from models.resnet.load import load_resnet

# Mapping: YOLO Class ID (0-12) -> LINEMOD Object ID (1-15)
YOLO_TO_LINEMOD_ID = {
    0: 1,
    1: 2,
    2: 4,
    3: 5,
    4: 6,
    5: 8,
    6: 9,
    7: 10,
    8: 11,
    9: 12,
    10: 13,
    11: 14,
    12: 15,
}


def get_args():
    parser = argparse.ArgumentParser(
        description="6D Pose Estimation Inference Pipeline"
    )

    # Data arguments
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data/linemod_yolo",
        help="Path to the preprocessed dataset root",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which split to run inference on",
    )
    parser.add_argument(
        "--models_info",
        type=str,
        default="data/linemod_yolo/models/models_info.yml",
        help="Path to LINEMOD_YOLO models_info.yml for object diameters",
    )

    # Model arguments
    parser.add_argument(
        "--yolo_weights", type=str, required=True, help="Path to best.pt"
    )
    parser.add_argument(
        "--resnet_weights", type=str, required=True, help="Path to resnet.pth"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    # Output arguments
    parser.add_argument(
        "--output_file",
        type=str,
        default="results.json",
        help="Path to save the output JSON file",
    )

    return parser.parse_args()


def linemod_collate_fn(batch):
    """
    Custom collate to handle dictionary targets and variable box counts.
    Returns:
        tuple(list_of_images, list_of_targets)
    """
    return tuple(zip(*batch))


def load_diameters(path):
    """
    Parses models_info.yml to extract object diameters in meters.

    Args:
        - path: path to the models_info.yml file

    Returns:
        - object diameters in meters
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"models_info.yml not found at {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # Map string ID to float diameter (converted mm to meters)
    # ===================================================================================
    # CONVERTION TO METERS IS DONE SINCE THAT IS THE STANDARD UNIT FOR 6D POSE ESTIMATION
    # ===================================================================================
    return {int(k): v["diameter"] / 1000.0 for k, v in data.items()}
    # return {int(k): v['diameter'] for k, v in data.items()} # keep it as mm


def save_results(results, output_path):
    """
    Saves the list of dictionaries to a JSON file.
    """
    print(f"Saving {len(results)} predictions to {output_path}...")

    # Ensure the directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Save complete.")

def parse_yolo_result(result):
    """
    Parses YOLO result and filters to keep ONLY the best (highest confidence) 
    bounding box for each distinct class ID found in the image.
    """
    if result.boxes.shape[0] == 0:
        return []

    # 1. Extract data to CPU numpy
    boxes = result.boxes.xywh.cpu().numpy()  # (N, 4)
    classes = result.boxes.cls.cpu().numpy() # (N,)
    confs = result.boxes.conf.cpu().numpy()  # (N,) # BEST w.r.t this value

    # 2. Filter: Keep only the best box per class
    unique_classes = np.unique(classes)
    keep_indices = []

    for cls in unique_classes:
        # Find all indices where the class matches
        cls_indices = np.where(classes == cls)[0]
        
        # Find the index (within the subset) that has the maximum confidence
        best_idx_subset = np.argmax(confs[cls_indices])
        
        # Map back to the global index
        best_idx_global = cls_indices[best_idx_subset]
        keep_indices.append(best_idx_global)

    # 3. Apply filter
    boxes = boxes[keep_indices]
    classes = classes[keep_indices]

    # 4. Stack and return [xc, yc, w, h, class_id]
    # We strictly respect the order expected by your pipeline
    detections = np.column_stack((boxes, classes))
    
    return detections.tolist()

def run_inference(dataloader, yolo_model, resnet_model, diameters, device):
    """
    Main pipeline loop:
    1. YOLO Detect
    2. Pinhole Translate
    3. ResNet Rotate
    """
    transformer = get_val_transforms(image_size=224)
    results = []
    yolo_model.eval()
    resnet_model.eval()
    print(f"Running inference on {len(dataloader.dataset)} images...")

    for batch_images, batch_targets in tqdm(dataloader):
        images_tensor = torch.stack(batch_images).to(device)

        # --- 1. YOLO Stage ---
        # Returns list of detections per image: [[class_id, x, y, w, h], ...]
        yolo_results = yolo_model(images_tensor)
        batch_detections = list(map(parse_yolo_result, yolo_results))

        # --- 2. Instance Processing ---
        for i, detections in enumerate(batch_detections):
            if detections is None or len(detections) == 0:
                continue

            # Extract Metadata from the new dictionary structure
            target = batch_targets[i]
            # Convert intrinsics to numpy
            K = target["intrinsics"].numpy()
            filename = target["image_id"]

            # Convert Tensor Image back to HWC Numpy for cropping/geometry if needed
            # Image is (C, H, W) normalized 0-1 -> (H, W, C) 0-255 uint8
            img_tensor = batch_images[i]
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(
                np.uint8
            )  # (H, W, C)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            for det in detections:
                x_c, y_c, w, h, cls_id = det

                # Retrieve diameter based on class ID
                linemod_id = YOLO_TO_LINEMOD_ID.get(int(cls_id))
                if linemod_id is None:
                    continue

                obj_diameter = diameters.get(linemod_id, 0.1)  # Default 10cm if missing

                # --- 3. Translation (Geometry) ---
                bbox = (x_c, y_c, w, h)

                # Ensure K is in the format your geometry module expects
                t_pred = pinhole_translation(*bbox, K, obj_diameter)

                # --- 4. Rotation (ResNet) ---
                # Preprocess patch (crop & resize)
                # Pass the numpy image and the bbox
                patch = transformer(img_np, bbox).to(device)

                with torch.no_grad():
                    # Unsqueeze to add batch dim: (1, C, H, W)
                    q_pred = resnet_model(patch.unsqueeze(0))

                # Normalize and convert
                q_pred = q_pred / torch.norm(q_pred)
                R_pred = quaternion_to_matrix(q_pred)

                # Save Result
                results.append(
                    {
                        "file": filename,
                        "obj_id": linemod_id,
                        "R": R_pred.cpu().numpy().tolist(),
                        "t": (
                            t_pred.tolist()
                            if isinstance(t_pred, np.ndarray)
                            else t_pred
                        ),
                    }
                )

    return results


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Setup Data
    print("Loading data...")
    diameters = load_diameters(args.models_info)

    # Using LinemodDataset (handles the intrinsic txt files)
    dataset = LinemodDataset(
        root_dir=args.dataset_root,
        split=args.split,
        img_ext=".png",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=linemod_collate_fn,
    )

    # 2. Load Models
    print("Loading models...")
    # Ensure these load functions handle putting model on `device` if needed
    yolo_model = load_yolo(args.yolo_weights)
    # If yolo needs to be explicitly moved:
    if hasattr(yolo_model, "to"):
        yolo_model.to(device)

    resnet_model = load_resnet(args.resnet_weights, device)

    # 3. Execution
    all_poses = run_inference(dataloader, yolo_model, resnet_model, diameters, device)

    print(f"Done. Processed {len(all_poses)} poses.")

    # 4. Save Results
    save_results(all_poses, args.output_file)


if __name__ == "__main__":
    main()
