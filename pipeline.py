import argparse
import os
import yaml
from tqdm import tqdm
import torch
import json
import cv2
import numpy as np
import trimesh

from ultralytics import YOLO
from geometry.pinhole_camera_model import pinhole_translation
from geometry.quaternion_to_rotation_matrix import quaternion_to_matrix
from dataset.linemod_dataset import LinemodInferenceDataset, collate_fn
from torch.utils.data import DataLoader
from augmentations import get_val_transforms
from models.yolo.load import load_yolo
from models.resnet.load import load_resnet

from metrics.add import compute_add, compute_add_s

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

# LINEMOD IDs that are symmetric (require ADD-S metric)
# 10: Eggbox, 11: Glue
SYMMETRIC_IDS = [10, 11]


def get_args():
    parser = argparse.ArgumentParser(
        description="6D Pose Estimation Inference Pipeline"
    )

    # Data arguments
    parser.add_argument(
        "--yolo_dataset_root",
        type=str,
        default="data/linemod_yolo",
        help="Path to the preprocessed dataset root",
    )

    parser.add_argument(
        "--linemod_orig_root",
        type=str,
        default="data/linemod/Linemod_preprocessed",
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
    #return {int(k): v["diameter"] / 1000.0 for k, v in data.items()}
    return {int(k): v['diameter'] for k, v in data.items()} # keep it as mm


# models_dir should point to. data/linemod/Linemod_preprocessed/models
def load_meshes(models_dir): 
    """
    Loads .ply meshes for all objects in YOLO_TO_LINEMOD_ID.
    Returns a dict: {obj_id: torch.Tensor(N, 3)}
    """
    print(f"Loading 3D meshes from {models_dir}...")
    meshes = {}
    
    # Iterate over unique Linemod IDs
    unique_ids = set(YOLO_TO_LINEMOD_ID.values())
    
    for obj_id in unique_ids:
        # filename format: obj_01.ply, obj_05.ply, etc.
        filename = f"obj_{obj_id:02d}.ply"
        path = os.path.join(models_dir, filename)
        
        if not os.path.exists(path):
            print(f"Warning: Mesh not found for ID {obj_id} at {path}. Metrics for this object will fail.")
            continue
            
        # Use trimesh to load vertices
        try:
            mesh = trimesh.load(path)
            # trimesh.load can return Scene or Trimesh
            if isinstance(mesh, trimesh.Scene):
                 # Concatenate all geometries if it's a scene
                vertices = np.concatenate([g.vertices for g in mesh.geometry.values()])
            else:
                vertices = mesh.vertices
            
            # Subsample if too heavy? usually ~3000 points is fine for ADD
            # Storing as float32 tensor
            meshes[obj_id] = torch.tensor(vertices, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            
    return meshes


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

def run_inference(dataloader, yolo_model, resnet_model, diameters, meshes, device):
    """
    Main pipeline loop:
    1. YOLO Detect
    2. Pinhole Translate
    3. ResNet Rotate
    4. Compute ADD/ADD-S Metrics
    """
    transformer = get_val_transforms(image_size=224)
    results = []

    # Metric Accumulators
    metrics = {"add": [], "add_s": []}
    pass_count = 0  # Number of poses with error < 0.1 * diameter
    total_poses = 0

    yolo_model.eval()
    resnet_model.eval()
    print(f"Running inference on {len(dataloader.dataset)} images...")

    for batch_images, batch_targets in tqdm(dataloader):
        images_tensor = torch.stack(batch_images).to(device)

        # --- 1. YOLO Stage ---
        # Returns list of detections per image: [[class_id, x, y, w, h], ...]
        yolo_results = yolo_model(images_tensor)

        # --- 2. Instance Processing ---
        for i, result in enumerate(yolo_results):
            detections = parse_yolo_result(result)
            if not detections: continue

            # Extract Metadata from the new dictionary structure
            target = batch_targets[i]
            filename = target['image_id']
            # Convert intrinsics to numpy
            K = target['intrinsics'].numpy()

            # Ground Truth Data (for evaluation)
            gt_id = target['gt_class_id']
            gt_R = target['gt_R'].to(device)
            gt_t = target['gt_t'].to(device)

            # Convert Tensor Image back to HWC Numpy for cropping/geometry if needed
            # Image is (C, H, W) normalized 0-1 -> (H, W, C) 0-255 uint8
            img_tensor = batch_images[i]
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(
                np.uint8
            )  # (H, W, C)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            for det in detections:
                x_c, y_c, w, h, cls_id = det
                bbox = (x_c, y_c, w, h)

                # Retrieve diameter based on class ID
                linemod_id = YOLO_TO_LINEMOD_ID.get(int(cls_id))
                if linemod_id is None:
                    continue

                # Only evaluate if detection matches the GT object in this file
                # (Since your files are named 01_xxxx, they contain object 01)
                if linemod_id != gt_id:
                    continue


                # --- 3. Translation (Geometry) ---
                obj_diameter = diameters.get(linemod_id, 0.1)  # Default 10cm if missing
                
                # Extract camera intrinsics from K matrix
                # K = [[f_x,  0, c_x],
                #      [ 0, f_y, c_y],
                #      [ 0,  0,  1]]
                f_x = K[0, 0]
                f_y = K[1, 1]
                c_x = K[0, 2]
                c_y = K[1, 2]
            
                t_pred = pinhole_translation(bbox, f_x, f_y, c_x, c_y, obj_diameter)

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

                # Result Dictionary
                res_entry = {
                    "file": filename,
                    "obj_id": linemod_id,
                    "R": R_pred.cpu().numpy().tolist(),
                    "t": t_pred.cpu().numpy().flatten().tolist(),
                }

                # --- 5. Metric Evaluation ---
                model_pts = meshes[linemod_id].to(device)
                
                if linemod_id in SYMMETRIC_IDS:
                    err = compute_add_s(R_pred, t_pred, gt_R, gt_t, model_pts)
                    metrics["add_s"].append(err)
                else:
                    err = compute_add(R_pred, t_pred, gt_R, gt_t, model_pts)
                    metrics["add"].append(err)
                
                # Pass threshold: 10% of diameter
                if err < (0.1 * obj_diameter):
                    pass_count += 1
                total_count += 1

                results.append(res_entry)

    # --- Summary ---
    if total_count > 0:
        print("\n" + "="*30)
        print(f"Total Evaluated: {total_count}")
        print(f"Accuracy (ADD < 0.1d): {100 * pass_count / total_count:.2f}%")
        print("="*30)
    
    return results



def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Setup Data
    print("Loading data...")
    diameters = load_diameters(args.models_info)

    # Using LinemodInferenceDataset 
    dataset = LinemodInferenceDataset(
        yolo_root=args.yolo_dataset_root,
        linemod_orig_root = args.linemod_orig_root,
        split=args.split,
        img_ext=".png"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
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
