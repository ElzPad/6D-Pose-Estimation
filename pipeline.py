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
from dataset.linemod_inference_dataset import LinemodInferenceDataset, collate_fn
from torch.utils.data import DataLoader
from augmentations import get_val_transforms
from models.yolo.load import load_yolo
from models.resnet.load import load_resnet

from metrics.add import compute_add, compute_add_s

# Mapping: YOLO Class ID (0-12) -> LINEMOD Object ID (1-15)
YOLO_TO_LINEMOD_ID = {
    0: 1, 1: 2, 2: 4, 3: 5, 4: 6, 5: 8, 6: 9, 
    7: 10, 8: 11, 9: 12, 10: 13, 11: 14, 12: 15,
}

SYMMETRIC_IDS = [10, 11]

def get_args():
    parser = argparse.ArgumentParser(description="6D Pose Estimation Inference Pipeline")

    parser.add_argument("--dataset_root", type=str, default="data/linemod_yolo")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--models_info", type=str, default="data/linemod_yolo/models/models_info.yml")
    parser.add_argument("--models_dir", type=str, default="data/linemod/Linemod_preprocessed/models", 
                        help="Path to directory containing .ply files")

    parser.add_argument("--yolo_weights", type=str, required=True)
    parser.add_argument("--resnet_weights", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_file", type=str, default="results.json")

    return parser.parse_args()

def load_diameters(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"models_info.yml not found at {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return {int(k): v['diameter'] for k, v in data.items()}

def load_meshes(models_dir): 
    print(f"Loading 3D meshes from {models_dir}...")
    meshes = {}
    unique_ids = set(YOLO_TO_LINEMOD_ID.values())
    
    for obj_id in unique_ids:
        filename = f"obj_{obj_id:02d}.ply"
        path = os.path.join(models_dir, filename)
        
        if not os.path.exists(path):
            print(f"Warning: Mesh not found for ID {obj_id} at {path}. Metrics will fail.")
            continue
            
        try:
            mesh = trimesh.load(path)
            if isinstance(mesh, trimesh.Scene):
                vertices = np.concatenate([g.vertices for g in mesh.geometry.values()])
            else:
                vertices = mesh.vertices
            meshes[obj_id] = torch.tensor(vertices, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            
    return meshes

def save_results(results, output_path):
    print(f"Saving {len(results)} predictions to {output_path}...")
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Save complete.")

def parse_yolo_result(result):
    if result.boxes.shape[0] == 0:
        return []
    boxes = result.boxes.xywh.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    unique_classes = np.unique(classes)
    keep_indices = []

    for cls in unique_classes:
        cls_indices = np.where(classes == cls)[0]
        best_idx_subset = np.argmax(confs[cls_indices])
        keep_indices.append(cls_indices[best_idx_subset])

    boxes = boxes[keep_indices]
    classes = classes[keep_indices]
    detections = np.column_stack((boxes, classes))
    return detections.tolist()

def run_inference(dataloader, yolo_model, resnet_model, diameters, meshes, device):
    transformer = get_val_transforms(image_size=224)
    results = []
    metrics = {"add": [], "add_s": []}
    pass_count = 0
    total_count = 0
    
    # Check if ResNet uses object conditioning
    use_object_id = getattr(resnet_model, 'use_object_id', False)
    if use_object_id:
        print("ResNet using object identity conditioning")

    yolo_model.eval()
    resnet_model.eval()
    print(f"Running inference on {len(dataloader.dataset)} images...")

    for batch_images, batch_targets in tqdm(dataloader):
        images_tensor = torch.stack(batch_images).to(device)

        # 1. YOLO Stage
        yolo_results = yolo_model(images_tensor, verbose=False)

        # 2. Instance Processing
        for i, result in enumerate(yolo_results):
            detections = parse_yolo_result(result)
            if not detections: continue

            target = batch_targets[i]
            filename = target['image_id']
            K = target['intrinsics'].numpy()
            gt_id = int(target['gt_class_id'])
            gt_R = target['gt_R'].to(device)
            gt_t = target['gt_t'].to(device)

            img_tensor = batch_images[i]
            # Keep as RGB - the transformer expects RGB (same as training)
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            for det in detections:
                x_c, y_c, w, h, cls_id = det
                bbox = (x_c, y_c, w, h)
                linemod_id = YOLO_TO_LINEMOD_ID.get(int(cls_id))
                if linemod_id is None or linemod_id != gt_id:
                    continue

                # 3. Rotation & Depth Scale
                patch = transformer(img_np, bbox).to(device)
                with torch.no_grad():
                    if use_object_id:
                        object_ids = torch.tensor([linemod_id], device=device)
                        q_pred, depth_scale = resnet_model(patch.unsqueeze(0), object_ids)
                    else:
                        q_pred, depth_scale = resnet_model(patch.unsqueeze(0), None)
                q_pred = q_pred[0] if q_pred.ndim == 2 else q_pred
                depth_scale = depth_scale[0].item() if hasattr(depth_scale, 'shape') else float(depth_scale)
                q_pred = q_pred / torch.norm(q_pred)
                R_pred = quaternion_to_matrix(q_pred).squeeze(0)

                # 4. Translation (pinhole + depth scale)
                obj_diameter = diameters.get(linemod_id, 0.1)
                f_x, f_y = K[0, 0], K[1, 1]
                c_x, c_y = K[0, 2], K[1, 2]
                pinhole_X, pinhole_Y, pinhole_Z = pinhole_translation(bbox, f_x, f_y, c_x, c_y, obj_diameter, depth_scale)

                t_pred_np = np.array([pinhole_X, pinhole_Y, pinhole_Z])
                t_pred_tensor = torch.from_numpy(t_pred_np).float().to(device).view(3, 1)

                # 5. Metrics
                if linemod_id not in meshes:
                    continue
                    
                model_pts = meshes[linemod_id].to(device)
                
                if linemod_id in SYMMETRIC_IDS:
                    err = compute_add_s(R_pred, t_pred_tensor, gt_R, gt_t, model_pts)
                    metrics["add_s"].append(err)
                else:
                    err = compute_add(R_pred, t_pred_tensor, gt_R, gt_t, model_pts)
                    metrics["add"].append(err)
                
                if err < (0.1 * obj_diameter):
                    pass_count += 1
                total_count += 1

                # Compute separate translation and rotation errors
                t_error = torch.norm(t_pred_tensor - gt_t).item()  # Euclidean distance (mm)
                
                # Rotation error: angular distance between R_pred and gt_R
                R_diff = R_pred @ gt_R.T
                trace = torch.clamp(R_diff.trace(), -1.0, 3.0)
                r_error = torch.acos((trace - 1) / 2).item() * 180 / np.pi  # degrees

                results.append({
                    "file": filename,
                    "obj_id": linemod_id,
                    "R": R_pred.cpu().numpy().tolist(),
                    "t": t_pred_np.flatten().tolist(), 
                    "error": err,
                    "t_error": t_error,
                    "r_error": r_error,
                    "gt_R": gt_R.cpu().numpy().tolist(),
                    "gt_t": gt_t.cpu().numpy().flatten().tolist()
                })

    if total_count > 0:
        print("\n" + "="*30)
        print(f"Total Evaluated: {total_count}")
        print(f"Accuracy (ADD < 0.1d): {100 * pass_count / total_count:.2f}%")
        print("="*30)
    
    return results

def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    diameters = load_diameters(args.models_info)
    
    meshes = load_meshes(args.models_dir)

    dataset = LinemodInferenceDataset(
        yolo_root=args.dataset_root,
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

    print("Loading models...")
    yolo_model = load_yolo(args.yolo_weights)
    if hasattr(yolo_model, "to"):
        yolo_model.to(device)

    resnet_model = load_resnet(args.resnet_weights, device)

    # Correct argument order: diameters -> meshes -> device
    all_poses = run_inference(dataloader, yolo_model, resnet_model, diameters, meshes, device)

    print(f"Done. Processed {len(all_poses)} poses.")
    save_results(all_poses, args.output_file)

if __name__ == "__main__":
    main()