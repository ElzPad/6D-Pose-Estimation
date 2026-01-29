import argparse
import os
import yaml
from tqdm import tqdm
import torch
import json
import numpy as np
import trimesh
import cv2
from pathlib import Path
import open3d as o3d

from ultralytics import YOLO
from geometry.pinhole_camera_model import pinhole_translation
from geometry.quaternion_to_rotation_matrix import quaternion_to_matrix

# from dataset.linemod_inference_dataset import LinemodInferenceDataset, collate_fn
from dataset.linemod_inference_dataset_with_depth import LinemodInferenceDataset, collate_fn
from torch.utils.data import DataLoader
from augmentations import get_val_transforms, get_val_translation_transforms, get_val_rgbd_translation_transforms
from models.yolo.load import load_yolo
from models.rgbd.resnetRotationRGBD.load import load_resnet_rotation_rgbd
from models.rgbd.resnetTranslationRGBD.load import load_resnet_translation

from metrics.add import compute_add, compute_add_s

# Mapping: YOLO Class ID (0-12) -> LINEMOD Object ID (1-15)
YOLO_TO_LINEMOD_ID = {
    0: 1,  # ape
    1: 2,  # benchvise
    2: 4,  # camera
    3: 5,  # can
    4: 6,  # cat
    5: 8,  # driller
    6: 9,  # duck
    7: 10,  # eggbox
    8: 11,  # glue
    9: 12,  # holepuncher
    10: 13,  # iron
    11: 14,  # lamp
    12: 15,  # phone
}

LINEMOD_ID_TO_YOLO = {v: k for k, v in YOLO_TO_LINEMOD_ID.items()}

SYMMETRIC_IDS = [10, 11]


def get_args():
    parser = argparse.ArgumentParser(description="6D Pose Estimation Inference Pipeline")

    parser.add_argument("--dataset_root", type=str, default="data/linemod_yolo")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--models_info", type=str, default="data/linemod_yolo/models/models_info.yml")
    parser.add_argument(
        "--models_dir",
        type=str,
        default="data/linemod_yolo/models",
        help="Path to directory containing .ply files",
    )

    parser.add_argument("--yolo_weights", type=str, required=True)
    parser.add_argument("--resnet_rot_weights", type=str, required=True)
    parser.add_argument("--resnet_tra_weights", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_file", type=str, default="results.json")

    # ICP refinement options
    parser.add_argument("--icp_max_iter", type=int, default=50, help="Maximum ICP iterations")
    parser.add_argument("--icp_threshold", type=float, default=5.0, help="ICP distance threshold in mm")

    # Image saving (only affects images, NOT results.json)
    parser.add_argument("--save_images", action="store_true", help="Save images with predicted bbox overlay")
    parser.add_argument("--results_dir", type=str, default="results_images", help="Directory for saved result images")

    return parser.parse_args()


def load_diameters(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"models_info.yml not found at {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return {int(k): v["diameter"] for k, v in data.items()}


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


def xywh_center_to_xyxy(xywh):
    """Ultralytics xywh is center-based: (cx, cy, w, h). Convert to (x1, y1, x2, y2)."""
    cx, cy, w, h = xywh
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return x1, y1, x2, y2


def draw_bbox_rgb(img_rgb, bbox_xywh, label=None, thickness=2):
    """
    Draw bbox on an RGB image (uint8). Optionally draws label in top-left with a background.
    Returns RGB.
    """
    out = img_rgb.copy()
    x1, y1, x2, y2 = xywh_center_to_xyxy(bbox_xywh)
    h, w = out.shape[:2]

    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y2 = int(np.clip(y2, 0, h - 1))

    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.rectangle(out_bgr, (x1, y1), (x2, y2), (0, 255, 0), thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        org = (10, 30)

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        pad = 6
        x_bg1 = max(org[0] - pad, 0)
        y_bg1 = max(org[1] - text_h - pad, 0)
        x_bg2 = min(org[0] + text_w + pad, w - 1)
        y_bg2 = min(org[1] + baseline + pad, h - 1)

        cv2.rectangle(out_bgr, (x_bg1, y_bg1), (x_bg2, y_bg2), (0, 0, 0), -1)
        cv2.putText(out_bgr, label, org, font, font_scale, (0, 255, 0), font_thickness, lineType=cv2.LINE_AA)

    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def project_points(K, pts_cam):
    """
    Project Nx3 camera-frame points into pixels using intrinsics K (3x3).
    pts_cam: (N,3) numpy
    Returns (N,2) pixel coords, plus a mask of valid Z>0.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    X = pts_cam[:, 0]
    Y = pts_cam[:, 1]
    Z = pts_cam[:, 2]

    valid = Z > 1e-6
    u = np.zeros_like(X, dtype=np.float32)
    v = np.zeros_like(Y, dtype=np.float32)
    u[valid] = fx * (X[valid] / Z[valid]) + cx
    v[valid] = fy * (Y[valid] / Z[valid]) + cy

    return np.stack([u, v], axis=1), valid


def draw_pose_axes_rgb(img_rgb, K, R_pred, t_pred, axis_len, thickness=1):
    """
    Draw a 3D coordinate frame (X,Y,Z axes) projected into the image.

    - K: (3,3) numpy intrinsics
    - R_pred: (3,3) torch or numpy
    - t_pred: (3,1) torch or numpy, in same units as axis_len
    - axis_len: float, axis length in object units (e.g., mm)
    """
    if isinstance(R_pred, torch.Tensor):
        R = R_pred.detach().cpu().numpy()
    else:
        R = np.asarray(R_pred)

    if isinstance(t_pred, torch.Tensor):
        t = t_pred.detach().cpu().numpy().reshape(3, 1)
    else:
        t = np.asarray(t_pred).reshape(3, 1)

    # Object-frame points: origin + endpoints
    pts_obj = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_len, 0.0, 0.0],  # X
            [0.0, axis_len, 0.0],  # Y
            [0.0, 0.0, axis_len],  # Z
        ],
        dtype=np.float32,
    )

    # Transform into camera frame: P_cam = R * P_obj + t
    pts_cam = (R @ pts_obj.T + t).T  # (4,3)

    pix, valid = project_points(K, pts_cam)
    if not valid[0]:
        # origin behind camera => nothing reliable to draw
        return img_rgb

    o = pix[0].astype(int)

    out_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    def draw_line_if_valid(idx, color):
        if valid[idx]:
            p = pix[idx].astype(int)
            cv2.line(out_bgr, tuple(o), tuple(p), color, thickness, lineType=cv2.LINE_AA)

    # Conventional axis colors in BGR:
    # X = Red, Y = Green, Z = Blue (in BGR that’s (0,0,255), (0,255,0), (255,0,0))
    draw_line_if_valid(1, (0, 0, 255))  # X
    draw_line_if_valid(2, (0, 255, 0))  # Y
    draw_line_if_valid(3, (255, 0, 0))  # Z

    # draw origin point
    cv2.circle(out_bgr, tuple(o), 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def rotation_error_degrees(R_pred: torch.Tensor, R_gt: torch.Tensor) -> float:
    """
    Angular distance between two rotation matrices in degrees.
    """
    R_diff = R_pred @ R_gt.T
    tr = torch.clamp(torch.trace(R_diff), -1.0, 3.0)
    cos_theta = torch.clamp((tr - 1.0) / 2.0, -1.0, 1.0)
    return float(torch.acos(cos_theta).item() * 180.0 / np.pi)


def crop_resize_rgbd_infer(
    image_raw, depth_raw, bbox, image_size=224, depth_unit_scale=1000.0, depth_clip_m=2.0, bbox_pad=1.2
):
    """Crop and resize RGB and depth to padded bbox, normalize as in training."""
    img_h, img_w = image_raw.shape[:2]
    cx, cy, bw, bh = bbox
    # Apply padding as in training
    bw = bw * bbox_pad
    bh = bh * bbox_pad
    x1 = int(round(cx - bw / 2.0))
    y1 = int(round(cy - bh / 2.0))
    x2 = int(round(cx + bw / 2.0))
    y2 = int(round(cy + bh / 2.0))
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(1, min(x2, img_w))
    y2 = max(1, min(y2, img_h))
    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = 0, 0, img_w, img_h
    rgb = image_raw[y1:y2, x1:x2]
    depth = depth_raw[y1:y2, x1:x2]
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(depth, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    rgb = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    depth = depth.astype(np.float32)
    if depth_unit_scale > 0:
        depth = depth / depth_unit_scale
    valid = (depth > 0).astype(np.float32)
    depth = np.clip(depth, 0.0, depth_clip_m)
    depth_norm = (depth / depth_clip_m) * valid
    depth_tensor = torch.from_numpy(depth_norm).unsqueeze(0).contiguous()
    return rgb_tensor, depth_tensor


def depth_to_pointcloud(depth_img, K, bbox=None, depth_scale=1.0):
    """
    Back-project depth image to 3D point cloud in camera frame.

    Args:
        depth_img: (H, W) uint16 depth in mm
        K: (3, 3) camera intrinsics
        bbox: Optional (cx, cy, w, h) to extract points only inside bbox
        depth_scale: Scale factor (1.0 = mm, 0.001 = m)

    Returns:
        points: (N, 3) numpy array of 3D points in camera frame
    """
    if isinstance(K, torch.Tensor):
        K = K.numpy()

    # Fix: squeeze singleton channel if present
    if depth_img.ndim == 3 and depth_img.shape[2] == 1:
        depth_img = depth_img[:, :, 0]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    H, W = depth_img.shape

    # Create mask for valid depth
    valid_mask = depth_img > 0

    # If bbox provided, mask to only use points inside bbox
    if bbox is not None:
        bcx, bcy, bw, bh = bbox
        x1 = int(max(0, bcx - bw / 2))
        y1 = int(max(0, bcy - bh / 2))
        x2 = int(min(W, bcx + bw / 2))
        y2 = int(min(H, bcy + bh / 2))

        bbox_mask = np.zeros((H, W), dtype=bool)
        bbox_mask[y1:y2, x1:x2] = True
        valid_mask = valid_mask & bbox_mask

    # Get valid pixel coordinates
    v, u = np.where(valid_mask)
    z = depth_img[v, u].astype(np.float32) * depth_scale

    # Back-project to 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=1)
    return points


def refine_pose_icp(R_pred, t_pred, depth_img, K, model_pts, bbox=None, max_iterations=50, threshold=5.0):
    """
    Refine predicted pose using ICP between observed depth and CAD model.

    Args:
        R_pred: (3, 3) predicted rotation matrix (torch or numpy)
        t_pred: (3, 1) predicted translation (torch or numpy), in mm
        depth_img: (H, W) uint16 depth image in mm
        K: (3, 3) camera intrinsics
        model_pts: (N, 3) CAD model vertices in mm
        bbox: (cx, cy, w, h) bounding box to extract observed points
        max_iterations: Maximum ICP iterations
        threshold: Distance threshold in mm for ICP correspondences

    Returns:
        R_refined: (3, 3) numpy rotation matrix
        t_refined: (3, 1) numpy translation vector
        success: bool indicating if ICP converged
    """

    # Convert inputs to numpy
    if isinstance(R_pred, torch.Tensor):
        R_pred = R_pred.cpu().numpy()
    if isinstance(t_pred, torch.Tensor):
        t_pred = t_pred.cpu().numpy().reshape(3, 1)
    if isinstance(model_pts, torch.Tensor):
        model_pts = model_pts.cpu().numpy()

    # Extract observed point cloud from depth
    observed_pts = depth_to_pointcloud(depth_img, K, bbox=bbox, depth_scale=1.0)  # mm

    if len(observed_pts) < 50:
        # Not enough points for reliable ICP
        return R_pred, t_pred, False

    # Transform model points to camera frame using initial pose
    model_pts_cam = (R_pred @ model_pts.T + t_pred).T  # (N, 3)

    # Create Open3D point clouds
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(model_pts_cam)

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(observed_pts)

    # Run ICP (point-to-point)
    # Initial transformation is identity since we already transformed model to camera frame
    init_transform = np.eye(4)

    reg_result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        threshold,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
    )

    # Extract refined transformation
    T_refine = reg_result.transformation
    R_refine = T_refine[:3, :3]
    t_refine = T_refine[:3, 3:4]

    # Compose with initial pose: R_final = R_refine @ R_pred, t_final = R_refine @ t_pred + t_refine
    R_refined = R_refine @ R_pred
    t_refined = R_refine @ t_pred + t_refine

    # Check if ICP converged reasonably
    success = reg_result.fitness > 0.3  # At least 30% inlier correspondences

    return R_refined, t_refined, success


def run_inference(
    dataloader,
    yolo_model,
    resnet_rotation_model,
    resnet_translation_rgbd_model,
    diameters,
    meshes,
    meshes_numpy,
    device,
    save_images=False,
    results_dir="results_images",
    icp_max_iter=50,
    icp_threshold=5.0,
):
    transformer = get_val_transforms(image_size=224)
    rgbd_transformer = get_val_rgbd_translation_transforms(image_size=224)

    results = []
    pass_count = 0
    total_count = 0
    icp_success_count = 0

    if save_images:
        os.makedirs(results_dir, exist_ok=True)

    use_object_id = getattr(resnet_rotation_model, "use_object_id", False)
    if use_object_id:
        print("ResNet using object identity conditioning")

    yolo_model.eval()
    resnet_rotation_model.eval()
    resnet_translation_rgbd_model.eval()
    print(f"Running inference on {len(dataloader.dataset)} images...")

    for batch in tqdm(dataloader):
        batch_images, batch_depths, batch_targets = batch
        target = batch_targets[0]
        filename = target["image_id"]

        K = target["intrinsics"].numpy()
        gt_id = int(target["gt_class_id"])
        gt_R = target["gt_R"].to(device)
        gt_t = target["gt_t"].to(device)

        # 1) YOLO stage
        images_tensor = torch.stack(batch_images).to(device)
        yolo_result = yolo_model(images_tensor, verbose=False)[0]

        detections = yolo_result.boxes.xywh.cpu().numpy()
        classes = yolo_result.boxes.cls.cpu().numpy()
        confs = yolo_result.boxes.conf.cpu().numpy()

        if len(detections) == 0:
            continue

        yolo_class = LINEMOD_ID_TO_YOLO.get(gt_id, None)
        if yolo_class is None:
            continue

        cls_indices = np.where(classes == yolo_class)[0]
        if len(cls_indices) == 0:
            continue

        best_idx = cls_indices[np.argmax(confs[cls_indices])]
        pred_bbox = detections[best_idx]
        pred_conf = float(confs[best_idx])

        img_tensor = batch_images[0]
        depth_tensor = batch_depths[0]
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # RGB
        depth_np = depth_tensor.numpy()
        # Fix: squeeze singleton channel if present (could be (1, H, W) or (H, W, 1))
        if depth_np.ndim == 3:
            if depth_np.shape[0] == 1:
                depth_np = depth_np[0]
            elif depth_np.shape[2] == 1:
                depth_np = depth_np[:, :, 0]

        # For rotation: crop and preprocess as in training
        rgb_patch, depth_patch = crop_resize_rgbd_infer(
            img_np, depth_np, pred_bbox, image_size=224, depth_unit_scale=1000.0, depth_clip_m=2.0, bbox_pad=1.2
        )
        rgb_patch = rgb_patch.unsqueeze(0).to(device)
        depth_patch = depth_patch.unsqueeze(0).to(device)
        with torch.no_grad():
            if use_object_id:
                object_ids = torch.tensor([gt_id], device=device)
                q_pred = resnet_rotation_model(rgb_patch, depth_patch, object_ids)
            else:
                q_pred = resnet_rotation_model(rgb_patch, depth_patch, None)

        q_pred = q_pred[0] if q_pred.ndim == 2 else q_pred
        q_pred = q_pred / torch.norm(q_pred)
        R_pred = quaternion_to_matrix(q_pred).squeeze(0)

        # 3) Translation (RGBD)
        # Recreate the full-image RGBD tensor for translation
        rgbd_tensor = rgbd_transformer(img_np, depth_np).to(device)
        obj_diameter = diameters.get(gt_id, 0.1)
        img_h, img_w = img_np.shape[:2]
        cx, cy, bw, bh = pred_bbox
        pred_bbox_norm = np.array([cx / img_w, cy / img_h, bw / img_w, bh / img_h], dtype=np.float32)
        bbox_tensor = torch.tensor(pred_bbox_norm, dtype=torch.float32, device=device).unsqueeze(0)
        diameter_tensor = torch.tensor([obj_diameter], dtype=torch.float32, device=device)
        with torch.no_grad():
            t_pred = resnet_translation_rgbd_model(
                rgbd_tensor.unsqueeze(0), torch.tensor([gt_id], device=device), bbox_tensor, diameter_tensor
            )
        t_pred_np = t_pred[0].cpu().numpy() * 1000
        t_pred_tensor = t_pred[0].view(3, 1).to(device) * 1000

        # 4) ICP Refinement
        icp_applied = False
        if gt_id in meshes_numpy:
            model_pts_np = meshes_numpy[gt_id]
            R_refined, t_refined, icp_success = refine_pose_icp(
                R_pred,
                t_pred_tensor,
                depth_np,
                target["intrinsics"],
                model_pts_np,
                bbox=pred_bbox,
                max_iterations=icp_max_iter,
                threshold=icp_threshold,
            )
            if icp_success:
                R_pred = torch.from_numpy(R_refined.astype(np.float32)).to(device)
                t_pred_tensor = torch.from_numpy(t_refined.astype(np.float32)).to(device)
                t_pred_np = t_refined.flatten()
                icp_success_count += 1
                icp_applied = True

        # --- Save visualization image (bbox + 3D axes) ---
        if save_images:
            # bbox label (optional)
            angle_deg = rotation_error_degrees(R_pred, gt_R)
            label = f"obj={gt_id} yolo_cls={yolo_class} conf={pred_conf:.3f} ang_err={angle_deg:.2f}deg"

            vis = draw_bbox_rgb(img_np, pred_bbox, label=label)

            # axis length: pick something visible but not huge (e.g., 0.5 * diameter)
            axis_len = float(0.5 * obj_diameter)

            # draw predicted axes using predicted pose
            vis = draw_pose_axes_rgb(
                vis,
                K=K,
                R_pred=R_pred,
                t_pred=t_pred_tensor,
                axis_len=axis_len,
                thickness=1,
            )

            stem = Path(str(filename)).stem
            out_path = os.path.join(results_dir, f"{stem}.png")
            cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        # 4) Metrics
        if gt_id not in meshes:
            continue
        model_pts = meshes[gt_id].to(device)

        if gt_id in SYMMETRIC_IDS:
            err = compute_add_s(R_pred, t_pred_tensor, gt_R, gt_t, model_pts)
        else:
            err = compute_add(R_pred, t_pred_tensor, gt_R, gt_t, model_pts)

        if err < (0.1 * obj_diameter):
            pass_count += 1
        total_count += 1

        # Compute separate translation and rotation errors (unchanged from your original)
        t_error = torch.norm(t_pred_tensor - gt_t).item()

        R_diff = R_pred @ gt_R.T
        trace = torch.clamp(R_diff.trace(), -1.0, 3.0)
        r_error = torch.acos((trace - 1) / 2).item() * 180 / np.pi

        results.append(
            {
                "file": filename,
                "obj_id": gt_id,
                "R": R_pred.cpu().numpy().tolist(),
                "t": t_pred_np.flatten().tolist(),
                "error": err,
                "t_error": t_error,
                "r_error": r_error,
                "gt_R": gt_R.cpu().numpy().tolist(),
                "gt_t": gt_t.cpu().numpy().flatten().tolist(),
            }
        )

    if total_count > 0:
        print("\n" + "=" * 30)
        print(f"Total Evaluated: {total_count}")
        print(f"Accuracy (ADD < 0.1d): {100 * pass_count / total_count:.2f}%")
        print(f"ICP successful refinements: {icp_success_count}/{total_count}")
        print("=" * 30)

    return results


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    diameters = load_diameters(args.models_info)
    meshes = load_meshes(args.models_dir)

    # Also create numpy version of meshes for ICP
    meshes_numpy = {k: v.numpy() for k, v in meshes.items()}

    dataset = LinemodInferenceDataset(
        yolo_root=args.dataset_root,
        split=args.split,
        img_ext=".png",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    print("Loading models...")
    yolo_model = load_yolo(args.yolo_weights, device=device)
    if hasattr(yolo_model, "to"):
        yolo_model.to(device)

    resnet_rotation_model = load_resnet_rotation_rgbd(args.resnet_rot_weights, device)
    resnet_translation_rgbd_model = load_resnet_translation(args.resnet_tra_weights, device)

    all_poses = run_inference(
        dataloader,
        yolo_model,
        resnet_rotation_model,
        resnet_translation_rgbd_model,
        diameters,
        meshes,
        meshes_numpy,
        device,
        save_images=args.save_images,
        results_dir=args.results_dir,
        icp_max_iter=args.icp_max_iter,
        icp_threshold=args.icp_threshold,
    )

    print(f"Done. Processed {len(all_poses)} poses.")
    save_results(all_poses, args.output_file)


if __name__ == "__main__":
    main()
