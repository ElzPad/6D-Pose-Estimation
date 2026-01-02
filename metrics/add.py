import torch


def compute_add(
    pred_R: torch.Tensor,
    pred_t: torch.Tensor,
    gt_R: torch.Tensor,
    gt_t: torch.Tensor,
    model_points: torch.Tensor,
) -> float:
    """
    Computes the standard ADD metric (Non-Symmetric).
    Used for asymmetric objects.
    Args:
        pred_R: (3, 3) Rotation Matrix
        pred_t: (3,) or (3, 1) Translation Vector (in mm)
        gt_R:   (3, 3) Ground Truth Rotation Matrix
        gt_t:   (3,) or (3, 1) Ground Truth Translation Vector (in mm)
        model_points: (N, 3) Object point cloud (in mm)

    Returns:
        float: The ADD error.
    """
    device = pred_R.device
    model_points = model_points.to(device)
    pred_t = pred_t.view(3, 1)
    gt_t = gt_t.view(3, 1)

    pts_t = model_points.T

    # Transform
    pred_pts = torch.matmul(pred_R, pts_t) + pred_t
    gt_pts = torch.matmul(gt_R, pts_t) + gt_t

    # Standard Euclidean Distance (Point-to-Point)
    # (3, N) -> Norm over dim 0 -> (N,)
    distances = torch.norm(pred_pts - gt_pts, dim=0)

    return torch.mean(distances).item()


def compute_batch_add(
    pred_R: torch.Tensor,
    pred_t: torch.Tensor,
    gt_R: torch.Tensor,
    gt_t: torch.Tensor,
    model_points: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the ADD metric for a batch of items.

    Args:
        pred_R: (B, 3, 3) Predicted Rotation
        pred_t: (B, 3) or (B, 3, 1) Predicted Translation
        gt_R:   (B, 3, 3) Ground Truth Rotation
        gt_t:   (B, 3) or (B, 3, 1) Ground Truth Translation
        model_points: (B, N, 3) Batch of distinct object point clouds.

    Returns:
        torch.Tensor: (B,) ADD error per sample.
    """
    B = pred_R.shape[0]
    device = pred_R.device
    model_points = model_points.to(device)

    pred_t = pred_t.view(B, 3, 1)
    gt_t = gt_t.view(B, 3, 1)

    # Prepare Points: (B, N, 3) -> (B, 3, N) for matrix mult
    pts = model_points.permute(0, 2, 1)

    # (B, 3, 3) @ (B, 3, N) -> (B, 3, N)
    pred_pts = torch.bmm(pred_R, pts) + pred_t
    gt_pts = torch.bmm(gt_R, pts) + gt_t

    # Euclidean Distance (Point-to-Point)
    # Difference: (B, 3, N)
    diff = pred_pts - gt_pts

    # Norm over dim 1 (x,y,z): (B, N)
    distances = torch.norm(diff, dim=1)

    # Average over points (dim 1): (B,)
    return torch.mean(distances, dim=1)


def compute_add_s(
    pred_R: torch.Tensor,
    pred_t: torch.Tensor,
    gt_R: torch.Tensor,
    gt_t: torch.Tensor,
    model_points: torch.Tensor,
) -> float:
    """
    Computes the ADD-S metric (Symmetric) .
    Standard metric for symmetric objects (e.g., Eggbox, Glue).
    Args:
        pred_R: (3, 3) Rotation Matrix
        pred_t: (3,) or (3, 1) Translation Vector (in mm)
        gt_R:   (3, 3) Ground Truth Rotation Matrix
        gt_t:   (3,) or (3, 1) Ground Truth Translation Vector (in mm)
        model_points: (N, 3) Object point cloud (in mm)

    Returns:
        float: The ADD-S error.
    """
    device = pred_R.device
    model_points = model_points.to(device)

    pred_t = pred_t.view(3, 1)
    gt_t = gt_t.view(3, 1)

    pts_t = model_points.T

    pred_pts = torch.matmul(pred_R, pts_t) + pred_t
    gt_pts = torch.matmul(gt_R, pts_t) + gt_t

    pred_pts = pred_pts.T
    gt_pts = gt_pts.T

    # Matrix Shape: (N, N)
    dist_matrix = torch.cdist(gt_pts, pred_pts, p=2)

    # For each GT point (row), find the minimum distance to ANY Predicted point (col)
    min_dists, _ = torch.min(dist_matrix, dim=1)

    # Average the distances
    add_s_error = torch.mean(min_dists)

    return add_s_error.item()


def compute_batch_add_s(
    pred_R: torch.Tensor,
    pred_t: torch.Tensor,
    gt_R: torch.Tensor,
    gt_t: torch.Tensor,
    model_points: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the ADD-S metric for a batch where each item has its own 3D model.

    Args:
        pred_R: (B, 3, 3) Predicted Rotation
        pred_t: (B, 3) or (B, 3, 1) Predicted Translation
        gt_R:   (B, 3, 3) Ground Truth Rotation
        gt_t:   (B, 3) or (B, 3, 1) Ground Truth Translation
        model_points: (B, N, 3) Batch of distinct object point clouds.


    Returns:
        torch.Tensor: (B,) ADD-S error per sample.
    """
    B = pred_R.shape[0]
    device = pred_R.device
    model_points = model_points.to(device)

    pred_t = pred_t.view(B, 3, 1)
    gt_t = gt_t.view(B, 3, 1)

    # (B, N, 3) -> (B, 3, N)
    pts = model_points.permute(0, 2, 1)

    # Output: (B, 3, N)
    pred_pts = torch.bmm(pred_R, pts) + pred_t
    gt_pts = torch.bmm(gt_R, pts) + gt_t

    # Prepare for CDIST
    # torch.cdist expects (B, N, 3) ("Batch", "Row", "Vector")
    # So we permute back to (B, N, 3)
    pred_pts_cdist = pred_pts.permute(0, 2, 1)
    gt_pts_cdist = gt_pts.permute(0, 2, 1)

    # Compute Pairwise Distance Matrix
    # Shape: (B, N, N)
    dist_matrix = torch.cdist(gt_pts_cdist, pred_pts_cdist, p=2)

    # Nearest Neighbor Search
    # For each GT point (dim 1), find min distance to ANY Pred point (dim 2)
    # Result: (B, N)
    min_dists, _ = torch.min(dist_matrix, dim=2)

    # Average over points: (B,)
    return torch.mean(min_dists, dim=1)
