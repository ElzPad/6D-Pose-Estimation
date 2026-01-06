import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.models import resnet50, ResNet50_Weights

from dataset.linemod_rotation_dataset import LineModRotationDataset

def parse_args():
    p = argparse.ArgumentParser("Fine-tune ResNet50 for LINEMOD rotation (quaternion)")
    p.add_argument("--dataset_path", type=str, default="data/linemod/Linemod_preprocessed/data",
                   help="Path to LINEMOD preprocessed root (contains data/01, data/02 ... OR object folders).")
    p.add_argument("--object_id", type=int, default=0, help="LINEMOD object id (e.g., 0=all, 1=ape, 5=can, etc.)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--split_percentage", type=float, default=0.8,
                   help="Fraction used for train in your dataset implementation.")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="runs_resnet")
    p.add_argument("--run_name", type=str, default="resnet50_rot")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="If set, freeze all ResNet layers except the final FC head.")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


class ResNetRotation(nn.Module):
    """
    ResNet-50 backbone (ImageNet) + quaternion head (4 dims).
    Outputs raw quaternion; we normalize in forward().
    """
    def __init__(self, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 4)

        if freeze_backbone:
            for name, p in self.backbone.named_parameters():
                # keep fc trainable
                if not name.startswith("fc."):
                    p.requires_grad = False

    def forward(self, x):
        q = self.backbone(x)                    # (B,4)
        q = F.normalize(q, p=2, dim=1)          # unit quaternion
        return q

def quat_l2_sign_invariant(q_pred, q_gt):
    """
    Sign-invariant quaternion loss: q and -q represent the same rotation.
    q_pred, q_gt: (B,4), already normalized (or close).
    """
    # both are (B,4)
    loss1 = (q_pred - q_gt).pow(2).sum(dim=1)
    loss2 = (q_pred + q_gt).pow(2).sum(dim=1)
    return torch.min(loss1, loss2).mean()


@torch.no_grad()
def quat_angle_error_deg(q_pred, q_gt):
    """
    Computes angular distance between quaternions in degrees (sign invariant).
    """
    # Ensure unit
    q_pred = F.normalize(q_pred, p=2, dim=1)
    q_gt = F.normalize(q_gt, p=2, dim=1)

    # |dot| in [0,1]
    dot = torch.abs(torch.sum(q_pred * q_gt, dim=1)).clamp(0.0, 1.0)
    # angle = 2*acos(dot)
    ang = 2.0 * torch.acos(dot)
    return (ang * 180.0 / torch.pi).mean().item()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_ang = 0.0
    n = 0

    for imgs, q_gt in loader:
        imgs = imgs.to(device, non_blocking=True).float()
        q_gt = q_gt.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        q_pred = model(imgs)
        loss = quat_l2_sign_invariant(q_pred, q_gt)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_ang += quat_angle_error_deg(q_pred, q_gt) * bs
        n += bs

    return total_loss / max(n, 1), total_ang / max(n, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_ang = 0.0
    n = 0

    for imgs, q_gt in loader:
        imgs = imgs.to(device, non_blocking=True).float()
        q_gt = q_gt.to(device, non_blocking=True).float()

        q_pred = model(imgs)
        loss = quat_l2_sign_invariant(q_pred, q_gt)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_ang += quat_angle_error_deg(q_pred, q_gt) * bs
        n += bs

    return total_loss / max(n, 1), total_ang / max(n, 1)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print("Device:", device)

    train_ds = LineModRotationDataset(
        root_dir=args.dataset_path,
        object_id=args.object_id if args.object_id!=0 else None,
        split="train",
        split_percentage=args.split_percentage
    )
    val_ds = LineModRotationDataset(
        root_dir=args.dataset_path,
        object_id=args.object_id if args.object_id!=0 else None,
        split="test",
        split_percentage=args.split_percentage
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )

    model = ResNetRotation(freeze_backbone=args.freeze_backbone).to(device)

    # Only optimize trainable params (important if freezing)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    save_dir = Path(args.save_dir) / args.run_name / f"obj_{args.object_id:02d}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best_resnet50_quat.pth"
    last_path = save_dir / "last_resnet50_quat.pth"

    best_val_ang = float("inf")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ang = train_one_epoch(model, train_loader, optimizer, device)
        va_loss, va_ang = eval_one_epoch(model, val_loader, device)

        scheduler.step(va_ang)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"[{epoch:03d}/{args.epochs}] lr={current_lr:.1e} | "
              f"train loss={tr_loss:.4f} ang={tr_ang:.2f}° | "
              f"val loss={va_loss:.4f} ang={va_ang:.2f}°")

        # Save last
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "object_id": args.object_id,
            "args": vars(args),
        }, last_path)

        # Save best (by val angular error)
        if va_ang < best_val_ang:
            best_val_ang = va_ang
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_ang_deg": best_val_ang,
                "object_id": args.object_id,
                "args": vars(args),
            }, best_path)
            print(f"  ✓ New best saved: {best_path} (val ang {best_val_ang:.2f}°)")

    print("Done.")

if __name__ == "__main__":
    main()
