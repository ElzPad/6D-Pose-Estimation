import argparse
from pathlib import Path

import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.linemod_translation_dataset import LineModTranslationDataset  # <-- You need to implement this
from .model import ResNetTranslationRGBD

def parse_args():
    p = argparse.ArgumentParser("Fine-tune ResNet50 for LINEMOD translation (depth)")
    p.add_argument("--dataset_path", type=str, default="data/linemod_yolo",
                   help="Path to LINEMOD preprocessed root (contains data/01, data/02 ... OR object folders).")
    p.add_argument("--object_id", type=int, default=0, help="LINEMOD object id (e.g., 0=all, 1=ape, 5=can, etc.)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--split_percentage", type=float, default=0.8,
                   help="Fraction used for train in your dataset implementation.")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="runs_resnet")
    p.add_argument("--run_name", type=str, default="resnet50_tra")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="If set, freeze all ResNet layers except the final FC head.")
    p.add_argument("--unfreeze_epoch", type=int, default=30,
                   help="Epoch at which to unfreeze backbone (only used if freeze_backbone is set).")
    p.add_argument("--use_object_id", action="store_true",
                   help="If set, use object identity conditioning (one-hot) for multi-object training.")
    p.add_argument("--depth_init", type=str, default="avg",
                   choices=["avg", "zero", "green", "scaled_avg"],
                   help="How to initialize depth channel weights: avg (mean of RGB), zero, green (copy green), scaled_avg")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume training from. Loads model weights only (not optimizer).")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


# LINEMOD object IDs (13 objects total)
# These are the valid IDs in the dataset: 1,2,4,5,6,8,9,10,11,12,13,14,15
LINEMOD_OBJECT_IDS = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
NUM_OBJECTS = len(LINEMOD_OBJECT_IDS)

# Create mapping from object_id to one-hot index
OBJECT_ID_TO_INDEX = {obj_id: idx for idx, obj_id in enumerate(LINEMOD_OBJECT_IDS)}

def train_one_epoch(model, loader, optimizer, device, use_object_id=False):
    model.train()
    total_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for imgs, object_ids, bboxes, diameters, targets in pbar:
        imgs = imgs.to(device).float()
        object_ids = object_ids.to(device)
        bboxes = bboxes.to(device).float()
        diameters = diameters.to(device).float()
        targets = targets.to(device).float()  # (B, 3)

        optimizer.zero_grad()
        
        outputs = model(imgs, object_ids, bboxes, diameters)
        diff = outputs - targets
        w = torch.tensor([1.0, 1.0, 5.0], device=diff.device).view(1, 3)  # z x5
        
        loss = (w * diff.pow(2)).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / n

@torch.no_grad()
def eval_one_epoch(model, loader, device, use_object_id=False):
    model.eval()
    total_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for imgs, object_ids, bboxes, diameters, targets in pbar:
            imgs = imgs.to(device).float()
            object_ids = object_ids.to(device)
            bboxes = bboxes.to(device).float()
            diameters = diameters.to(device).float()
            targets = targets.to(device).float()

            outputs = model(imgs, object_ids, bboxes, diameters)
            
            outputs = model(imgs, object_ids, bboxes, diameters)
            diff = outputs - targets
            w = torch.tensor([1.0, 1.0, 5.0], device=diff.device).view(1, 3)  # z x5
            
            loss = (w * diff.pow(2)).mean()

            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
    return total_loss / n


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print("Device:", device)

    train_ds = LineModTranslationDataset(
        root_dir=args.dataset_path,
        object_id=args.object_id if args.object_id != 0 else None,
        split="train",
        split_percentage=args.split_percentage
    )
    val_ds = LineModTranslationDataset(
        root_dir=args.dataset_path,
        object_id=args.object_id if args.object_id != 0 else None,
        split="val",
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

    use_object_id = True
    print("Using ResNetTranslationRGBD (RGBD input, object identity conditioning, bbox, diameter)")
    model = ResNetTranslationRGBD(
        num_objects=NUM_OBJECTS,
        freeze_backbone=args.freeze_backbone,
        depth_init_method=args.depth_init
    ).to(device)

    resumed_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
            resumed_epoch = checkpoint.get('epoch', 0)
            resumed_ang = checkpoint.get('best_val_ang_deg', None)
            print(f"  Loaded weights from epoch {resumed_epoch}" +
                  (f" (best val ang: {resumed_ang:.2f}°)" if resumed_ang else ""))
        else:
            state_dict = checkpoint
            print("  Loaded raw state dict")

        model.load_state_dict(state_dict, strict=False)
        print(f"  Backbone frozen: {args.freeze_backbone}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, threshold=1e-4
    )

    save_dir = Path(args.save_dir) / args.run_name / f"obj_{args.object_id:02d}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best_resnet50_quat.pth"
    last_path = save_dir / "last_resnet50_quat.pth"

    best_val_ang = float("inf")
    backbone_unfrozen = not args.freeze_backbone

    total_epochs = resumed_epoch + args.epochs
    for epoch in range(resumed_epoch + 1, total_epochs + 1):
        if args.freeze_backbone and not backbone_unfrozen and epoch > args.unfreeze_epoch:
            model.unfreeze_backbone()
            backbone_unfrozen = True
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=args.lr * 0.1, weight_decay=args.wd)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
            print(f"[Epoch {epoch}] Optimizer recreated with lr={args.lr * 0.1:.1e} for fine-tuning")
            # Lower batch size to 32 when unfreezing backbone
            train_loader = DataLoader(
                train_ds,
                batch_size=32,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                drop_last=True
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=32,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                drop_last=False
            )

        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        va_loss = eval_one_epoch(model, val_loader, device)

        scheduler.step(va_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"[{epoch:03d}/{total_epochs:03d}] lr={current_lr:.1e} | train loss={tr_loss:.6f} | val loss={va_loss:.6f}")

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "object_id": args.object_id,
            "use_object_id": use_object_id,
            "args": vars(args),
        }, last_path)

        # Save best model based on val loss
        if va_loss < best_val_ang:
            best_val_ang = va_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_ang,
                "object_id": args.object_id,
                "use_object_id": use_object_id,
                "args": vars(args),
            }, best_path)
            print(f"  [OK] New best saved: {best_path} (val loss {best_val_ang:.4f})")

        if epoch % 10 == 0:
            epoch_path = save_dir / f"epoch_{epoch:02d}.pth"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": va_loss,
                "object_id": args.object_id,
                "use_object_id": use_object_id,
                "args": vars(args),
            }, epoch_path)
            print(f"  [OK] Checkpoint saved: {epoch_path}")

    print("Done.")

if __name__ == "__main__":
    main()
