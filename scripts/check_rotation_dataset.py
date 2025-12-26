import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from dataset import LineModRotationDataset


def parse_args():
    """
    Parses command-line arguments for dataset verification.
    """
    parser = argparse.ArgumentParser(
        description="Verify LineModRotationDataset implementation."
    )

    # Dataset Parameters
    parser.add_argument(
        "--root_dir",
        type=str,
        default="data/linemod/Linemod_preprocessed/data",
        help="Path to the preprocessed LINEMOD dataset root.",
    )
    parser.add_argument(
        "--obj_id",
        type=str,
        default="05",
        help="Object ID to verify (e.g., '05' or '5').",
    )

    # Visualization Parameters
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for the DataLoader."
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=4,
        help="Number of images to display in the plot.",
    )
    parser.add_argument(
        "--split_percentage",
        type=float,
        default=0.15,
        help="Percentage of data used for training split.",
    )

    return parser.parse_args()


def visualize_batch(dataloader, num_images=4):
    """
    Fetches a batch and displays images to verify:
    1. Cropping is correct (object is centered).
    2. Augmentation is applied (if using 'train' split).
    3. Tensors are normalized correctly.
    """
    try:
        # Get a batch
        images, quats = next(iter(dataloader))
    except StopIteration:
        print("Error: Dataloader is empty.")
        return

    print(f"\n--- Batch Info ---")
    print(f"Image Tensor Shape: {images.shape}")  # Should be (Batch, 3, 224, 224)
    print(f"Label Tensor Shape: {quats.shape}")  # Should be (Batch, 4)

    # Setup Plot
    # Ensure we don't try to plot more images than exist in the batch
    plot_count = min(num_images, images.shape[0])
    fig, axes = plt.subplots(1, plot_count, figsize=(15, 5))

    # If only 1 image, axes is not a list, wrap it
    if plot_count == 1:
        axes = [axes]

    # ImageNet Mean/Std for denormalization (to make images viewable)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(plot_count):
        # Convert Tensor to Numpy: (C, H, W) -> (H, W, C)
        img = images[i].permute(1, 2, 0).numpy()

        # Denormalize
        img = std * img + mean
        img = np.clip(img, 0, 1)

        q = quats[i].numpy()

        axes[i].imshow(img)
        axes[i].set_title(f"Quat: [{q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f}]")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def verify_quaternions(dataloader):
    """
    Checks if quaternion labels are valid (unit norm).
    """
    try:
        _, quats = next(iter(dataloader))
    except StopIteration:
        return

    norms = torch.norm(quats, dim=1)
    print(f"\n--- Label Verification ---")
    print(f"Quaternion norms (should be approx 1.0): {norms[:4]}")

    if torch.allclose(norms, torch.ones_like(norms), atol=1e-4):
        print("SUCCESS: All quaternions are unit length.")
    else:
        print("WARNING: Quaternions are not normalized!")


# $> python -m scripts.check_rotation_dataset --obj_id 05
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Test Training Set (Expect Augmentations)
    print(f"Initializing Training Dataset for Object {args.obj_id}...")
    try:
        train_ds = LineModRotationDataset(
            root_dir=args.root_dir,
            object_id=args.obj_id,
            split="train",
            split_percentage=args.split_percentage,
        )

        if len(train_ds) > 0:
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True
            )
            visualize_batch(train_loader, num_images=args.num_images)
            verify_quaternions(train_loader)
        else:
            print(f"Warning: Training dataset is empty for object {args.obj_id}.")

        # Test Validation Set (Expect Clean, Centered Images)
        print(f"\nInitializing Validation Dataset for Object {args.obj_id}...")
        val_ds = LineModRotationDataset(
            root_dir=args.root_dir,
            object_id=args.obj_id,
            split="test",
            split_percentage=args.split_percentage,
        )

        if len(val_ds) > 0:
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True)
            visualize_batch(val_loader, num_images=args.num_images)
        else:
            print(f"Warning: Validation dataset is empty for object {args.obj_id}.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please check if the root directory '{args.root_dir}' is correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
