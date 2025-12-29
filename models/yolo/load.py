import torch

def load_yolo(weights_path, device='cuda'):
    """
    Loads the YOLO model. Assumes standard Ultralytics YOLO structure.
    """
    print(f"Loading YOLO weights from {weights_path}...")
    try:
        model = torch.hub.load('ultralytics/yolov11s', 'custom', path=weights_path)
    except Exception:
        # Fallback: Load directly if it's a raw checkpoint
        checkpoint = torch.load(weights_path, map_location=device)
        model = checkpoint['model'] if 'model' in checkpoint else checkpoint

    model.to(device)
    model.eval()
    return model