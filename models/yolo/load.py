from ultralytics import YOLO

def load_yolo(weights_path, device='cuda'):
    """
    Loads the YOLO model. Assumes standard Ultralytics YOLO structure.
    """
    print(f"Loading YOLO from {weights_path}...")
    model = YOLO(weights_path)

    model.to(device)
    model.eval()
    return model