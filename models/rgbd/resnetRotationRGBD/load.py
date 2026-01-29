import torch
from .model import ResNetRotationRGBD, NUM_OBJECTS

def load_resnet_rotation_rgbd(weights_path, device=None):
    """
    Load a ResNetRotationRGBD model with pretrained weights.
    Args:
        weights_path (str): Path to the .pth weights file.
        device (str or torch.device, optional): Device to load the model onto.
    Returns:
        model (ResNetRotationRGBD): Loaded model in eval mode.
    """
    model = ResNetRotationRGBD(num_objects=NUM_OBJECTS, use_object_id=True)
    print("  Using ResNetRotationRGBD (object conditioning enabled)")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    # Extract the weights
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Warning: Exact key match failed. Attempting loose load. Error: {e}")
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model
