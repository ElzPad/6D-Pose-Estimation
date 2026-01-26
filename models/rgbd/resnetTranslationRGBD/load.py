import torch
from .model import ResNetTranslationRGBD, NUM_OBJECTS

def load_resnet_translation(weights_path, device='cpu'):
    """
    Loads a ResNetTranslationRGBD model from a checkpoint file (supports both full checkpoint and raw state_dict).
    Args:
        weights_path: Path to the checkpoint file
        device: Device to load the model on
    Returns:
        Loaded model in eval mode
    """
    print(f"Loading custom ResNetTranslationRGBD from {weights_path}...")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model = ResNetTranslationRGBD(num_objects=NUM_OBJECTS)
    print("  Using ResNetTranslationRGBD (object conditioning enabled)")
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
    model.to(device)
    model.eval()
    return model
