import torch
from .model import ResNetRotation, ResNetRotationWithObjectID, NUM_OBJECTS

def load_resnet_rotation(weights_path, device='cpu', freeze_backbone=False, use_object_id=False):
    """
    Loads the custom ResNetRotation model trained with your script.
    
    Args:
        weights_path: Path to the checkpoint file
        device: Device to load the model on
        freeze_backbone: Whether to freeze backbone weights
        use_object_id: If True, loads ResNetRotationWithObjectID for multi-object training
    
    Returns:
        Loaded model in eval mode
    """
    print(f"Loading custom ResNetRotation from {weights_path}...")
    
    # Check checkpoint to auto-detect model type
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    # Auto-detect from checkpoint if use_object_id was saved
    if isinstance(checkpoint, dict) and 'use_object_id' in checkpoint:
        use_object_id = checkpoint['use_object_id']
        print(f"  Auto-detected use_object_id={use_object_id} from checkpoint")
    
    # Create appropriate model
    if use_object_id:
        print("  Using ResNetRotationWithObjectID (object conditioning enabled)")
        model = ResNetRotationWithObjectID(
            num_objects=NUM_OBJECTS, 
            freeze_backbone=freeze_backbone
        )
    else:
        print("  Using ResNetRotation (no object conditioning)")
        model = ResNetRotation(freeze_backbone=freeze_backbone)
    
    # Extract the weights 
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        # Fallback in case a raw state_dict was saved manually
        state_dict = checkpoint
        
    # Load the state dictionary into the model
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Warning: Exact key match failed. Attempting loose load. Error: {e}")
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    
    # Store flag on model for pipeline to use
    model.use_object_id = use_object_id
    
    return model
