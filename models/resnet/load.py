import torch
from .train import ResNetRotation

def load_resnet(weights_path, device='cpu', freeze_backbone=False):
    """
    Loads the custom ResNetRotation model trained with your script.
    """
    print(f"Loading custom ResNetRotation from {weights_path}...")
    
    model = ResNetRotation(freeze_backbone=freeze_backbone)
    
    # 2. Load the checkpoint
    # map_location ensures we can load CUDA-trained weights on CPU if needed
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    
    # 3. Extract the weights 
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        # Fallback in case a raw state_dict was saved manually
        state_dict = checkpoint
        
    # 4. Load the state dictionary into the model
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Warning: Exact key match failed. Attempting loose load. Error: {e}")
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model
