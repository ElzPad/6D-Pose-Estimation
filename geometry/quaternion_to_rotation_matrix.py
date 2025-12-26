import torch

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: float tensor of shape (..., 4). 
                     The convention is [w, x, y, z] (real part first).
                     They are expected to be normalized (unit quaternions).

    Returns:
        float tensor of shape (..., 3, 3).
    """
    
    # Unpack the components
    #                       =====>      q = w + xi + yj + zk        <=====  
    # -- COMPONENTS:
    #       r = corresponts to w and stands for REAL
    #       i = corresponds to x (first imaginary component)
    #       j = corresponds to y (second imaginary component)
    #       k = corresponds to z (third imaginary component)

    r, i, j, k = torch.unbind(quaternions, -1)  
    
    # Compute common terms
    two_s = 2.0  # Assumes unit quaternion. If not unit, use 2.0 / dot(q, q)
    
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    
    # Reshape to (..., 3, 3)
    return o.view(quaternions.shape[:-1] + (3, 3))