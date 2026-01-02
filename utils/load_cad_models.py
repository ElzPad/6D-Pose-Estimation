import os
import torch
import trimesh
from tqdm import tqdm


def load_cad_models(models_dir, num_points=2000, num_objects=15):
    """
    Loads point clouds for all LINEMOD objects into a single tensor.

    Args:
        models_dir (str): Path to the directory containing .ply files (e.g., 'data/linemod/models').
        num_points (int): Number of points to sample from each mesh (N).
                          Required to stack them into a single tensor.
        num_objects (int): Total number of objects to load (default 15).
                           It assumes files are named obj_01.ply, obj_02.ply, etc.

    Returns:
        cad_models (torch.Tensor): A tensor of shape (num_objects, num_points, 3) containing the point clouds.
                      The index 0 corresponds to 'obj_01', index 1 to 'obj_02', etc.
    """
    print(f"Loading 3D models from {models_dir}...")

    all_point_clouds = []

    for obj_id in tqdm(range(1, num_objects + 1), desc="Loading Models"):
        filename = f"obj_{obj_id:02d}.ply"
        path = os.path.join(models_dir, filename)

        if not os.path.exists(path):
            print(f"Warning: Model file {filename} not found! Filling with zeros.")
            all_point_clouds.append(torch.zeros((num_points, 3)))
            continue

        mesh = trimesh.load(path)

        # Handle Trimesh Scene object (sometimes loaded instead of Geometry)
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) > 0:
                # taking the first geometry is standard for single objects
                mesh = list(mesh.geometry.values())[0]
            else:
                print(f"Warning: Empty mesh scene for {filename}.")
                all_point_clouds.append(torch.zeros((num_points, 3)))
                continue

        # Sample points uniformly from the surface
        # trimesh.sample.sample_surface returns (points, face_indices)
        points, _ = trimesh.sample.sample_surface(mesh, num_points)

        # Convert to Tensor (N, 3)
        tensor_pts = torch.from_numpy(points).float()
        all_point_clouds.append(tensor_pts)

    # Stack into (num_objects, N, 3)
    final_tensor = torch.stack(all_point_clouds)

    print(f"Models loaded.")
    return final_tensor


# Example usage block
if __name__ == "__main__":

    MODELS_PATH = "data/linemod/Linemod_preprocessed/models"

    if os.path.exists(MODELS_PATH):
        models_tensor = load_cad_models(MODELS_PATH)
    else:
        print(f"Path {MODELS_PATH} does not exist. Please check your structure.")
