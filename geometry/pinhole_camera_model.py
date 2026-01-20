import math

def pinhole_translation(bbox, f_x, f_y, c_x, c_y, diameter, precomputed_depth = None):
    """
        Compute translation from bbox using pinhole camera model

        Args:
            - bbox = [x_center, y_center, width, height]
            - f_x, f_y = focal length
            - c_x, c_y = optical center coordinates
            - diameter = diameter

        Returns:
            - translation triplet (X, Y, Z)
    """
    x_center, y_center, w, h = bbox
    bbox_diag = math.sqrt(w**2 + h**2)
    Z = f_x * diameter / bbox_diag if not precomputed_depth else precomputed_depth

    X = (x_center - c_x) / f_x * Z
    Y = (y_center - c_y) / f_y * Z

    return (X, Y, Z)