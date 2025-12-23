import math

def pinhole_translation(bbox, f_x, f_y, c_x, c_y, diameter):
    """
        Compute translation from bbox using pinhole camera model

        Parameters:
            - bbox = [x_center, y_center, width, height]
            - f_x, f_y = focal length
            - c_x, c_y = optical center coordinates
            - diameter = diameter
    """
    x_center, y_center, w, h = bbox
    bbox_diag = math.sqrt(w**2 + h**2)
    Z = f_x * diameter / bbox_diag

    X = (x_center - c_x) / f_x * Z
    Y = (y_center - c_y) / f_y * Z

    return (X, Y, Z)