def convert_linemod_to_yolo_detection(
    linemod_root,
    output_root,
    train_split=0.9,
):
    """
    Convert raw LINEMOD dataset into YOLO detection format.

    linemod_root: data/linemod/
    output_root:  data/linemod_yolo/
    """

    img_train = os.path.join(output_root, "images/train")
    img_val = os.path.join(output_root, "images/val")
    lbl_train = os.path.join(output_root, "labels/train")
    lbl_val = os.path.join(output_root, "labels/val")
