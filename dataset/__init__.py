import torch


def yolo_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)

    all_targets = []
    for img_idx, boxes in enumerate(targets):
        if boxes.numel() == 0:
            continue
        img_idx_col = torch.full((boxes.shape[0], 1), img_idx)
        all_targets.append(torch.cat([img_idx_col, boxes], dim=1))

    if all_targets:
        all_targets = torch.cat(all_targets, dim=0)
    else:
        all_targets = torch.empty((0, 6))

    return images, all_targets
