from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F


def _remove_small_components_np(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask

    h, w = mask.shape
    visited = np.zeros((h, w), dtype=np.bool_)
    out = mask.copy()

    for y in range(h):
        for x in range(w):
            if visited[y, x]:
                continue
            cls = out[y, x]
            visited[y, x] = True

            q: deque[tuple[int, int]] = deque([(y, x)])
            coords: list[tuple[int, int]] = [(y, x)]

            while q:
                cy, cx = q.popleft()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx] or out[ny, nx] != cls:
                        continue
                    visited[ny, nx] = True
                    q.append((ny, nx))
                    coords.append((ny, nx))

            if len(coords) >= min_area:
                continue

            neighbor_votes: dict[int, int] = {}
            for cy, cx in coords:
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    ncls = int(out[ny, nx])
                    if ncls == cls:
                        continue
                    neighbor_votes[ncls] = neighbor_votes.get(ncls, 0) + 1

            if not neighbor_votes:
                continue

            replace_cls = max(neighbor_votes.items(), key=lambda item: item[1])[0]
            for cy, cx in coords:
                out[cy, cx] = replace_cls

    return out


def _majority_filter(pred: torch.Tensor, num_classes: int, kernel_size: int) -> torch.Tensor:
    k = max(int(kernel_size), 1)
    if k == 1:
        return pred
    pad = k // 2
    one_hot = F.one_hot(pred.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    votes = F.avg_pool2d(one_hot, kernel_size=k, stride=1, padding=pad) * (k * k)
    return votes.argmax(dim=1)


def _median_filter(pred: torch.Tensor, kernel_size: int) -> torch.Tensor:
    k = max(int(kernel_size), 1)
    if k == 1:
        return pred
    pad = k // 2
    x = pred.float().unsqueeze(1)
    patches = F.unfold(x, kernel_size=k, padding=pad)
    median_vals = patches.median(dim=1).values
    return median_vals.view(pred.shape[0], pred.shape[1], pred.shape[2]).long()


@torch.no_grad()
def postprocess_prediction(
    pred: torch.Tensor,
    num_classes: int,
    min_component_area: int = 20,
    filter_mode: str = "majority",
    kernel_size: int = 3,
) -> torch.Tensor:
    out = pred.long()

    if filter_mode == "majority":
        out = _majority_filter(out, num_classes=num_classes, kernel_size=kernel_size)
    elif filter_mode == "median":
        out = _median_filter(out, kernel_size=kernel_size)

    if min_component_area > 1:
        out_np = out.detach().cpu().numpy()
        for i in range(out_np.shape[0]):
            out_np[i] = _remove_small_components_np(out_np[i], min_area=min_component_area)
        out = torch.from_numpy(out_np).to(pred.device, dtype=torch.long)

    return out
