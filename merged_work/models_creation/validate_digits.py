"""Quick validation for digit formations (N in 10..30)."""
import numpy as np

from merged_work.models_creation.digit_formations import (
    min_pairwise_distance_xy,
    sample_digit_offsets,
)

spacing = 2.0
dmin_threshold = 0.3 * spacing - 1e-3
failures = []
for N in [10, 15, 20, 25, 30]:
    heights = []
    for d in range(10):
        off = sample_digit_offsets(d, N, spacing=spacing)
        assert off.shape == (N, 3), (d, N, off.shape)
        assert np.all(np.isfinite(off))
        xy = off[:, :2]
        bbox_mid = 0.5 * (xy.max(axis=0) + xy.min(axis=0))
        center_err = float(np.linalg.norm(bbox_mid))
        dmin = min_pairwise_distance_xy(xy)
        heights.append(float(xy[:, 1].max() - xy[:, 1].min()))
        if dmin < dmin_threshold:
            failures.append((d, N, dmin))
        if center_err > 0.05:
            failures.append(("center", d, N, center_err))
    if max(heights) - min(heights) > 0.05:
        failures.append(("height", N, min(heights), max(heights)))

print("failures:", len(failures))
for f in failures:
    print(f)
if not failures:
    print("ALL OK")
