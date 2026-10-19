import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from calibrate import _Dom, fit_offsets, sigmoid  # noqa: E402
from decode import GROUPS, apply_decoding  # noqa: E402
from metrics import pedestrian_metrics, protocol_metrics  # noqa: E402


def test_fast_column_scores_match_reference():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(300, 6))
    y = (rng.random((300, 6)) < 0.3).astype(int)
    delta = rng.normal(scale=0.5, size=6)
    d = _Dom(z, y)
    d.set(delta)
    grid = np.array([-1.0, 0.0, 0.7])
    s = d.scores_for_column(2, grid)
    for g, v in zip(grid, s):
        dd = delta.copy()
        dd[2] = g
        ref = pedestrian_metrics(y, ((z + dd) > 0).astype(int))["Challenge_Avg"]
        assert np.isclose(v, ref)


def test_fit_offsets_improves():
    rng = np.random.default_rng(1)
    y = (rng.random((600, 5)) < 0.1).astype(int)
    z = rng.normal(size=(600, 5)) + 2 * y - 2.5  # under-confident positives
    src = np.array(["A", "B", "C"] * 200)
    before = protocol_metrics(y, sigmoid(z), src)["mean"]["Challenge_Avg"]
    d = fit_offsets(z, y, src, log=lambda *_: None)
    after = protocol_metrics(y, sigmoid(z + d), src)["mean"]["Challenge_Avg"]
    assert after > before


def test_decoding_forces_one_positive():
    p = np.full((2, 40), 0.1, dtype=np.float32)
    p[0, 1] = 0.3
    p[1, 0], p[1, 1] = 0.9, 0.8
    out = apply_decoding(p, {"age": GROUPS["age"]})
    assert out[0, 1] > 0.5 and (out[0, [0, 2]] < 0.5).all()
    assert np.allclose(out[1], p[1])
