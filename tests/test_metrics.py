import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from metrics import pedestrian_metrics, protocol_metrics  # noqa: E402


def test_perfect():
    gt = np.array([[1, 0, 1], [0, 1, 0]])
    m = pedestrian_metrics(gt, gt)
    assert np.isclose(m["mA"], 1) and np.isclose(m["Instance_F1"], 1) and np.isclose(m["Challenge_Avg"], 1)


def test_hand_computed():
    gt = np.array([[1, 0, 1, 0],
                   [0, 1, 1, 0],
                   [1, 1, 0, 0]])
    pred = np.array([[1, 1, 0, 0],
                     [0, 1, 1, 0],
                     [0, 0, 0, 1]])
    m = pedestrian_metrics(gt, pred)
    # attr0: TPR 1/2, TNR 1/1 -> .75 ; attr1: TPR 1/2, TNR 0/1 -> .25
    # attr2: TPR 1/2, TNR 1/1 -> .75 ; attr3: TPR 0/0->0, TNR 2/3 -> 1/3
    assert np.isclose(m["mA"], (0.75 + 0.25 + 0.75 + 1 / 3) / 4)
    # per image: prec 1/2, 1, 0 ; rec 1/2, 1, 0
    p = r = (0.5 + 1 + 0) / 3
    assert np.isclose(m["Instance_Prec"], p) and np.isclose(m["Instance_Rec"], r)
    assert np.isclose(m["Instance_F1"], 2 * p * r / (p + r))
    # acc: 1/3, 1, 0
    assert np.isclose(m["Instance_Acc"], (1 / 3 + 1) / 3)
    ca = 2 * m["mA"] * m["Instance_F1"] / (m["mA"] + m["Instance_F1"])
    assert np.isclose(m["Challenge_Avg"], ca)


def test_all_zero_predictions():
    gt = np.array([[1, 0], [0, 1], [1, 1]])
    pred = np.zeros_like(gt)
    m = pedestrian_metrics(gt, pred)
    assert np.isclose(m["mA"], 0.5)  # TPR 0, TNR 1 per attribute
    assert np.isclose(m["Instance_F1"], 0) and np.isclose(m["Challenge_Avg"], 0)
    assert np.isfinite(m["Challenge_Avg"])


def test_protocol_averaging():
    gt = np.array([[1, 0], [0, 1], [0, 1], [0, 1]])
    probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.1]])
    src = np.array(["A", "A", "B", "B"])
    # an attribute with no negatives (or no positives) in a subset gets TNR (TPR) = 0
    # B: attr0 has no positives and all-wrong preds (TPR 0, TNR 0); attr1 no negatives, TPR 0
    assert np.isclose(protocol_metrics(gt[2:], probs[2:], src[2:])["B"]["mA"], 0.0)
    # a perfect prediction on a constant attribute still only reaches 0.5
    assert np.isclose(protocol_metrics(gt[2:], 1 - probs[2:], src[2:])["B"]["mA"], 0.5)
    res = protocol_metrics(gt, probs, src)
    assert np.isclose(res["A"]["Challenge_Avg"], 1)
    assert np.isclose(res["mean"]["Challenge_Avg"], (res["A"]["Challenge_Avg"] + res["B"]["Challenge_Avg"]) / 2)


def test_threshold_strict():
    gt = np.array([[1]])
    assert pedestrian_metrics(gt, (np.array([[0.5]]) > 0.5).astype(int))["Instance_F1"] == 0
