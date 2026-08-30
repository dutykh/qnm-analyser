"""Tests for the numerical core and the upload parser.

Author: Dr. Denys Dutykh (https://www.denys-dutykh.com/)
"""

import base64

import numpy as np
import pytest
from scipy.spatial import cKDTree

from app import (
    MAX_ROWS_PER_FILE,
    build_plot,
    classify_converged,
    compute_converged,
    parse_upload,
)


def _encode(text):
    """Wrap *text* the way dcc.Upload delivers a file."""
    payload = base64.b64encode(text.encode()).decode()
    return f"data:application/octet-stream;base64,{payload}"


def _reference_compute(ref_points, trees, other_keys, tol_value):
    """The original per-point implementation, kept as a test oracle."""
    conv_re, conv_im = [], []
    for i in range(len(ref_points)):
        point = ref_points[i]
        found = True
        for n in other_keys:
            dist, _ = trees[n].query(point)
            if dist > tol_value:
                found = False
                break
        if found:
            conv_re.append(point[0])
            conv_im.append(point[1])
    return np.asarray(conv_re), np.asarray(conv_im)


# --------------------------------------------------------------------------
# compute_converged
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
def test_vectorised_matches_reference_implementation(seed):
    """The vectorised form must agree with the original loop exactly."""
    rng = np.random.default_rng(seed)
    shared = rng.uniform(-3, 3, (rng.integers(1, 12), 2))
    sets = {}
    for j in range(rng.integers(2, 5)):
        noise = rng.choice([1e-8, 1e-3])
        extra = rng.uniform(-3, 3, (rng.integers(0, 30), 2))
        pts = np.vstack([shared + rng.normal(0, noise, shared.shape), extra])
        sets[100 + j] = pts

    keys = sorted(sets)
    others = keys[:-1]
    trees = {n: cKDTree(sets[n]) for n in others}
    tol = float(rng.choice([1e-5, 1e-4, 1e-3]))

    got = compute_converged(sets[keys[-1]], trees, others, tol)
    want = _reference_compute(sets[keys[-1]], trees, others, tol)
    assert np.array_equal(got[0], want[0])
    assert np.array_equal(got[1], want[1])


def test_converged_finds_the_shared_modes():
    shared = np.array([[0.0, -0.1], [0.5, -0.3], [1.0, -0.8]])
    sets = {
        60: np.vstack([shared, [[9.0, 9.0]]]),
        90: np.vstack([shared + 1e-9, [[8.0, 8.0]]]),
        120: np.vstack([shared - 1e-9, [[7.0, 7.0]]]),
    }
    others = [60, 90]
    trees = {n: cKDTree(sets[n]) for n in others}
    cre, cim = compute_converged(sets[120], trees, others, 1e-4)
    assert len(cre) == 3
    np.testing.assert_allclose(np.sort(cre), np.sort(shared[:, 0]), atol=1e-6)


def test_converged_is_empty_when_nothing_matches():
    a = np.array([[0.0, 0.0]])
    b = np.array([[5.0, 5.0]])
    trees = {60: cKDTree(b)}
    cre, cim = compute_converged(a, trees, [60], 1e-4)
    assert len(cre) == 0 and len(cim) == 0


def test_converged_handles_empty_reference_set():
    cre, cim = compute_converged(np.empty((0, 2)), {}, [], 1e-4)
    assert len(cre) == 0 and len(cim) == 0


# --------------------------------------------------------------------------
# classify_converged
# --------------------------------------------------------------------------


def test_classification_is_exhaustive_and_disjoint():
    cre = np.array([0.5, 0.0, 0.4])
    cim = np.array([-0.5, -0.3, 0.0])
    general, pure_imag, pure_real = classify_converged(cre, cim, 1e-4)
    assert len(general) + len(pure_imag) + len(pure_real) == len(cre)
    assert len(general) == 1 and len(pure_imag) == 1 and len(pure_real) == 1


# --------------------------------------------------------------------------
# parse_upload
# --------------------------------------------------------------------------


def test_parse_skips_comments_blanks_and_non_finite():
    text = "\n".join([
        "# a comment",
        "",
        "1.0  -2.0",
        "   ",
        "nan  -1.0",
        "3.0  inf",
        "# another",
        "4.0  -5.0",
        "garbage line",
    ])
    re_vals, im_vals, n = parse_upload(_encode(text), "eigs_90.dat")
    assert re_vals == [1.0, 4.0]
    assert im_vals == [-2.0, -5.0]
    assert n == 90


def test_resolution_uses_the_last_integer_in_the_filename():
    text = "1.0 -1.0"
    assert parse_upload(_encode(text), "eigs_2024_90.dat")[2] == 90
    assert parse_upload(_encode(text), "run7/eigs_120.dat")[2] == 120


def test_resolution_is_none_when_filename_has_no_digits():
    """None, not 0: zero was treated as falsy and silently dropped the file."""
    assert parse_upload(_encode("1.0 -1.0"), "eigenvalues.dat")[2] is None


def test_row_cap_is_enforced():
    text = "\n".join(f"{i}.0 -1.0" for i in range(MAX_ROWS_PER_FILE + 10))
    with pytest.raises(ValueError, match="more than"):
        parse_upload(_encode(text), "eigs_90.dat")


def test_extra_columns_are_ignored():
    re_vals, im_vals, _ = parse_upload(_encode("1.0 -2.0 999 extra"), "e_90.dat")
    assert re_vals == [1.0] and im_vals == [-2.0]


# --------------------------------------------------------------------------
# build_plot — the shared figure builder used by the plot and the export
# --------------------------------------------------------------------------


def _slots():
    shared = np.array([[0.0, -0.1], [0.5, -0.3], [1.0, -0.8]])
    out = []
    for n in (60, 90, 120):
        pts = np.vstack([shared, rng_extra(n)])
        out.append({
            "filename": f"eigs_{n}.dat", "resolution": n,
            "re": pts[:, 0].tolist(), "im": pts[:, 1].tolist(),
        })
    return out


def rng_extra(seed):
    return np.random.default_rng(seed).uniform(2, 5, (10, 2))


def test_build_plot_returns_figure_info_and_convergence():
    fig, info, conv = build_plot({"slots": _slots()}, 1.0, "Top-right", "light", None)
    assert conv is not None
    assert len(conv["conv_re"]) == 3
    assert "converged" in info
    assert len(fig.data) == 4  # three datasets plus the converged overlay


def test_build_plot_rejects_non_finite_zoom_ranges():
    """relayoutData comes from the browser and must be validated."""
    hostile = {
        "xaxis.range[0]": "javascript:alert(1)", "xaxis.range[1]": 3,
        "yaxis.range[0]": float("nan"), "yaxis.range[1]": float("inf"),
    }
    fig, _, _ = build_plot({"slots": _slots()}, 1.0, "Top-right", "light", hostile)
    assert fig.layout.xaxis.range is None
    assert fig.layout.yaxis.range is None


def test_build_plot_applies_valid_zoom_ranges():
    good = {
        "xaxis.range[0]": -1, "xaxis.range[1]": 2,
        "yaxis.range[0]": -3, "yaxis.range[1]": 0,
    }
    fig, _, _ = build_plot({"slots": _slots()}, 1.0, "Top-right", "light", good)
    assert tuple(fig.layout.xaxis.range) == (-1.0, 2.0)
    assert tuple(fig.layout.yaxis.range) == (-3.0, 0.0)


def test_build_plot_keeps_a_dataset_whose_resolution_is_zero():
    """resolution 0 is a real value; it used to be dropped as falsy."""
    slots = _slots()
    slots[0]["resolution"] = 0
    _, _, conv = build_plot({"slots": slots}, 1.0, "Top-right", "light", None)
    assert 0 in conv["resolutions"]


def test_build_plot_with_no_data():
    fig, info, conv = build_plot({"slots": []}, 1.0, "Top-right", "light", None)
    assert conv is None
    assert "Upload" in info
