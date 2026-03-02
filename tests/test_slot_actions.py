import base64

from app import apply_slot_action, normalise_store_slots


def _as_upload_content(text):
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"data:text/plain;base64,{encoded}"


def test_reset_clears_all_slots():
    store = {
        "slots": [
            {"filename": "a.dat", "resolution": 80, "re": [1.0], "im": [2.0]},
            {"filename": "b.dat", "resolution": 90, "re": [3.0], "im": [4.0]},
        ]
    }
    upload_contents = [None, None, None]
    filenames = [None, None, None]
    res_values = [None, None, None]

    prepared = normalise_store_slots(store, num_slots=3)
    updated, err = apply_slot_action(
        store=prepared,
        triggered="btn-reset",
        upload_contents=upload_contents,
        filenames=filenames,
        res_values=res_values,
    )

    assert err is None
    assert updated["slots"] == [None, None, None]


def test_invalid_upload_returns_error_and_clears_slot():
    upload_contents = [_as_upload_content("not numeric data\n")]
    filenames = ["bad_95.dat"]
    res_values = [None]
    prepared = normalise_store_slots(None, num_slots=1)

    updated, err = apply_slot_action(
        store=prepared,
        triggered={"type": "upload", "index": 0},
        upload_contents=upload_contents,
        filenames=filenames,
        res_values=res_values,
    )

    assert updated["slots"][0] is None
    assert err is not None
    assert "Upload failed for bad_95.dat" in err


def test_valid_upload_sets_slot_and_infers_resolution():
    upload_contents = [_as_upload_content("1.0 2.0\n3.0 4.0\n")]
    filenames = ["eigs_90.dat"]
    res_values = [None]
    prepared = normalise_store_slots(None, num_slots=1)

    updated, err = apply_slot_action(
        store=prepared,
        triggered={"type": "upload", "index": 0},
        upload_contents=upload_contents,
        filenames=filenames,
        res_values=res_values,
    )

    assert err is None
    assert updated["slots"][0]["filename"] == "eigs_90.dat"
    assert updated["slots"][0]["resolution"] == 90
    assert updated["slots"][0]["re"] == [1.0, 3.0]
    assert updated["slots"][0]["im"] == [2.0, 4.0]


def test_resolution_edit_updates_existing_slot():
    prepared = {
        "slots": [
            {"filename": "eigs_90.dat", "resolution": 90, "re": [1.0], "im": [2.0]}
        ]
    }

    updated, err = apply_slot_action(
        store=prepared,
        triggered={"type": "resolution", "index": 0},
        upload_contents=[None],
        filenames=["eigs_90.dat"],
        res_values=[120],
    )

    assert err is None
    assert updated["slots"][0]["resolution"] == 120
