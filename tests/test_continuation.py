"""Regression checks for verified dataset bytes and evaluation/inference parity."""
import hashlib
import json
from pathlib import Path
import shutil
import pytest
from training.maps import load_verified_maps
from training.evaluate import decision_mode


@pytest.mark.parametrize("saved,override,expected", [
    (True,None,True),(False,None,False),(True,False,False),(False,True,True)])
def test_evaluation_preserves_checkpoint_decision_mode(saved,override,expected):
    assert decision_mode({"deterministic":saved},override) is expected


def test_all_fifty_existing_maps_have_verified_disjoint_bytes():
    rows=load_verified_maps()
    assert len(rows)==50
    assert [sum(r["split"]==s for r in rows) for s in ("train","validation","test")]==[30,10,10]


def dataset(tmp_path, duplicate=False):
    original=json.loads(Path("training/maps/manifest.json").read_text())["maps"]
    rows=[]
    for i,split in enumerate(("train","validation")):
        src=original[0 if duplicate else i]
        target=tmp_path/f"{split}.wad"; shutil.copyfile(src["path"],target)
        rows.append(dict(src,path=str(target),split=split,
                         sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    (tmp_path/"manifest.json").write_text(json.dumps({"maps":rows}))
    return rows


def test_changed_wad_bytes_are_rejected(tmp_path):
    rows=dataset(tmp_path)
    path=Path(rows[0]["path"]); path.write_bytes(path.read_bytes()+b"tampered")
    with pytest.raises(ValueError,match="hash mismatch"): load_verified_maps(tmp_path)


def test_cross_split_identical_wads_are_rejected(tmp_path):
    dataset(tmp_path,duplicate=True)
    with pytest.raises(ValueError,match="overlap"): load_verified_maps(tmp_path)
