from pathlib import Path


def test_pathlib_baseline():
    p = Path(__file__).parent
    s = p.joinpath("test_pathlib.py")
    assert s.exists()

    s2 = p.joinpath(Path("test_pathlib.py"))
    assert s2.exists()

    assert not p.is_relative_to(s2)
    assert s2.is_relative_to(p)
    assert s2.name == "test_pathlib.py"
    assert s2.name[0:-len(s2.suffix)] == "test_pathlib"
