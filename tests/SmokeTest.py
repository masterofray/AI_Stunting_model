"""Legacy smoke-test filename retained for compatibility with older CI runners."""

from pathlib import Path

from src.research.data import load_raw_data


def test_repository_raw_data_is_readable():
    root = Path(__file__).resolve().parents[1]
    df = load_raw_data(root / "src" / "query" / "RawData.sql")
    assert len(df) > 0
