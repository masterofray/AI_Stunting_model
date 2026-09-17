from pathlib import Path

import pandas as pd

from src.research.data import load_raw_data, validate_dataset
from src.research.analysis import ResearchAnalyzer


ROOT = Path(__file__).resolve().parents[1]
RAW_SQL = ROOT / "src" / "query" / "RawData.sql"


def test_raw_sql_loader_and_validation():
    df = load_raw_data(RAW_SQL)
    assert not df.empty
    assert "Stunting" in df.columns
    report = validate_dataset(df)
    assert report["valid"] is True
    assert set(df["Stunting"].unique()).issubset({0, 1})


def test_analysis_outputs_have_expected_shape():
    df = load_raw_data(RAW_SQL)
    analyzer = ResearchAnalyzer(df)
    district = analyzer.district_summary()
    binary = analyzer.binary_associations()
    chi = analyzer.chi_square_tests()
    assert len(district) == df["District_Name"].nunique()
    assert len(binary) == 6
    assert len(chi) == 6


def test_loader_accepts_csv(tmp_path):
    sample = pd.DataFrame(
        {
            "District_Name": ["A", "A", "B", "B"],
            "Stunting": [0, 1, 0, 1],
            "Gender": [0, 1, 0, 1],
            "Malnutrition_Level": [0, 2, 0, 3],
            "Exclusive_Milky": [1, 0, 1, 0],
            "Smoke_Habit": [0, 1, 0, 1],
            "PureWater_Access": [1, 1, 0, 0],
            "Healthy_Toilet": [1, 0, 1, 0],
            "Difficult_Acess": [0, 1, 0, 1],
        }
    )
    path = tmp_path / "raw.csv"
    sample.to_csv(path, index=False)
    loaded = load_raw_data(path)
    assert loaded.equals(sample)
