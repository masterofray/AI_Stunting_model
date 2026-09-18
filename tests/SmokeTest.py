#!/usr/bin/env python3
from __future__ import annotations

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, masterofray/AI_Stunting_model"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.2.0"
__maintainer__ = "Aryanto, M.Si"
__email__      = "aryanto.dandan@gmail.com"
__created__    = "2026-08-31"
__modified__   = "2026-09-18"



"""Legacy smoke-test filename retained for compatibility with older CI runners."""

from pathlib import Path

from src.research.data import load_raw_data


def test_repository_raw_data_is_readable():
    root = Path(__file__).resolve().parents[1]
    df = load_raw_data(root / "src" / "query" / "RawData.sql")
    assert len(df) > 0
