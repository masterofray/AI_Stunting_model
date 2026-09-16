import pandas as pd
from src.analysis.data_quality import DataQualityAnalyzer, DataLoader
from src.analysis.descriptive import DescriptiveAnalyzer
from src.analysis.statistical import StatisticalAnalyzer

# Load data
df = DataLoader.load_csv("DataProc.csv")

# Data quality
quality = DataQualityAnalyzer(df)
report = quality.generate_full_report()

# Descriptive statistics
desc = DescriptiveAnalyzer(df)
district_sum = desc.district_summary()

# Statistical tests
stat = StatisticalAnalyzer(df)
associations, chi_tests = stat.association_tables()