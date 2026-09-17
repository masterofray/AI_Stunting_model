#!/usr/bin/env python3
"""Main Entry Point for Mamberamo Stunting Analysis"""
import argparse
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from analysis.data_quality import DataQualityAnalyzer, DataLoader
from analysis.descriptive import DescriptiveAnalyzer
from analysis.statistical import StatisticalAnalyzer
from analysis.bayesian import BayesianAnalyzer
from analysis.visualization import DataVisualizer
from model.logistic import LogisticModel
from model.evaluation import ModelEvaluator
from model.prediction import PredictionEngine

def parse_args():
    parser = argparse.ArgumentParser(description="Mamberamo Stunting Analysis Pipeline")
    parser.add_argument("--input", type=str, default="DataProc.csv", help="Input CSV file")
    parser.add_argument("--output", type=str, default="analysis_output", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Run quick analysis only")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()

def run_full_analysis(input_path: str, output_dir: str, verbose: bool, skip_plots: bool) -> None:
    df = DataLoader.load_csv(input_path)
    if verbose:
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data quality
    if verbose: print("Running data quality analysis...")
    analyzer = DataQualityAnalyzer(df)
    analyzer.save_report(output_dir / "data_quality.json")
    analyzer.get_deduplicated_data().to_csv(output_dir / "data_deduplicated.csv", index=False)

    # Descriptive
    if verbose: print("Running descriptive analysis...")
    desc = DescriptiveAnalyzer(df)
    desc.save_all_summaries(output_dir)

    # Statistical
    if verbose: print("Running statistical analysis...")
    stat = StatisticalAnalyzer(df)
    stat.save_all_tests(output_dir)

    # Bayesian
    if verbose: print("Running Bayesian analysis...")
    bayes = BayesianAnalyzer(df)
    bayes.save_all_bayesian(output_dir)

    # Visualization
    if not skip_plots:
        if verbose: print("Generating visualizations...")
        viz = DataVisualizer(df)
        viz.save_all_plots(output_dir)

    # Modeling
    if verbose: print("Running modeling...")
    lr = LogisticModel(df)
    cv_results = lr.cross_validate()
    with open(output_dir / "logistic_cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)
    lr.fit()
    lr.get_coefficients().to_csv(output_dir / "logistic_coefficients.csv", index=False)
    lr.save_model(output_dir / "logistic_model.joblib")
    sens = lr.compare_with_deduplication()
    with open(output_dir / "deduplication_sensitivity_modeling.json", "w") as f:
        json.dump(sens, f, indent=2)

    # Bayesian modeling
    if verbose: print("Running Bayesian modeling...")
    try:
        bm = BayesianModel(df)
        dist, hyper = bm.fit_beta_binomial_hierarchical()
        dist.to_csv(output_dir / "bayesian_hierarchical_district.csv", index=False)
        with open(output_dir / "bayesian_hierarchical_hyperparams.json", "w") as f:
            json.dump(hyper, f, indent=2)
        try:
            fix, re = bm.fit_mixed_logit_variational()
            fix.to_csv(output_dir / "bayesian_mixed_logit_fixed.csv", index=False)
            with open(output_dir / "bayesian_mixed_logit_random.json", "w") as f:
                json.dump(re, f, indent=2)
        except Exception as e:
            if verbose: print(f"Warning: {e}")
    except Exception as e:
        if verbose: print(f"Warning: {e}")

    # Evaluation
    if verbose: print("Running evaluation...")
    X, y = lr.prepare_data()
    evaluator = ModelEvaluator()
    report = evaluator.generate_evaluation_report(lr, X, y)
    with open(output_dir / "model_evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Prediction
    if verbose: print("Running prediction...")
    predictor = PredictionEngine(lr)
    predictions = predictor.batch_predict(df)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    try:
        predictor.get_feature_importance().to_csv(output_dir / "feature_importance.csv", index=False)
    except:
        pass

    # Comprehensive report
    comprehensive = {
        "timestamp": datetime.now().isoformat(),
        "n_observations": len(df),
        "n_districts": int(df["District_Name"].nunique()),
        "stunting_prevalence": float(df["Stunting"].mean()),
    }
    with open(output_dir / "COMPREHENSIVE_REPORT.json", "w") as f:
        json.dump(comprehensive, f, indent=2)

    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"Stunting prevalence: {df['Stunting'].mean():.3%}")
    print(f"Districts: {df['District_Name'].nunique()}")

def run_quick_analysis(input_path: str, output_dir: str, verbose: bool) -> None:
    df = DataLoader.load_csv(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = DataQualityAnalyzer(df)
    analyzer.save_report(output_dir / "data_quality.json")

    desc = DescriptiveAnalyzer(df)
    desc.save_all_summaries(output_dir)

    stat = StatisticalAnalyzer(df)
    stat.save_all_tests(output_dir)

    lr = LogisticModel(df)
    cv_results = lr.cross_validate(n_splits=3)
    with open(output_dir / "logistic_cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)

    print(f"\nQuick analysis complete! Results saved to: {output_dir}")

def main():
    args = parse_args()
    if args.quick:
        run_quick_analysis(args.input, args.output, args.verbose)
    else:
        run_full_analysis(args.input, args.output, args.verbose, args.skip_plots)

if __name__ == "__main__":
    main()