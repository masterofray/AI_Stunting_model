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



"""Diverse, feasibility-constrained counterfactual explanations.

Generates actionable "what-if" scenarios on the *raw* (pre-transform)
feature space so the output stays interpretable for domain experts
(e.g. public health workers).

Design:
* A `FeatureSpec` declares type, mutability, allowed direction, bounds, cost.
* Genetic algorithm optimises: prediction flip + low effort + sparsity + diversity.
* Categoricals are handled natively (no one-hot mutation).
* Optional grouping: several counterfactuals returned per instance, all diverse.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..research.data import save_json


@dataclass
class FeatureSpec:
    """Declarative, domain-expert-friendly description of one raw feature."""

    name: str
    kind: str = "numeric"  # "numeric" | "categorical"
    mutable: bool = True
    direction: str = "both"  # "both" | "increase" | "decrease"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    categories: Optional[List[Any]] = None
    cost: float = 1.0
    # For numeric: mutation step as a fraction of the observed range
    step_frac: float = 0.05

    def clip_numeric(self, value: float) -> float:
        lo = self.min_value
        hi = self.max_value
        if lo is not None:
            value = max(lo, value)
        if hi is not None:
            value = min(hi, value)
        return float(value)


class CounterfactualExplainer:
    """Genetic-algorithm counterfactual generator for a fitted pipeline."""

    def __init__(
        self,
        fitted_pipeline,
        specs: Sequence[FeatureSpec],
        target_class: int = 1,
        seed: int = 42,
        population_size: int = 200,
        generations: int = 60,
        n_counterfactuals: int = 3,
        proximity_weight: float = 0.6,
        sparsity_weight: float = 0.15,
        diversity_weight: float = 0.25,
    ):
        self.pipeline = fitted_pipeline
        self.specs = list(specs)
        self.spec_by_name = {s.name: s for s in self.specs}
        self.feature_order = [s.name for s in self.specs]
        self.target_class = int(target_class)
        self.seed = int(seed)
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.n_counterfactuals = int(n_counterfactuals)
        self.proximity_weight = float(proximity_weight)
        self.sparsity_weight = float(sparsity_weight)
        self.diversity_weight = float(diversity_weight)
        self.rng = np.random.default_rng(self.seed)
        # Precompute numeric ranges for normalisation
        self._ranges: Dict[str, float] = {}
        for s in self.specs:
            if s.kind == "numeric":
                lo = s.min_value if s.min_value is not None else 0.0
                hi = s.max_value if s.max_value is not None else 1.0
                self._ranges[s.name] = max(hi - lo, 1e-6)

    # ---------- public API ----------

    def explain_instance(
        self,
        instance: pd.Series,
        X_reference: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Any]]:
        """Return up to `n_counterfactuals` diverse counterfactuals for one row."""
        x0 = instance[self.feature_order].copy()
        base_prob = float(self._predict_proba(pd.DataFrame([x0]))[0])
        # If already in target class, no flip needed
        if (base_prob >= 0.5) == (self.target_class == 1):
            return []

        counterfactuals: List[Dict[str, Any]] = []
        for _ in range(self.n_counterfactuals):
            best = self._evolve(x0, existing=counterfactuals)
            if best is None:
                continue
            counterfactuals.append(best)
        return counterfactuals

    def explain_many(
        self,
        X_test: pd.DataFrame,
        output_dir: str | Path,
        max_samples: int = 10,
    ) -> pd.DataFrame:
        """Run counterfactual search for many rows and persist results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, Any]] = []
        for position in range(min(max_samples, len(X_test))):
            row = X_test.iloc[position]
            cfs = self.explain_instance(row)
            fig = self._plot_row(row, cfs, sample_id=row.name)
            fig.savefig(
                output_dir / f"counterfactual_{row.name}.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)
            for cid, cf in enumerate(cfs):
                for feat, (old, new) in cf["changes"].items():
                    rows.append(
                        {
                            "sample_index": row.name,
                            "counterfactual_id": cid,
                            "feature": feat,
                            "old_value": old,
                            "new_value": new,
                            "delta": (
                                float(new) - float(old)
                                if isinstance(old, (int, float, np.floating))
                                and isinstance(new, (int, float, np.floating))
                                else None
                            ),
                            "predicted_probability": cf["probability"],
                            "distance": cf["distance"],
                            "sparsity": cf["sparsity"],
                        }
                    )
        result = pd.DataFrame(rows)
        result.to_csv(output_dir / "counterfactuals.csv", index=False)
        save_json(
            {
                "target_class": self.target_class,
                "n_samples": int(min(max_samples, len(X_test))),
                "n_counterfactuals_per_sample": self.n_counterfactuals,
                "features": [s.__dict__ for s in self.specs],
            },
            output_dir / "counterfactual_config.json",
        )
        return result

    # ---------- internals ----------

    def _predict_proba(self, X_raw: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X_raw)[:, 1]

    def _random_population(self, x0: pd.Series) -> List[pd.Series]:
        pop: List[pd.Series] = []
        for _ in range(self.population_size):
            candidate = x0.copy()
            for s in self.specs:
                if not s.mutable:
                    continue
                if self.rng.random() < 0.5:
                    candidate[s.name] = self._mutate_value(s, x0[s.name])
            pop.append(candidate)
        # Include x0 so the search can measure the baseline
        pop[0] = x0.copy()
        return pop

    def _mutate_value(self, spec: FeatureSpec, current: Any) -> Any:
        if spec.kind == "categorical":
            cats = spec.categories
            if not cats:
                return current
            if spec.direction == "both":
                pool = cats
            else:
                # interpret categorical direction as ordinal position
                try:
                    idx = cats.index(current)
                except ValueError:
                    pool = cats
                else:
                    pool = cats[idx + 1 :] if spec.direction == "increase" else cats[:idx]
                if not pool:
                    return current
            if len(pool) == 1:
                return pool[0]
            return pool[int(self.rng.integers(0, len(pool)))]
        # numeric
        step = spec.step_frac * self._ranges.get(spec.name, 1.0)
        delta = float(self.rng.normal(0.0, step))
        if spec.direction == "increase":
            delta = abs(delta)
        elif spec.direction == "decrease":
            delta = -abs(delta)
        new_val = spec.clip_numeric(float(current) + delta)
        return new_val

    def _distance(self, x: pd.Series, y: pd.Series) -> float:
        total = 0.0
        weight = 0.0
        for s in self.specs:
            w = s.cost
            weight += w
            a, b = x[s.name], y[s.name]
            if s.kind == "numeric":
                denom = self._ranges.get(s.name, 1.0)
                d = abs(float(a) - float(b)) / denom
            else:
                d = 0.0 if a == b else 1.0
            total += w * d
        return total / weight if weight > 0 else 0.0

    def _sparsity(self, x: pd.Series, y: pd.Series) -> float:
        changed = 0
        mutable = 0
        for s in self.specs:
            if not s.mutable:
                continue
            mutable += 1
            if s.kind == "numeric":
                if abs(float(x[s.name]) - float(y[s.name])) > 1e-9:
                    changed += 1
            else:
                if x[s.name] != y[s.name]:
                    changed += 1
        return changed / max(mutable, 1)

    def _fitness(
        self,
        x0: pd.Series,
        candidate: pd.Series,
        existing: Sequence[Dict[str, Any]],
        probability: float,
    ) -> float:
        target_score = probability if self.target_class == 1 else 1.0 - probability
        distance = self._distance(x0, candidate)
        sparsity = self._sparsity(x0, candidate)
        # Diversity: penalise similarity to already-found counterfactuals
        diversity_penalty = 0.0
        for cf in existing:
            prev = cf["_raw"]
            diversity_penalty += 1.0 - self._distance(prev, candidate)
        return (
            target_score
            - self.proximity_weight * distance
            - self.sparsity_weight * sparsity
            - self.diversity_weight * (diversity_penalty / max(len(existing), 1))
        )

    def _evolve(
        self,
        x0: pd.Series,
        existing: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        population = self._random_population(x0)
        best_record: Optional[Dict[str, Any]] = None

        for _ in range(self.generations):
            pop_frame = pd.DataFrame(population)[self.feature_order]
            probs = self._predict_proba(pop_frame)
            scored = [
                (
                    self._fitness(x0, cand, existing, float(p)),
                    float(p),
                    cand,
                )
                for cand, p in zip(population, probs)
            ]
            scored.sort(key=lambda t: t[0], reverse=True)

            # Track best valid flip
            for score, prob, cand in scored:
                flipped = (prob >= 0.5) == (self.target_class == 1)
                if flipped:
                    record = self._make_record(x0, cand, prob)
                    if best_record is None or record["distance"] < best_record["distance"]:
                        best_record = record
                    break

            # Selection
            survivors = [c for _, _, c in scored[: self.population_size // 3]]
            children: List[pd.Series] = []
            while len(children) < self.population_size - len(survivors):
                p1 = survivors[int(self.rng.integers(0, len(survivors)))]
                p2 = survivors[int(self.rng.integers(0, len(survivors)))]
                child = p1.copy()
                for s in self.specs:
                    if not s.mutable:
                        continue
                    if self.rng.random() < 0.5:
                        child[s.name] = p1[s.name]
                    else:
                        child[s.name] = p2[s.name]
                    if self.rng.random() < 0.15:
                        child[s.name] = self._mutate_value(s, child[s.name])
                children.append(child)
            population = survivors + children

        if best_record is None:
            return None
        # store raw candidate for diversity penalty
        best_record["_raw"] = population[0]
        return best_record

    def _make_record(
        self, x0: pd.Series, candidate: pd.Series, probability: float
    ) -> Dict[str, Any]:
        changes: Dict[str, Any] = {}
        for s in self.specs:
            a, b = x0[s.name], candidate[s.name]
            if s.kind == "numeric":
                if abs(float(a) - float(b)) > 1e-9:
                    changes[s.name] = (float(a), float(b))
            else:
                if a != b:
                    changes[s.name] = (a, b)
        return {
            "probability": float(probability),
            "distance": float(self._distance(x0, candidate)),
            "sparsity": float(self._sparsity(x0, candidate)),
            "changes": changes,
            "_raw": candidate,
        }

    def _plot_row(
        self,
        row: pd.Series,
        counterfactuals: Sequence[Dict[str, Any]],
        sample_id: Any,
    ) -> plt.Figure:
        labels: List[str] = []
        old_vals: List[float] = []
        new_vals: List[float] = []
        for cf in counterfactuals:
            for feat, (old, new) in cf["changes"].items():
                spec = self.spec_by_name[feat]
                if spec.kind != "numeric":
                    continue
                labels.append(feat)
                old_vals.append(float(old))
                new_vals.append(float(new))
        fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(labels) + 1)))
        if labels:
            idx = np.arange(len(labels))
            ax.barh(idx - 0.2, old_vals, height=0.4, label="original", color="#4C72B0")
            ax.barh(idx + 0.2, new_vals, height=0.4, label="counterfactual", color="#DD8452")
            ax.set_yticks(idx)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.legend(loc="best")
        ax.set_title(f"Counterfactuals for row {sample_id}")
        fig.tight_layout()
        return fig