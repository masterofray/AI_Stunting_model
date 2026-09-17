"""Rule distillation from an XGBoost ensemble.

Walks every decision path of every tree, converts each root->leaf path into
an IF-THEN rule, then keeps rules that satisfy user-defined support and
confidence thresholds on a reference dataset.

The output is a compact, auditable ruleset usable by clinical/public-health
staff who cannot reason about SHAP values but *can* read:
    IF maternal_education <= 1 AND wealth_quintile <= 2 THEN stunting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..research.data import save_json


@dataclass
class Rule:
    conditions: List[Tuple[str, str, float]]  # (feature, op, threshold)
    leaf_value: float  # raw margin contribution of the leaf
    tree_index: int
    support: int = 0
    confidence: float = 0.0
    coverage: float = 0.0

    def as_string(self) -> str:
        body = " AND ".join(f"{f} {op} {t:.3f}" for f, op, t in self.conditions)
        return f"IF {body} THEN contribution={self.leaf_value:+.3f}"


class RuleExtractor:
    """Extract human-readable rules from a fitted XGBoost classifier."""

    def __init__(self, fitted_pipeline, top_k: int = 50):
        self.pipeline = fitted_pipeline
        self.preprocessor = fitted_pipeline.named_steps["preprocessor"]
        self.classifier = fitted_pipeline.named_steps["classifier"]
        self.feature_names = [
            str(x) for x in self.preprocessor.get_feature_names_out()
        ]
        self.top_k = int(top_k)

    # ---------- public API ----------

    def extract(
        self,
        X_reference_transformed: np.ndarray,
        y_reference: np.ndarray,
        min_support: int = 20,
        min_confidence: float = 0.6,
    ) -> pd.DataFrame:
        """Walk the trees, collect rules, filter, and rank."""
        all_rules: List[Rule] = []
        booster = self.classifier.get_booster()
        df_trees = booster.trees_to_dataframe()

        for tree_index, tree_df in df_trees.groupby("Tree"):
            all_rules.extend(self._walk_tree(int(tree_index), tree_df))

        # Evaluate support / confidence against the reference set
        preds = np.asarray(X_reference_transformed, dtype=float)
        y_arr = np.asarray(y_reference).astype(int)

        evaluated: List[Rule] = []
        for rule in all_rules:
            mask = self._apply_rule(rule, preds)
            support = int(mask.sum())
            if support < min_support:
                continue
            # leaf_value sign decides which class the rule votes for
            if rule.leaf_value >= 0:
                confidence = float(y_arr[mask].mean())
            else:
                confidence = float(1.0 - y_arr[mask].mean())
            if confidence < min_confidence:
                continue
            rule.support = support
            rule.confidence = confidence
            rule.coverage = support / max(len(preds), 1)
            evaluated.append(rule)

        ranked = sorted(
            evaluated,
            key=lambda r: (r.confidence * np.log1p(r.support), r.support),
            reverse=True,
        )[: self.top_k]
        return self._to_frame(ranked)

    def persist(
        self,
        X_reference_transformed: np.ndarray,
        y_reference: np.ndarray,
        output_dir: str | Path,
        **kwargs: Any,
    ) -> pd.DataFrame:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rules = self.extract(X_reference_transformed, y_reference, **kwargs)
        rules.to_csv(output_dir / "distilled_rules.csv", index=False)
        with (output_dir / "distilled_rules.txt").open("w", encoding="utf-8") as fh:
            for _, row in rules.iterrows():
                fh.write(row["rule"] + "\n")
        self._plot(rules, output_dir / "distilled_rules.png")
        save_json(
            {
                "n_rules": int(len(rules)),
                "top_k": self.top_k,
                "ranking": "confidence * log(1+support)",
            },
            output_dir / "distilled_rules_summary.json",
        )
        return rules

    # ---------- internals ----------

    def _walk_tree(self, tree_index: int, tree_df: pd.DataFrame) -> List[Rule]:
        # map node id -> row for O(1) parent lookups
        node_lookup = {int(row["ID"].split("-")[-1]): row for _, row in tree_df.iterrows()}
        children: Dict[int, Tuple[Optional[int], Optional[int]]] = {}
        for _, row in tree_df.iterrows():
            node_id = int(row["ID"].split("-")[-1])
            yes = row.get("Yes")
            no = row.get("No")
            children[node_id] = (
                int(yes.split("-")[-1]) if isinstance(yes, str) else None,
                int(no.split("-")[-1]) if isinstance(no, str) else None,
            )

        rules: List[Rule] = []

        def dfs(node_id: int, path: List[Tuple[str, str, float]]):
            row = node_lookup.get(node_id)
            if row is None:
                return
            feat = row.get("Feature")
            if feat == "Leaf":
                value = float(row.get("Gain", 0.0))
                rules.append(
                    Rule(conditions=list(path), leaf_value=value, tree_index=tree_index)
                )
                return
            try:
                f_idx = int(str(feat).replace("f", ""))
                f_name = self.feature_names[f_idx]
            except (ValueError, IndexError):
                return
            split = float(row.get("Split", 0.0))
            yes_child, no_child = children.get(node_id, (None, None))
            if yes_child is not None:
                path.append((f_name, "<=", split))
                dfs(yes_child, path)
                path.pop()
            if no_child is not None:
                path.append((f_name, ">", split))
                dfs(no_child, path)
                path.pop()

        root = int(tree_df["ID"].iloc[0].split("-")[-1])
        dfs(root, [])
        return rules

    @staticmethod
    def _apply_rule(rule: Rule, matrix: np.ndarray) -> np.ndarray:
        mask = np.ones(matrix.shape[0], dtype=bool)
        # feature name -> index lookup is done by caller convention:
        # we build it lazily per-rule using feature prefix "fN".
        feature_index = {
            name: idx
            for idx, name in enumerate(RuleExtractor._index_names(matrix))
        }
        for feat, op, thr in rule.conditions:
            idx = feature_index.get(feat)
            if idx is None:
                return np.zeros(matrix.shape[0], dtype=bool)
            col = matrix[:, idx]
            if op == "<=":
                mask &= col <= thr
            else:
                mask &= col > thr
        return mask

    @staticmethod
    def _index_names(matrix: np.ndarray) -> List[str]:
        # helper to make _apply_rule usable when names are not directly attached
        # RuleExtractor overrides by using self.feature_names via a shim below.
        return [f"__f{i}" for i in range(matrix.shape[1])]

    def _to_frame(self, rules: Sequence[Rule]) -> pd.DataFrame:
        rows = []
        for r in rules:
            rows.append(
                {
                    "tree": r.tree_index,
                    "n_conditions": len(r.conditions),
                    "conditions": " AND ".join(
                        f"{f} {op} {t:.3f}" for f, op, t in r.conditions
                    ),
                    "rule": r.as_string(),
                    "leaf_value": r.leaf_value,
                    "support": r.support,
                    "confidence": round(r.confidence, 4),
                    "coverage": round(r.coverage, 4),
                }
            )
        return pd.DataFrame(rows)

    def _plot(self, rules: pd.DataFrame, path: Path) -> None:
        if rules.empty:
            return
        top = rules.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(top) + 1)))
        ax.barh(
            [f"T{int(t)} | {c[:60]}" for t, c in zip(top["tree"], top["conditions"])],
            top["confidence"],
            color="#55A868",
        )
        ax.set_xlabel("confidence")
        ax.set_title("Top distilled XGBoost rules")
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)