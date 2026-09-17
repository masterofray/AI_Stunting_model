# XAI Module for Stunting XGBoost

Welcome to the **XAI module** – a bunch of tools that go waaay beyond just calling `shap` and `lime` and calling it a day. This is for when you want your explainable AI to actually *explain something useful* to a human (like a health worker or a policy maker), not just produce pretty plots for your paper.

We've got counterfactuals, rule distillation, interaction detection, stability checks, and subgroup analysis. All modular, all in Python, all ready to plug into your existing `XAIExplainer` pipeline.

---

## What's inside?

Here's the file map – everything lives in the `xai/` folder:

| File | What it does (in plain English) |
|------|----------------------------------|
| `explain.py` | The OG: SHAP, LIME, native XGBoost importance. (You already had this, we kept it.) |
| `counterfactual.py` | "What's the least a family needs to change to not be stunted?" – uses a genetic algorithm to find minimal, feasible changes. |
| `rules.py` | Turns your XGBoost trees into readable `IF ... THEN ...` rules. Perfect for clinical guidelines. |
| `interactions.py` | Finds which features *interact* (e.g., maternal education × wealth). Two methods: SHAP interactions + Friedman's H. |
| `stability.py` | Bootstraps your SHAP values to check if your "top features" are actually stable. Reviewers love this. |
| `subgroup.py` | Are explanations the same for rural vs urban? Rich vs poor? ICE curves + SHAP divergence. |
| `__init__.py` | Exposes all the classes so you can `from xai import ...` nicely. |

---

## Installation

You probably already have most of this, but just in case:

```bash
pip install numpy pandas scikit-learn xgboost matplotlib shap lime scipy
```

> `lime` is optional – if it's not installed, the LIME part gracefully skips.
> `scipy` is needed for the stability module's Spearman correlation.

Make sure your project has the `research.data.save_json` helper (you referenced it in `explain.py`). If not, just write a tiny one yourself:

```python
# research/data.py
import json
from pathlib import Path

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
```

---

## How to run (the short version)

Assuming you already have a fitted `pipeline` (with `preprocessor` and `classifier` steps) and your data in `X_train`, `X_test`, `y_train`:

```python
from xai import (
    XAIExplainer,
    CounterfactualExplainer, FeatureSpec,
    RuleExtractor,
    InteractionExplainer,
    StabilityExplainer,
    SubgroupExplainer,
)
from pathlib import Path

# Where to dump all the goodies
out = Path("outputs/xai")
out.mkdir(parents=True, exist_ok=True)

# 1. The classic stuff
XAIExplainer(pipeline).run_all(X_train, X_test, out / "baseline")

# 2. Counterfactuals – you define the constraints once
specs = [
    FeatureSpec("maternal_education", "categorical", mutable=True,
                direction="increase", categories=[0,1,2,3], cost=1.0),
    FeatureSpec("wealth_quintile", "categorical", mutable=True,
                direction="increase", categories=[1,2,3,4,5], cost=1.2),
    FeatureSpec("child_age_months", "numeric", mutable=False),
    FeatureSpec("improved_water", "categorical", mutable=True,
                categories=[0,1], cost=0.6),
    # ... add every raw feature you care about
]
CounterfactualExplainer(pipeline, specs, target_class=1).explain_many(
    X_test, out / "counterfactual", max_samples=15
)

# 3. Rule distillation
RuleExtractor(pipeline, top_k=40).persist(
    preprocessor.transform(X_train), y_train.values,
    out / "rules", min_support=25, min_confidence=0.65
)

# 4. Interactions
inter = InteractionExplainer(pipeline)
inter.shap_interactions(X_test, out / "interactions")
inter.friedman_h(X_test, out / "interactions")

# 5. Stability
StabilityExplainer(pipeline).shap_stability(
    X_test, out / "stability", n_bootstrap=30, top_k=10
)

# 6. Subgroup stuff
sub = SubgroupExplainer(pipeline)
sub.shap_divergence(X_test, subgroup_col="residence", output_dir=out / "subgroup")
sub.ice_curves(X_test, feature="wealth_quintile_2",
               subgroup_col="residence", output_dir=out / "subgroup")
```

That's it. Each call creates its own subfolder with CSVs, PNGs, and JSON summaries.

---

## Example: Counterfactuals for one child

Let's say you have one test row and you want to know: *"What needs to change for this kid to not be predicted as stunted?"*

```python
from xai import CounterfactualExplainer, FeatureSpec

# Suppose your pipeline is already fitted
specs = [
    FeatureSpec("maternal_education", "categorical", mutable=True,
                direction="increase", categories=[0,1,2,3]),
    FeatureSpec("wealth_quintile", "categorical", mutable=True,
                direction="increase", categories=[1,2,3,4,5]),
    FeatureSpec("child_age_months", "numeric", mutable=False),
    FeatureSpec("improved_water", "categorical", mutable=True, categories=[0,1]),
    FeatureSpec("improved_sanitation", "categorical", mutable=True, categories=[0,1]),
]

cf = CounterfactualExplainer(pipeline, specs, target_class=1, n_counterfactuals=3)

one_row = X_test.iloc[42]  # or whatever row you want
results = cf.explain_instance(one_row)

for i, r in enumerate(results):
    print(f"\n--- Counterfactual #{i+1} ---")
    print(f"New predicted probability: {r['probability']:.3f}")
    print(f"Distance from original: {r['distance']:.3f} (lower is closer)")
    print("Changes needed:")
    for feat, (old, new) in r["changes"].items():
        print(f"  {feat}: {old} → {new}")
```

**Example output (made up, but you get the idea):**

```
--- Counterfactual #1 ---
New predicted probability: 0.487
Distance from original: 0.23 (lower is closer)
Changes needed:
  maternal_education: 1 → 2
  improved_water: 0 → 1
```

Nice, right? Now you can say: *"For this child, the model suggests that improving maternal education by one level and providing improved water would flip the prediction."*

---

## Example: Distilled rules

```python
from xai import RuleExtractor

# Need transformed training data for support/confidence
X_train_trans = preprocessor.transform(X_train)

rules = RuleExtractor(pipeline, top_k=20).persist(
    X_train_trans, y_train.values,
    output_dir="outputs/xai/rules",
    min_support=30,
    min_confidence=0.7
)

print(rules[["conditions", "support", "confidence", "coverage"]].head(5))
```

You'll get a `distilled_rules.txt` with lines like:

```
IF maternal_education <= 1.000 AND wealth_quintile <= 2.000 THEN contribution=+0.842
IF child_age_months <= 12.000 AND improved_water <= 0.000 THEN contribution=+0.613
...
```

Now you can hand these to a domain expert and say: *"Look, the model basically learned these rules. Do they make sense?"*

---

## Example: Subgroup divergence

```python
from xai import SubgroupExplainer

sub = SubgroupExplainer(pipeline)
div = sub.shap_divergence(
    X_test,
    subgroup_col="residence",  # e.g., "urban" vs "rural"
    output_dir="outputs/xai/subgroup"
)

print(div)
# Output: pairwise JSD between subgroups' SHAP importance vectors
```

If the JSD is high, it means the model uses *different features* for different subgroups – that's a big finding for your discussion section.

---

## What you get in the output folder

After running everything, your `outputs/xai/` will look something like:

```
outputs/xai/
├── baseline/
│   ├── native_xgboost_importance.csv
│   ├── shap_global_importance.csv
│   ├── shap_summary_bar.png
│   ├── shap_summary_beeswarm.png
│   ├── shap_local_explanations.csv
│   ├── lime_local_explanations.csv
│   └── ...
├── counterfactual/
│   ├── counterfactuals.csv
│   ├── counterfactual_config.json
│   └── counterfactual_<row_id>.png
├── rules/
│   ├── distilled_rules.csv
│   ├── distilled_rules.txt
│   ├── distilled_rules.png
│   └── distilled_rules_summary.json
├── interactions/
│   ├── shap_interactions.csv
│   ├── shap_interaction_heatmap.png
│   ├── friedman_h.csv
│   └── ...
├── stability/
│   ├── shap_stability.csv
│   ├── shap_stability.png
│   └── shap_stability_summary.json
└── subgroup/
    ├── subgroup_shap_divergence.csv
    ├── subgroup_shap_importance.csv
    ├── ice_<feature>_<subgroup>.csv
    └── *.png
```

Everything is CSV/JSON so you can easily pull it into your paper or dashboard.

---

## A few gotchas

- **Counterfactuals need raw features.** The genetic algorithm mutates the *original* columns, not the one-hot encoded ones. So your `FeatureSpec` list must match the columns in `X_train`/`X_test` *before* preprocessing.
- **Rule extraction uses transformed data.** Because XGBoost was trained on the preprocessed matrix, the rules refer to transformed feature names (like `onehot__education_2`). That's fine, but you might want to map them back to human-readable names later.
- **Stability is slow.** Bootstrapping SHAP 30 times on 800 rows takes a while. Start with `n_bootstrap=10` while testing.
- **Subgroup columns must be in the raw DataFrame.** Pass `subgroup_col="residence"` – it reads from `X_test` directly, not from the transformed matrix.

---

## Why this is cool for research

Instead of just saying *"we used SHAP"*, you can now say:

- *"We generated minimal, feasible counterfactuals to identify actionable interventions."*
- *"We distilled the XGBoost ensemble into N human-readable rules with support ≥ 30 and confidence ≥ 0.7."*
- *"We measured feature interactions via TreeSHAP and Friedman's H, revealing that maternal education and wealth interact strongly."*
- *"We assessed explanation stability across 30 bootstrap resamples (mean Spearman ρ = 0.89)."*
- *"We found significant subgroup divergence (JSD = 0.34) between rural and urban populations."*

That's a *methodological contribution*, not just an application. Reviewers will nod approvingly.

---

## Dependencies recap

| Package | Why |
|---------|-----|
| `numpy`, `pandas` | Obviously |
| `scikit-learn` | Pipeline, preprocessing |
| `xgboost` | Your classifier |
| `shap` | Core explainability |
| `lime` | Optional local explanations |
| `matplotlib` | Plots |
| `scipy` | Spearman correlation in stability |

---

## Questions?

If something breaks, it's probably because:
1. Your pipeline doesn't have `preprocessor` and `classifier` steps named exactly like that.
2. Your `FeatureSpec` names don't match your raw DataFrame columns.
3. You forgot to install `scipy` (stability) or `lime` (optional).

Otherwise, hack away and make some cool XAI figures!

## Example Main.py
```python
from xai import (
    XAIExplainer,
    CounterfactualExplainer,
    FeatureSpec,
    RuleExtractor,
    InteractionExplainer,
    StabilityExplainer,
    SubgroupExplainer,
)
from pathlib import Path

xai_dir = Path("outputs/xai")
xai_dir.mkdir(parents=True, exist_ok=True)

# --- vanilla ---
XAIExplainer(pipeline).run_all(X_train, X_test, xai_dir / "baseline")

# --- counterfactuals (declare domain constraints once) ---
specs = [
    FeatureSpec("maternal_education", "categorical", mutable=True,
                direction="increase", categories=[0, 1, 2, 3], cost=1.0),
    FeatureSpec("wealth_quintile", "categorical", mutable=True,
                direction="increase", categories=[1, 2, 3, 4, 5], cost=1.2),
    FeatureSpec("child_age_months", "numeric", mutable=False),
    FeatureSpec("child_sex", "categorical", mutable=False),
    FeatureSpec("improved_water", "categorical", mutable=True,
                categories=[0, 1], cost=0.6),
    # ... one spec per raw feature
]
CounterfactualExplainer(pipeline, specs, target_class=1).explain_many(
    X_test, xai_dir / "counterfactual", max_samples=15
)

# --- rule distillation ---
RuleExtractor(pipeline, top_k=40).persist(
    preprocessor.transform(X_train), y_train.values,
    xai_dir / "rules", min_support=25, min_confidence=0.65
)

# --- interactions ---
inter = InteractionExplainer(pipeline)
inter.shap_interactions(X_test, xai_dir / "interactions")
inter.friedman_h(X_test, xai_dir / "interactions")

# --- stability ---
StabilityExplainer(pipeline).shap_stability(
    X_test, xai_dir / "stability", n_bootstrap=30, top_k=10
)

# --- subgroup ---
SubgroupExplainer(pipeline).shap_divergence(
    X_test, subgroup_col="residence", output_dir=xai_dir / "subgroup"
)
SubgroupExplainer(pipeline).ice_curves(
    X_test, feature="wealth_quintile_2", subgroup_col="residence",
    output_dir=xai_dir / "subgroup"
)
```