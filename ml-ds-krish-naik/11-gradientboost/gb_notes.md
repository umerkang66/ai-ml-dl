# Gradient Boosting — Complete Study Notes

> **Self-contained notes:** After reading this, you will not need any lecture, blog, or book on Gradient Boosting. Everything is here.

---

## Table of Contents

1. [Intuition — What Problem Does It Solve?](#1-intuition)
2. [Ensemble Learning Background](#2-ensemble-learning-background)
3. [The Core Idea of Boosting](#3-the-core-idea-of-boosting)
4. [Gradient Boosting — Step-by-Step](#4-gradient-boosting-step-by-step)
5. [The Math Behind Gradient Boosting](#5-the-math-behind-gradient-boosting)
6. [Worked Example (Regression)](#6-worked-example-regression)
7. [Gradient Boosting for Classification](#7-gradient-boosting-for-classification)
8. [Loss Functions and Their Gradients](#8-loss-functions-and-their-gradients)
9. [Hyperparameters — What They Do and How to Tune](#9-hyperparameters)
10. [Regularization in Gradient Boosting](#10-regularization-in-gradient-boosting)
11. [XGBoost, LightGBM, CatBoost — Variants Explained](#11-popular-variants)
12. [Gradient Boosting vs Other Algorithms](#12-comparison-with-other-algorithms)
13. [Pros and Cons](#13-pros-and-cons)
14. [When to Use Gradient Boosting](#14-when-to-use-gradient-boosting)
15. [Python Implementation (from Scratch + sklearn)](#15-python-implementation)
16. [Common Exam/Interview Questions](#16-common-questions)
17. [Quick-Reference Summary](#17-quick-reference-summary)

---

## 1. Intuition

### The Student Analogy

Imagine you are learning to solve math problems. You take a test, score 60%, and look at your mistakes. You **study specifically those mistakes** and take another test. You score better. You look at what is still wrong, study those again. Repeat.

**Gradient Boosting does exactly this — but with decision trees.**

Each new tree focuses on the errors made by the combination of all previous trees. Errors get smaller and smaller over each round.

### The Residual Repair Shop

Think of the current model's output as a rough estimate. Each new tree is a **repair worker** that looks at what is still wrong (the residuals) and fixes it. After many rounds, the final prediction is the sum of all the repairs.

---

## 2. Ensemble Learning Background

Before understanding Gradient Boosting, you need to know about **ensemble methods** — combining multiple models to get better performance than any single model.

### Two Main Ensemble Strategies

| Strategy     | How It Works                                                                       | Key Algorithm               |
| ------------ | ---------------------------------------------------------------------------------- | --------------------------- |
| **Bagging**  | Train models in **parallel** on random subsets of data. Average their outputs.     | Random Forest               |
| **Boosting** | Train models **sequentially**. Each model corrects the errors of the previous one. | Gradient Boosting, AdaBoost |

### Why Boosting Beats a Single Strong Model

A single deep decision tree overfits. A single shallow tree (stump) underfits. Boosting combines **many shallow trees (weak learners)** — each one barely better than random — into a **strong learner** that generalizes well.

> **Key Theorem (Freund & Schapire):** Any weak learner (slightly better than 50% accuracy) can be boosted into an arbitrarily accurate strong learner.

---

## 3. The Core Idea of Boosting

### AdaBoost vs Gradient Boosting

Both are boosting methods but they differ in _how_ they correct errors:

**AdaBoost:** Re-weights training examples. Misclassified examples get higher weights so the next tree focuses on them.

**Gradient Boosting:** Fits the next tree to the **residual errors** (negative gradient of the loss function) of the current ensemble. It is a more general framework.

### The Functional Gradient Descent View

Gradient Boosting is **gradient descent in function space**, not parameter space.

- Normal gradient descent: adjusts model parameters (weights) to minimize loss.
- Gradient boosting: adjusts the model's **prediction function** step by step, each step adding a new tree that points in the direction that reduces loss.

---

## 4. Gradient Boosting — Step-by-Step

### Algorithm (General Form)

**Given:** Training data `(x₁,y₁), (x₂,y₂), ..., (xₙ,yₙ)`, a differentiable loss function `L(y, F(x))`, number of trees `M`.

**Step 0 — Initialize with a constant prediction:**

```
F₀(x) = argmin_γ Σ L(yᵢ, γ)
```

For MSE loss, this is just the mean of y. For log-loss, it is log(p/(1-p)) where p is the proportion of the positive class.

**Step 1 to M — For each round m:**

**1a. Compute pseudo-residuals (negative gradient):**

```
rᵢₘ = -[∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ)]  for i = 1, 2, ..., n
```

These are the "directions" each prediction needs to move in to reduce the loss.

**1b. Fit a weak learner (decision tree) hₘ(x) to the pseudo-residuals:**

```
Train a regression tree on (x₁, r₁ₘ), ..., (xₙ, rₙₘ)
```

**1c. Find the optimal step size (learning rate for this tree):**

```
γₘ = argmin_γ Σ L(yᵢ, Fₘ₋₁(xᵢ) + γ · hₘ(xᵢ))
```

**1d. Update the model:**

```
Fₘ(x) = Fₘ₋₁(x) + η · γₘ · hₘ(x)
```

Where `η` (eta) is the **learning rate** (shrinkage parameter), a value between 0 and 1.

**Final model:**

```
F(x) = F₀(x) + Σₘ η · γₘ · hₘ(x)
```

---

## 5. The Math Behind Gradient Boosting

### Why "Gradient"?

In standard gradient descent for a neural network:

```
θ ← θ - η · ∇_θ L(θ)
```

We move the **parameters** θ in the direction that decreases loss.

In Gradient Boosting, we instead think of the prediction `F(xᵢ)` as the variable:

```
F(xᵢ) ← F(xᵢ) - η · ∂L(yᵢ, F(xᵢ))/∂F(xᵢ)
```

But we can't directly update F — it must be represented as a sum of trees. So we **train a tree to approximate** those negative gradients, and add it to F.

### Pseudo-Residuals for MSE Loss

Loss: `L(y, F) = ½(y - F)²`

Gradient: `∂L/∂F = -(y - F)`

Negative gradient (pseudo-residual): `rᵢ = y - F(xᵢ)`

So for MSE, pseudo-residuals are literally the ordinary residuals (prediction errors). The tree in each step is fit to predict `(yᵢ - current prediction)`.

### Why Shallow Trees (Weak Learners)?

- A deep tree would perfectly fit the residuals → overfit.
- A shallow tree (depth 1–5) captures a rough correction → regularizes.
- Many small corrections converge better than one big correction.

---

## 6. Worked Example (Regression)

### Dataset

| x   | y   |
| --- | --- |
| 1   | 1.5 |
| 2   | 2.5 |
| 3   | 2.0 |
| 4   | 3.5 |
| 5   | 4.0 |

**Loss:** MSE. **Learning rate η = 0.5. Trees with max_depth = 1 (stumps).**

### Iteration 0 — Initialize

```
F₀(x) = mean(y) = (1.5 + 2.5 + 2.0 + 3.5 + 4.0) / 5 = 2.7
```

### Compute Pseudo-Residuals (Round 1)

```
r₁ = 1.5 - 2.7 = -1.2
r₂ = 2.5 - 2.7 = -0.2
r₃ = 2.0 - 2.7 = -0.7
r₄ = 3.5 - 2.7 =  0.8
r₅ = 4.0 - 2.7 =  1.3
```

### Fit Tree h₁ to Residuals

A stump splits at `x = 3.5`:

- Left leaf (x ≤ 3.5): mean of r₁, r₂, r₃, r₄ = (-1.2 - 0.2 - 0.7 + 0.8)/4 = -0.325
- Right leaf (x > 3.5): mean of r₅ = 1.3

### Update Model

```
F₁(x) = F₀(x) + 0.5 · h₁(x)

For x ≤ 3.5:  F₁(x) = 2.7 + 0.5 × (-0.325) = 2.7 - 0.1625 = 2.5375
For x > 3.5:  F₁(x) = 2.7 + 0.5 × 1.3 = 2.7 + 0.65 = 3.35
```

Predictions after round 1 are closer to true y. Next round will compute new residuals from F₁ and fit another stump. After many rounds, predictions converge to near-perfect values.

---

## 7. Gradient Boosting for Classification

### Binary Classification

For binary classification (y ∈ {0, 1}), the loss function is **binary cross-entropy (log-loss)**:

```
L(y, F) = -[y · log(p) + (1-y) · log(1-p)]

where p = sigmoid(F) = 1 / (1 + e^(-F))
```

**Pseudo-residuals:**

```
rᵢ = yᵢ - pᵢ     (i.e., actual label minus predicted probability)
```

**Initialization:**

```
F₀ = log(p₀ / (1 - p₀))    where p₀ = proportion of class 1 in training set
```

**Leaf values** in each tree are computed differently from regression:

```
γⱼ = Σᵢ∈Rⱼ rᵢ / Σᵢ∈Rⱼ [pᵢ(1 - pᵢ)]
```

**Final output:** Convert log-odds to probability using sigmoid.

### Multi-Class Classification

For K classes, we train **K separate gradient boosting models**, one per class (one-vs-all), using softmax probabilities and multi-class cross-entropy loss.

---

## 8. Loss Functions and Their Gradients

| Task            | Loss Function         | Pseudo-Residuals (Negative Gradient) | Robustness                           |
| --------------- | --------------------- | ------------------------------------ | ------------------------------------ |
| Regression      | MSE: ½(y−F)²          | y − F                                | ✗ Not robust — sensitive to outliers |
| Regression      | MAE: \|y−F\|          | sign(y − F)                          | ✓ Robust                             |
| Regression      | Huber loss            | MSE for small errors, MAE for large  | ✓ Robust                             |
| Binary classif. | Log-loss              | y − sigmoid(F)                       | ~ Moderate                           |
| Multi-class     | Softmax cross-entropy | yₖ − pₖ                              | ~ Moderate                           |
| Ranking         | LambdaRank            | Custom gradient                      | —                                    |

> **Huber Loss** is often preferred in practice — it acts like MSE near zero (smooth gradient) and MAE for large errors (robust to outliers).

---

## 9. Hyperparameters

### The Most Important Ones

#### 1. `n_estimators` (Number of Trees, M)

- **What it does:** More trees = more corrections = lower training error.
- **Too low:** Underfitting (high bias).
- **Too high:** Overfitting + slow inference.
- **Typical range:** 100–1000.
- **Rule:** Higher `n_estimators` always pairs with lower `learning_rate`.

#### 2. `learning_rate` (η, Shrinkage)

- **What it does:** Scales the contribution of each tree.
- **Too high:** Overshoots, unstable.
- **Too low:** Needs many trees to converge (slow, but often better).
- **Typical range:** 0.01–0.3.
- **Golden Rule:** Lower learning rate + more trees = better generalization (but slower to train).

#### 3. `max_depth`

- **What it does:** Controls how complex each tree is.
- **Depth 1:** Stumps (only capture one feature interaction at a time).
- **Depth 3–5:** Captures interactions between a few features (most common choice).
- **Depth > 6:** Risk of overfitting.
- **Typical range:** 3–5.

#### 4. `subsample`

- **What it does:** Fraction of training data randomly sampled to build each tree (without replacement).
- **< 1.0:** Stochastic Gradient Boosting — adds randomness = reduces overfitting.
- **Typical range:** 0.5–0.9.
- **Benefit:** Faster training + regularization (like bagging within boosting).

#### 5. `min_samples_leaf` / `min_child_weight`

- **What it does:** Minimum number of samples in a leaf node.
- **Higher value:** Larger leaves → smoother model → less overfit.
- **Typical range:** 1–20.

#### 6. `max_features` (colsample)

- **What it does:** Fraction of features randomly considered for each split.
- **< 1.0:** Feature subsampling → reduces overfitting, increases speed.
- **Typical range:** 0.5–1.0.

### Hyperparameter Tuning Strategy

```
Step 1: Fix a moderate learning_rate (0.1) and tune n_estimators with early stopping.
Step 2: Tune tree-specific params (max_depth, min_samples_leaf, max_features).
Step 3: Add subsampling (subsample, max_features).
Step 4: Lower learning_rate and increase n_estimators proportionally.
Step 5: Final fine-tuning.
```

---

## 10. Regularization in Gradient Boosting

Gradient Boosting overfits easily with too many trees. Here are regularization techniques:

### 1. Shrinkage (Learning Rate η)

Scaling each tree's contribution by η < 1 prevents any single tree from dominating.

### 2. Subsampling

Using only a fraction of the training data per tree introduces variance (noise) that reduces overfitting.

### 3. Tree Constraints

- `max_depth`: Shallow trees are weaker learners → less overfit.
- `min_samples_leaf`: Forces smoother leaf predictions.

### 4. Early Stopping

Stop adding trees when validation loss stops improving:

```python
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=1000, validation_fraction=0.1,
                                   n_iter_no_change=10, tol=0.001)
```

### 5. L1/L2 Regularization on Leaf Weights (XGBoost)

XGBoost adds explicit regularization terms to the loss:

```
Ω(tree) = γT + ½λ Σ wⱼ²
```

Where T = number of leaves, wⱼ = leaf weight, γ and λ are regularization strength.

---

## 11. Popular Variants

### XGBoost (Extreme Gradient Boosting)

**Key improvements over vanilla Gradient Boosting:**

| Feature            | Vanilla GB                  | XGBoost                           |
| ------------------ | --------------------------- | --------------------------------- |
| Loss approximation | First-order (gradient only) | Second-order (gradient + Hessian) |
| Regularization     | None built-in               | L1 + L2 on leaf weights           |
| Missing values     | Must impute                 | Handles natively                  |
| Tree building      | Level-wise                  | Depth-wise (faster)               |
| Parallelism        | Sequential                  | Column-wise parallel              |
| Speed              | Slow                        | Much faster                       |

**Second-order approximation:**
XGBoost uses both gradient (gᵢ) and Hessian (hᵢ) of the loss:

```
gᵢ = ∂L(yᵢ, F(xᵢ))/∂F    (first derivative)
hᵢ = ∂²L(yᵢ, F(xᵢ))/∂F²  (second derivative)
```

This gives more accurate step sizes → faster convergence.

### LightGBM (Light Gradient Boosting Machine)

**Key innovations:**

- **Leaf-wise tree growth** (vs. level-wise): Grows the leaf with the highest loss reduction, not all leaves at the same depth. → Fewer nodes, better accuracy.
- **GOSS (Gradient-based One-Side Sampling):** Keeps all high-gradient samples, randomly drops low-gradient samples → faster with little accuracy loss.
- **EFB (Exclusive Feature Bundling):** Bundles sparse features that rarely take non-zero values together → reduces feature dimensionality.
- **Result:** 10–20× faster than XGBoost on large datasets.

### CatBoost (Categorical Boosting — Yandex)

**Key innovations:**

- **Ordered Boosting:** Avoids target leakage when encoding categorical features.
- **Symmetric trees:** Identical split conditions at each level → faster and less overfit.
- **Native categorical support:** No need to one-hot encode.
- **Best for:** Datasets with many categorical features.

### Comparison Summary

|             | sklearn GB        | XGBoost              | LightGBM       | CatBoost         |
| ----------- | ----------------- | -------------------- | -------------- | ---------------- |
| Speed       | Slowest           | Fast                 | Fastest        | Fast             |
| Categorical | No                | No                   | Limited        | Native           |
| Memory      | High              | Moderate             | Low            | Moderate         |
| Accuracy    | Good              | Great                | Great          | Great            |
| GPU Support | No                | Yes                  | Yes            | Yes              |
| Best use    | Learning/baseline | Tabular competitions | Large datasets | Categorical data |

---

## 12. Comparison with Other Algorithms

### Gradient Boosting vs Random Forest

| Aspect             | Gradient Boosting               | Random Forest          |
| ------------------ | ------------------------------- | ---------------------- |
| Trees              | Sequential (each corrects last) | Parallel (independent) |
| Bias-Variance      | Low bias, can overfit           | Balanced bias-variance |
| Speed (training)   | Slower                          | Faster                 |
| Speed (inference)  | Slower                          | Fast                   |
| Tuning difficulty  | More hyperparameters            | Fewer hyperparameters  |
| Noisy data         | Sensitive                       | More robust            |
| Accuracy (tabular) | Usually better                  | Slightly lower         |

### Gradient Boosting vs Neural Networks

| Aspect                | Gradient Boosting | Neural Networks |
| --------------------- | ----------------- | --------------- |
| Tabular data          | Often superior    | Often inferior  |
| Image/text/audio      | Not suitable      | Dominates       |
| Training data needed  | Less              | More            |
| Interpretability      | Moderate          | Low             |
| Feature engineering   | Needs it          | Can learn it    |
| Hyperparameter tuning | Moderate          | Extensive       |

---

## 13. Pros and Cons

### Pros

- **State-of-the-art on tabular data** — Gradient Boosting variants win most Kaggle competitions on structured data.
- **Handles mixed data types** — Numbers and categories (with encoding).
- **No feature scaling needed** — Trees are invariant to monotone transformations of features.
- **Handles missing values** — XGBoost/LightGBM natively.
- **Built-in feature importance** — Easy to interpret which features matter.
- **Flexible loss functions** — Can optimize any differentiable loss.
- **Captures non-linear relationships and interactions automatically.**

### Cons

- **Prone to overfitting** — Requires careful hyperparameter tuning.
- **Slow to train** (vanilla GB) — Each tree must wait for the previous one.
- **Memory intensive** — Stores all trees.
- **Harder to tune** than Random Forest — More hyperparameters.
- **Sensitive to outliers** with MSE loss — Use Huber loss instead.
- **Not ideal for real-time training** — Adding new data requires retraining from scratch.
- **Black box** — Hard to explain individual predictions (use SHAP values).

---

## 14. When to Use Gradient Boosting

### Use Gradient Boosting when:

- You have **structured/tabular data** (rows and columns).
- You want **best-in-class accuracy** and have time to tune.
- Your dataset has **non-linear relationships and complex feature interactions**.
- You have **medium to large datasets** (thousands to millions of rows).
- You need **feature importance** for interpretability.

### Don't Use Gradient Boosting when:

- Data is **image, audio, or text** → use neural networks.
- You need **real-time retraining** with streaming data.
- Dataset is **very small** (< few hundred rows) → simpler models may generalize better.
- You need an **extremely fast baseline** → try Random Forest or Logistic Regression first.
- **Interpretability** is critical and you need simple rules → use a Decision Tree.

---

## 15. Python Implementation

### From Scratch (Regression, MSE Loss)

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressorScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.F0 = None

    def fit(self, X, y):
        # Step 0: Initialize with mean
        self.F0 = np.mean(y)
        F = np.full(len(y), self.F0)

        for m in range(self.n_estimators):
            # Compute pseudo-residuals (negative gradient of MSE)
            residuals = y - F

            # Fit a decision tree to residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Update predictions
            F += self.learning_rate * tree.predict(X)

        return self

    def predict(self, X):
        F = np.full(X.shape[0], self.F0)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F


# Test it
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = GradientBoostingRegressorScratch(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, preds):.2f}")
```

### Using scikit-learn

```python
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.datasets import make_regression, make_classification

# --- REGRESSION ---
X, y = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

reg = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    min_samples_leaf=5,
    random_state=42
)
reg.fit(X_train, y_train)
print(f"Regression MSE: {mean_squared_error(y_test, reg.predict(X_test)):.2f}")

# --- CLASSIFICATION ---
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
clf.fit(X_train, y_train)
print(f"Accuracy: {accuracy_score(y_test, clf.predict(X_test)):.4f}")

# Feature importance
import pandas as pd
importances = pd.Series(clf.feature_importances_,
                         index=[f"f{i}" for i in range(10)])
print(importances.sort_values(ascending=False))
```

### Using XGBoost

```python
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,    # fraction of features per tree
    reg_alpha=0.1,           # L1 regularization
    reg_lambda=1.0,          # L2 regularization
    use_label_encoder=False,
    eval_metric='logloss',
    early_stopping_rounds=20,
    random_state=42
)

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)

print(f"XGBoost Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
print(f"Best iteration: {model.best_iteration}")
```

### Using LightGBM

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    num_leaves=31,           # LightGBM-specific: controls leaf-wise growth
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(50)])

print(f"LightGBM Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
```

---

## 16. Common Questions

**Q1. What is the difference between AdaBoost and Gradient Boosting?**

AdaBoost corrects errors by **re-weighting samples** — misclassified samples get higher weights. Gradient Boosting corrects errors by **fitting new trees to the residuals (negative gradients)**. Gradient Boosting is more general — by changing the loss function, it can handle regression, classification, and ranking. AdaBoost is mainly for classification.

---

**Q2. Why are shallow trees (weak learners) used instead of deep trees?**

Shallow trees have high bias and low variance. A deep tree would perfectly fit the residuals and overfit. Many shallow trees combined can capture complex patterns while staying generalized. The bias reduces with each boosting round; the low variance of each tree keeps the ensemble stable.

---

**Q3. What are pseudo-residuals? Are they the same as actual residuals?**

For MSE loss, pseudo-residuals equal actual residuals: `rᵢ = yᵢ - F(xᵢ)`.

For other loss functions (like log-loss), pseudo-residuals are the **negative gradient of the loss** with respect to the prediction — they indicate how much and in which direction each prediction needs to change. The name "pseudo" distinguishes them from classical regression residuals.

---

**Q4. What happens if learning rate is set to 1.0?**

Each tree's contribution is added without shrinkage. This is aggressive — the model may overshoot and oscillate, leading to overfitting. A lower learning rate forces the model to take small steps, requiring more trees but usually achieving better generalization.

---

**Q5. Why does Gradient Boosting work better than a single decision tree?**

A single deep tree overfits; a single shallow tree underfits. Gradient Boosting creates an additive model where each tree handles a small piece of the error. The **ensemble of many weak learners** reduces both bias (through more trees) and variance (through shallow trees and regularization).

---

**Q6. How does XGBoost handle missing values?**

During training, for each split, XGBoost tries sending missing values to both the left and right child and picks the direction that reduces loss most. This **learned default direction** is stored and used at inference time. No imputation is needed.

---

**Q7. What is early stopping and why is it used?**

Early stopping monitors validation loss after each tree. If the validation loss does not improve for N consecutive rounds (`n_iter_no_change` or `early_stopping_rounds`), training stops. This prevents overfitting without having to manually set the exact `n_estimators`.

---

**Q8. What is feature importance in Gradient Boosting?**

Three types:

- **Split importance (gain):** Total reduction in loss achieved by splits on a feature across all trees.
- **Split count (frequency):** Number of times a feature is used to split across all trees.
- **SHAP values:** Game-theory-based, gives each feature a fair contribution to each prediction. More reliable and consistent.

---

**Q9. Is Gradient Boosting sensitive to outliers?**

Yes, with MSE loss — outliers have large residuals which dominate training. Use **Huber loss** or **MAE loss** for robustness to outliers.

---

**Q10. Can Gradient Boosting be parallelized?**

Not across trees (they must be built sequentially). However, XGBoost and LightGBM parallelize **within each tree** — finding the best split across columns can be done in parallel. LightGBM also uses GOSS and EFB to reduce data and feature size, making it faster.

---

## 17. Quick-Reference Summary

### The Algorithm in Plain English

1. Start with a simple prediction (e.g., mean of y).
2. Calculate how wrong you are on each example (pseudo-residuals = negative gradients).
3. Train a shallow tree to predict those errors.
4. Add a fraction (learning rate) of that tree's predictions to your current model.
5. Repeat steps 2–4 for M rounds.
6. Final prediction = sum of all trees' contributions.

### Key Formulas

| Concept             | Formula                       |
| ------------------- | ----------------------------- |
| Pseudo-residual     | `rᵢ = -∂L(yᵢ, F(xᵢ))/∂F(xᵢ)`  |
| MSE pseudo-residual | `rᵢ = yᵢ - F(xᵢ)`             |
| Model update        | `Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x)` |
| Final model         | `F(x) = F₀ + Σ η · hₘ(x)`     |
| Log-loss residual   | `rᵢ = yᵢ - sigmoid(F(xᵢ))`    |

### Hyperparameters Cheat Sheet

| Hyperparameter     | Effect of Increasing       | Typical Value |
| ------------------ | -------------------------- | ------------- |
| `n_estimators`     | ↓ bias, risk overfit       | 100–1000      |
| `learning_rate`    | ↑ step size, risk overfit  | 0.01–0.3      |
| `max_depth`        | ↑ complexity, risk overfit | 3–5           |
| `subsample`        | ↑ noise, ↓ overfit         | 0.5–0.9       |
| `min_samples_leaf` | ↑ smoothing, ↓ overfit     | 1–20          |
| `max_features`     | ↑ speed, ↓ overfit         | 0.5–1.0       |

### One-Line Comparisons

- **GB vs AdaBoost:** GB fits trees to residuals; AdaBoost re-weights samples.
- **GB vs Random Forest:** GB is sequential + lower bias; RF is parallel + lower variance.
- **XGBoost vs LightGBM:** XGBoost uses second-order gradients; LightGBM uses leaf-wise growth and is faster on large data.
- **LightGBM vs CatBoost:** LightGBM is fastest; CatBoost handles categorical features natively.

---

_End of Notes — You now have everything you need to understand, implement, and apply Gradient Boosting._
