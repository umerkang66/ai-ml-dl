# AdaBoost (Adaptive Boosting) — Complete Notes

> **Goal of these notes:** After reading this, you will fully understand AdaBoost — what it is, why it exists, how it works mathematically, how to implement it, and when to use it. No lecture or external resource needed.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Motivation — Why AdaBoost?](#2-motivation--why-adaboost)
3. [Core Intuition](#3-core-intuition)
4. [Key Terminology](#4-key-terminology)
5. [AdaBoost Algorithm — Step by Step](#5-adaboost-algorithm--step-by-step)
6. [Complete Worked Example](#6-complete-worked-example)
7. [The Math Behind AdaBoost](#7-the-math-behind-adaboost)
8. [AdaBoost vs Other Algorithms](#8-adaboost-vs-other-algorithms)
9. [Implementation in Python](#9-implementation-in-python)
10. [Advantages and Disadvantages](#10-advantages-and-disadvantages)
11. [Common Exam Questions & Answers](#11-common-exam-questions--answers)
12. [Quick Revision Cheatsheet](#12-quick-revision-cheatsheet)

---

## 1. Prerequisites

Before diving in, you should be comfortable with:

- **Decision Trees** — especially shallow ones (depth = 1, called stumps)
- **Classification** — understanding of class labels (+1, -1)
- **Weighted averages** — computing weighted sums
- **Logarithms** — used in the weight update formula

If you know these, you are ready.

---

## 2. Motivation — Why AdaBoost?

### The Problem with Weak Learners

A **weak learner** is a model that is only slightly better than random guessing. For example, a decision stump (a tree with just one split) might get 60% accuracy on a classification task — not great, but better than the 50% you'd expect from flipping a coin.

The natural question is: **Can we combine many weak learners to build a strong learner?**

The answer is **yes**, and that is exactly what **Boosting** does.

### The Problem with Simple Averaging (Bagging)

You may already know **Bagging** (e.g., Random Forest), which trains multiple models on random subsets and averages their predictions. Bagging reduces **variance** (overfitting). However, it does not specifically focus on the hard-to-classify examples.

**Boosting is different:** it trains models **sequentially**, and each new model focuses more on the examples the previous model got wrong.

---

## 3. Core Intuition

Imagine you are a student taking practice exams. After each exam:

1. You identify which questions you got wrong.
2. You **focus more** on those questions next time.
3. After many rounds, you are good at all types of questions.

AdaBoost works exactly like this:

- Each training example has a **weight** (initially equal).
- After each weak learner is trained, **misclassified examples get higher weights**.
- The next weak learner focuses more on the heavy-weighted (hard) examples.
- At the end, all weak learners **vote**, but more accurate ones get a **louder vote**.

---

## 4. Key Terminology

| Term                               | Meaning                                                        |
| ---------------------------------- | -------------------------------------------------------------- |
| **Weak learner**                   | A model slightly better than random (e.g., decision stump)     |
| **Strong learner**                 | The final combined model (ensemble)                            |
| **Boosting**                       | Sequential training where each model corrects the previous one |
| **Sample weight** $w_i$            | How much attention to pay to example $i$                       |
| **Learner weight** $\alpha_t$      | How much to trust weak learner $t$ in the final vote           |
| **Decision stump**                 | A decision tree with depth = 1 (one split, two leaves)         |
| **Ensemble**                       | A collection of models combined together                       |
| **Weighted error** $\varepsilon_t$ | The total weight of misclassified examples                     |

---

## 5. AdaBoost Algorithm — Step by Step

### Setup

- Dataset: $N$ training examples $(x_1, y_1), (x_2, y_2), \ldots, (x_N, y_N)$
- Labels: $y_i \in \{-1, +1\}$ (binary classification)
- Number of rounds (weak learners): $T$

---

### Step 0 — Initialize Sample Weights

Give every training example an equal weight:

$$w_i^{(1)} = \frac{1}{N} \quad \text{for } i = 1, 2, \ldots, N$$

All examples start equally important. Total weight sums to 1.

---

### Step 1 — Train a Weak Learner on Weighted Data

At round $t$, train a weak learner $h_t(x)$ on the training data, **giving more attention to examples with higher weights**.

In practice, this means:

- Either train the learner on a **weighted loss function**, or
- **Resample** the data according to the weights (higher-weight samples are more likely to be picked)

The weak learner outputs $h_t(x) \in \{-1, +1\}$.

---

### Step 2 — Compute Weighted Error

Calculate how badly $h_t$ performed, weighted by the current sample weights:

$$\varepsilon_t = \sum_{i=1}^{N} w_i^{(t)} \cdot \mathbf{1}[h_t(x_i) \neq y_i]$$

Where $\mathbf{1}[\cdot]$ is 1 if the condition is true, 0 otherwise.

**Interpretation:**

- $\varepsilon_t = 0$ → perfect classifier (no errors)
- $\varepsilon_t = 0.5$ → no better than random guessing
- $\varepsilon_t > 0.5$ → worse than random (flip its predictions!)

> **Note:** AdaBoost requires $\varepsilon_t < 0.5$. If a learner is worse than random, you can simply flip its output.

---

### Step 3 — Compute Learner Weight (Alpha)

Assign a weight $\alpha_t$ to the weak learner based on how good it is:

$$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \varepsilon_t}{\varepsilon_t}\right)$$

**Understanding $\alpha_t$:**

| Weighted Error $\varepsilon_t$ | $\alpha_t$ value | Meaning                       |
| ------------------------------ | ---------------- | ----------------------------- |
| Close to 0 (few errors)        | Large positive   | Strong learner, trusted a lot |
| 0.5 (random)                   | 0                | Ignored completely            |
| Close to 1 (many errors)       | Large negative   | Predictions reversed          |

This is the key idea: **better learners get more say in the final decision.**

---

### Step 4 — Update Sample Weights

Now adjust the weights so misclassified examples get more attention:

$$w_i^{(t+1)} = w_i^{(t)} \cdot e^{-\alpha_t \cdot y_i \cdot h_t(x_i)}$$

Then **normalize** so weights sum to 1:

$$w_i^{(t+1)} \leftarrow \frac{w_i^{(t+1)}}{\sum_{j=1}^{N} w_j^{(t+1)}}$$

**Why does this work?**

- If example $i$ is **correctly classified**: $y_i \cdot h_t(x_i) = +1$, so $e^{-\alpha_t \cdot (+1)} = e^{-\alpha_t} < 1$ → weight **decreases**
- If example $i$ is **misclassified**: $y_i \cdot h_t(x_i) = -1$, so $e^{-\alpha_t \cdot (-1)} = e^{+\alpha_t} > 1$ → weight **increases**

---

### Step 5 — Repeat

Go back to Step 1 and train the next weak learner using the new weights. Repeat for $T$ rounds.

---

### Step 6 — Final Prediction (Weighted Majority Vote)

The final strong classifier is:

$$H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t \cdot h_t(x)\right)$$

Each weak learner casts a weighted vote. The sign of the total determines the final class label.

---

### Algorithm Summary (Pseudocode)

```
Input: Training data {(x1,y1), ..., (xN,yN)}, T rounds

Initialize: w_i = 1/N for all i

For t = 1 to T:
    1. Train weak learner h_t using weights w_i
    2. Compute weighted error:
          ε_t = Σ w_i * 1[h_t(x_i) ≠ y_i]
    3. Compute learner weight:
          α_t = 0.5 * ln((1 - ε_t) / ε_t)
    4. Update weights:
          w_i ← w_i * exp(-α_t * y_i * h_t(x_i))
    5. Normalize weights so they sum to 1

Output: H(x) = sign( Σ α_t * h_t(x) )
```

---

## 6. Complete Worked Example

Let's trace through AdaBoost manually on a tiny dataset.

### Dataset

| Example | Feature $x$ | True Label $y$ |
| ------- | ----------- | -------------- |
| 1       | 1           | +1             |
| 2       | 2           | +1             |
| 3       | 3           | +1             |
| 4       | 4           | -1             |
| 5       | 5           | -1             |
| 6       | 6           | -1             |

$N = 6$, so initial weights: $w_i^{(1)} = 1/6 \approx 0.167$ for all $i$.

---

### Round 1

**Train stump $h_1$:** The best stump splits at $x < 3.5$:

- Predict $+1$ if $x \leq 3.5$
- Predict $-1$ if $x > 3.5$

**Predictions:**

| Example | $h_1(x)$ | $y$ | Correct? | Weight |
| ------- | -------- | --- | -------- | ------ |
| 1       | +1       | +1  | ✓        | 0.167  |
| 2       | +1       | +1  | ✓        | 0.167  |
| 3       | +1       | +1  | ✓        | 0.167  |
| 4       | -1       | -1  | ✓        | 0.167  |
| 5       | -1       | -1  | ✓        | 0.167  |
| 6       | -1       | -1  | ✓        | 0.167  |

**Weighted error:** $\varepsilon_1 = 0$ — perfect!

**Learner weight:** $\alpha_1 = \frac{1}{2}\ln\left(\frac{1}{0}\right)$ — undefined at 0, so in practice a small minimum error $\varepsilon_{min}$ is used. Let's use a more interesting example.

---

### A More Realistic Round

Suppose stump $h_1$ misclassifies examples 4 and 5:

| Example | $h_1(x)$ | $y$ | Correct? | Weight |
| ------- | -------- | --- | -------- | ------ |
| 1       | +1       | +1  | ✓        | 0.167  |
| 2       | +1       | +1  | ✓        | 0.167  |
| 3       | +1       | +1  | ✓        | 0.167  |
| 4       | +1       | -1  | ✗        | 0.167  |
| 5       | +1       | -1  | ✗        | 0.167  |
| 6       | -1       | -1  | ✓        | 0.167  |

**Step 2 — Weighted Error:**

$$\varepsilon_1 = 0.167 + 0.167 = 0.333$$

**Step 3 — Learner Weight:**

$$\alpha_1 = \frac{1}{2}\ln\left(\frac{1 - 0.333}{0.333}\right) = \frac{1}{2}\ln(2) = \frac{1}{2}(0.693) = 0.347$$

**Step 4 — Update Weights:**

For correct examples (1, 2, 3, 6): multiply by $e^{-0.347} = 0.707$

$$w_{\text{correct}} = 0.167 \times 0.707 = 0.118$$

For wrong examples (4, 5): multiply by $e^{+0.347} = 1.415$

$$w_{\text{wrong}} = 0.167 \times 1.415 = 0.236$$

**Normalize** (sum = $4 \times 0.118 + 2 \times 0.236 = 0.472 + 0.472 = 0.944$):

$$w_{\text{correct}}^{(2)} = \frac{0.118}{0.944} = 0.125$$

$$w_{\text{wrong}}^{(2)} = \frac{0.236}{0.944} = 0.250$$

**New weights for Round 2:**

| Example | New Weight                           |
| ------- | ------------------------------------ |
| 1       | 0.125                                |
| 2       | 0.125                                |
| 3       | 0.125                                |
| 4       | **0.250** ← harder, gets more weight |
| 5       | **0.250** ← harder, gets more weight |
| 6       | 0.125                                |

Now the next stump will focus more on examples 4 and 5. This is AdaBoost in action!

---

## 7. The Math Behind AdaBoost

### Why Exponential Weight Update?

AdaBoost minimizes the **exponential loss function**:

$$L = \sum_{i=1}^{N} e^{-y_i F(x_i)}$$

Where $F(x) = \sum_{t=1}^{T} \alpha_t h_t(x)$ is the cumulative ensemble score.

At each round, we add a new term $\alpha_t h_t$ that minimizes this loss. The weight update formula directly follows from taking derivatives of this loss with respect to $\alpha_t$.

### Why Alpha Has the Form It Does

Minimizing the exponential loss with respect to $\alpha_t$:

$$\frac{\partial L}{\partial \alpha_t} = 0 \implies \alpha_t = \frac{1}{2} \ln\left(\frac{1-\varepsilon_t}{\varepsilon_t}\right)$$

This derivation is elegant — the formula arises naturally from the math, not from guesswork.

### Convergence Guarantee

AdaBoost has a provable bound on training error. After $T$ rounds:

$$\text{Training Error} \leq e^{-2 \sum_{t=1}^{T} \gamma_t^2}$$

Where $\gamma_t = 0.5 - \varepsilon_t$ is the "edge" of each weak learner over random guessing.

As long as each weak learner has even a tiny edge ($\gamma_t > 0$), the training error **decreases exponentially** as we add more rounds. This is a theoretical guarantee.

---

## 8. AdaBoost vs Other Algorithms

### AdaBoost vs Bagging (Random Forest)

| Property                | AdaBoost                         | Bagging (Random Forest) |
| ----------------------- | -------------------------------- | ----------------------- |
| Training order          | Sequential (depends on previous) | Parallel (independent)  |
| Focus                   | Hard examples (weighted)         | Random subsets          |
| What it reduces         | Bias                             | Variance                |
| Overfitting sensitivity | More sensitive to noise          | Less sensitive          |
| Speed                   | Slower (sequential)              | Faster (parallelizable) |
| Base learner            | Usually stumps                   | Full trees              |

### AdaBoost vs Gradient Boosting

| Property                 | AdaBoost           | Gradient Boosting                 |
| ------------------------ | ------------------ | --------------------------------- |
| How errors are corrected | Reweight examples  | Fit residuals (gradients)         |
| Loss function            | Fixed: exponential | Flexible: any differentiable loss |
| Flexibility              | Less flexible      | More flexible                     |
| Examples                 | AdaBoost           | XGBoost, LightGBM, CatBoost       |

> **Think of it this way:** AdaBoost reweights the data. Gradient Boosting fits the _residuals_ directly. Gradient Boosting is the generalized framework; AdaBoost is a special case.

### AdaBoost vs Single Decision Tree

| Property         | AdaBoost               | Single Decision Tree |
| ---------------- | ---------------------- | -------------------- |
| Accuracy         | Generally higher       | Moderate             |
| Interpretability | Low (many models)      | High (single tree)   |
| Training time    | More                   | Less                 |
| Overfitting      | Surprisingly resistant | Moderate             |

---

## 9. Implementation in Python

### Using Scikit-learn (Recommended for Practice)

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Create a sample dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    random_state=42
)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build AdaBoost with decision stumps
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # Weak learner = stump
    n_estimators=100,      # Number of rounds T
    learning_rate=1.0,     # Shrinks alpha (regularization)
    algorithm='SAMME.R',   # Modern variant using probability
    random_state=42
)

ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

### From-Scratch Implementation

```python
import numpy as np

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []          # Learner weights
        self.stumps = []          # Weak learners

    def _train_stump(self, X, y, weights):
        """Train a decision stump: find the best feature and threshold."""
        n_samples, n_features = X.shape
        best_error = float('inf')
        best_stump = {}

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                for polarity in [1, -1]:
                    # Predict based on threshold
                    predictions = np.where(
                        polarity * X[:, feature] < polarity * threshold, 1, -1
                    )
                    # Weighted error
                    error = np.sum(weights[predictions != y])
                    if error < best_error:
                        best_error = error
                        best_stump = {
                            'feature': feature,
                            'threshold': threshold,
                            'polarity': polarity
                        }
        return best_stump, best_error

    def _stump_predict(self, X, stump):
        """Make predictions using a decision stump."""
        feature = stump['feature']
        threshold = stump['threshold']
        polarity = stump['polarity']
        return np.where(polarity * X[:, feature] < polarity * threshold, 1, -1)

    def fit(self, X, y):
        n_samples = X.shape[0]
        # Step 0: Initialize weights
        weights = np.ones(n_samples) / n_samples

        for t in range(self.n_estimators):
            # Step 1: Train weak learner
            stump, error = self._train_stump(X, y, weights)

            # Clip error to avoid log(0)
            error = np.clip(error, 1e-10, 1 - 1e-10)

            # Step 3: Compute alpha
            alpha = 0.5 * np.log((1 - error) / error)

            # Step 4: Update weights
            predictions = self._stump_predict(X, stump)
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)  # Normalize

            self.alphas.append(alpha)
            self.stumps.append(stump)

    def predict(self, X):
        # Weighted vote
        total = sum(
            alpha * self._stump_predict(X, stump)
            for alpha, stump in zip(self.alphas, self.stumps)
        )
        return np.sign(total)
```

### Visualizing Performance vs Number of Estimators

```python
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_errors, test_errors = [], []
estimator_range = range(1, 201, 10)

for n in estimator_range:
    ada = AdaBoostClassifier(n_estimators=n, random_state=42)
    ada.fit(X_train, y_train)
    train_errors.append(1 - accuracy_score(y_train, ada.predict(X_train)))
    test_errors.append(1 - accuracy_score(y_test, ada.predict(X_test)))

plt.figure(figsize=(10, 5))
plt.plot(estimator_range, train_errors, label='Train Error', color='blue')
plt.plot(estimator_range, test_errors, label='Test Error', color='red')
plt.xlabel('Number of Estimators')
plt.ylabel('Error Rate')
plt.title('AdaBoost: Error vs Number of Estimators')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 10. Advantages and Disadvantages

### Advantages

- **Simple to implement** — the algorithm is clean and well-defined.
- **No need for feature scaling** — since it uses tree-based weak learners.
- **Resistant to overfitting** — surprisingly robust compared to individual trees, especially with small T.
- **Interpretable components** — each weak learner (stump) is interpretable.
- **Strong theoretical guarantees** — training error bounds are proven.
- **Works with any weak learner** — not limited to decision stumps; any classifier works.
- **Feature importance** — can derive feature importance from ensemble.

### Disadvantages

- **Sensitive to noisy data and outliers** — because it up-weights misclassified examples. Mislabeled or noisy examples will get very high weights and hurt performance.
- **Slow training** — models are trained sequentially; cannot be parallelized easily.
- **Sensitive to hyperparameter T** — too many rounds can lead to overfitting on noisy datasets.
- **Requires weak learners to be "weak enough"** — if base learners are too strong (e.g., deep trees), boosting loses its advantage.
- **Binary classification focus** — the original algorithm is designed for binary labels ($\{-1, +1\}$); multi-class extensions (SAMME) exist but are more complex.

---

## 11. Common Exam Questions & Answers

**Q1: What is the difference between boosting and bagging?**

Bagging trains models **in parallel** on random subsets of data to reduce variance. Boosting trains models **sequentially**, where each model corrects the errors of the previous one, primarily reducing bias. AdaBoost is a boosting method.

---

**Q2: Why does AdaBoost use decision stumps as weak learners?**

Decision stumps are simple and fast to train, making them ideal weak learners. They are slightly better than random, which is all boosting requires. Using deeper trees would introduce too much complexity per learner.

---

**Q3: What happens if $\varepsilon_t = 0.5$?**

$\alpha_t = \frac{1}{2}\ln\left(\frac{0.5}{0.5}\right) = \frac{1}{2}\ln(1) = 0$. The learner is given **zero weight** and contributes nothing to the final vote. This makes sense — a model no better than random should be ignored.

---

**Q4: What happens if $\varepsilon_t > 0.5$?**

$\alpha_t$ becomes **negative**. This means the model's predictions are flipped before voting. A model that is consistently wrong is actually informative — just use the opposite of what it predicts!

---

**Q5: Can AdaBoost overfit?**

Yes, but it is more resistant to overfitting than you might expect. With low-noise data, adding more rounds often keeps improving test accuracy even after training error hits 0. However, with **noisy data**, AdaBoost can overfit because it keeps increasing the weights of mislabeled examples.

---

**Q6: What loss function does AdaBoost minimize?**

AdaBoost minimizes the **exponential loss** $L = \sum_i e^{-y_i F(x_i)}$. This loss penalizes misclassified examples exponentially harshly, which is why it is sensitive to outliers.

---

**Q7: How is AdaBoost different from Gradient Boosting?**

AdaBoost corrects errors by **reweighting the training data**. Gradient Boosting corrects errors by fitting new models to the **residuals (negative gradients)** of the loss function. Gradient Boosting is a generalization of AdaBoost that works with any differentiable loss function.

---

**Q8: How do you handle multi-class problems with AdaBoost?**

The original AdaBoost handles binary classification. For multi-class problems, the **SAMME** algorithm (Stagewise Additive Modeling using a Multi-class Exponential loss) is used, which is the default in scikit-learn. SAMME.R uses predicted probabilities rather than class labels.

---

**Q9: Derive the $\alpha$ formula conceptually.**

We want to give high $\alpha$ to accurate models and low $\alpha$ to bad ones. We also need $\alpha = 0$ when $\varepsilon = 0.5$ (random). The logarithm $\ln\left(\frac{1-\varepsilon}{\varepsilon}\right)$ satisfies this: it is 0 when $\varepsilon = 0.5$, positive when $\varepsilon < 0.5$, and negative when $\varepsilon > 0.5$. The factor of $\frac{1}{2}$ comes from the mathematical derivation of minimizing exponential loss.

---

**Q10: What is the role of the learning rate in scikit-learn's AdaBoostClassifier?**

The learning rate $\eta$ (0 < $\eta$ ≤ 1) **shrinks the contribution of each learner**: $\alpha_t \leftarrow \eta \cdot \alpha_t$. Smaller learning rates require more estimators but often yield better generalization. It is a regularization technique.

---

## 12. Quick Revision Cheatsheet

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ADABOOST — QUICK REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 TYPE        : Ensemble method (Boosting)
 TASK        : Binary classification (extensions exist)
 BASE LEARNER: Decision stumps (depth=1 trees)
 TRAINING    : Sequential (one model at a time)

 KEY FORMULAS:
   Initial weights     → w_i = 1/N
   Weighted error      → ε_t = Σ w_i · 1[h_t(x_i) ≠ y_i]
   Learner weight      → α_t = ½ ln((1 - ε_t) / ε_t)
   Weight update       → w_i ← w_i · exp(-α_t · y_i · h_t(x_i))
   Final prediction    → H(x) = sign(Σ α_t · h_t(x))

 INTUITION:
   Misclassified → weight UP
   Correctly classified → weight DOWN
   Accurate learner → high α (loud vote)
   Inaccurate learner → low α (quiet vote)

 PROS:
   + Simple, theoretically grounded
   + No feature scaling needed
   + Resistant to overfitting (clean data)
   + Works with any weak learner

 CONS:
   - Sensitive to noise and outliers
   - Sequential = slow
   - Requires ε_t < 0.5

 SCIKIT-LEARN:
   AdaBoostClassifier(
       estimator=DecisionTreeClassifier(max_depth=1),
       n_estimators=100,
       learning_rate=1.0,
       algorithm='SAMME.R'
   )

 LOSS FUNCTION : Exponential loss
 RELATED       : Gradient Boosting (generalization)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

_End of AdaBoost Notes. You now have everything you need — concept, math, examples, code, and exam prep — in one place._
