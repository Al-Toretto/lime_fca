# PS-LIME: Local Black-Box Explanations with Pattern Structures

A big research-oriented homework for the **Ordered Sets for Data Analysis**
course.

This project is inspired by LIME: explain a complex model locally, around one
selected instance. In classical LIME, an explainer generates perturbed samples,
asks the black-box model to label them, and fits a simple local surrogate model,
often a sparse linear model.

In this project, the local surrogate is replaced by an FCA-based or
pattern-structure-based model. The explanation is not just a list of feature
weights. The main explanation object is a local concept lattice, or a meaningful
fragment of it. Its concepts can be used as local classifiers, while the lattice
order lets us inspect more general and more specific explanations.

## Task Description

You need to choose a binary classification task, train a black-box model, and
build local explanations for several of its predictions using Formal Concept
Analysis or pattern structures.

Minimum pipeline:

1. Train a black-box classifier `f`.
2. Choose a test instance `x` to explain.
3. Generate a local neighborhood `N(x)` using perturbations.
4. Get pseudo-labels `f(z)` for every perturbed sample `z`.
5. Build a local formal context or a local pattern structure.
6. Build a local concept lattice or a restricted lattice fragment.
7. Select concepts that can serve as local classifiers.
8. Compare the explanation with LIME, SHAP, or a local decision tree.

Main research question:

```text
Can the local linear surrogate in LIME be replaced by a local concept lattice
while preserving fidelity to the black-box model and obtaining a more structured
explanation?
```

## Learning Goals

After completing the project, you should be able to:

1. Explain the difference between black-box model quality and local explanation
   quality.
2. Generate local perturbations and pseudo-label them with a black-box model.
3. Build a local formal context or a local pattern structure.
4. Compute local concepts and analyze their order.
5. Select pure concepts or alpha-weak concepts as local classifiers.
6. Evaluate local fidelity, stability, and explanation compactness.
7. Interpret movement through the lattice: more general and more specific
   classifiers.

## To-do List

### Basic solution

- [ ] Find an open dataset for binary classification.
- [ ] Perform EDA and preprocessing.
- [ ] Train a black-box model: Random Forest, XGBoost, nonlinear SVM, or a small
      neural network.
- [ ] Choose 5-10 test instances to explain.
- [ ] Implement local perturbation generation around each instance.
- [ ] Obtain black-box pseudo-labels for the perturbed samples.
- [ ] Build a local context or a local pattern structure.
- [ ] Build a local concept lattice or a restricted lattice diagram.
- [ ] Select local concepts/classifiers.
- [ ] Compute local fidelity of the explanation to the black-box model.
- [ ] Prepare a PDF report and a short presentation.

### Good solution

- [ ] Compare several local-neighborhood generation strategies.
- [ ] Implement proximity weights or a similarity kernel around the explained
      instance.
- [ ] Compare pure concepts with alpha-weak concepts.
- [ ] Implement several concept scoring metrics: support, purity, weighted
      fidelity, stability-inspired score.
- [ ] Compare PS-LIME with LIME or SHAP.
- [ ] Show local lattices for 2-3 instances and explain paths from a general
      concept to more specific concepts.

### Excellent solution

- [ ] Study explanation stability under different random seeds.
- [ ] Add counterfactual analysis: which minimal changes flip the black-box
      prediction.
- [ ] Add alterfactual analysis: which features can change without changing the
      black-box prediction.
- [ ] Add constraints that keep the local lattice small and inspectable.
- [ ] Test the approach on several datasets or several black-box models.
- [ ] Propose your own local-context construction strategy.

## Submission

Your repository should contain:

- working code;
- one main notebook demonstrating the full pipeline;
- a PDF report;
- a short presentation;
- saved examples of local explanations;
- `requirements.txt` or `pyproject.toml`;
- `README.md` or `README_EN.md` with reproduction instructions.

The report must describe not only the accuracy of the black-box model, but also
the quality of explanations: local fidelity, compactness, stability, and the
interpretability of local concepts.

## Recommended Repository Structure

```text
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda_and_blackbox.ipynb
│   ├── 02_local_contexts.ipynb
│   └── 03_ps_lime_explanations.ipynb
├── src/
│   ├── data.py
│   ├── blackbox.py
│   ├── perturbation.py
│   ├── local_context.py
│   ├── pattern_structure.py
│   ├── lattice.py
│   ├── concept_scoring.py
│   ├── explanations.py
│   └── evaluation.py
├── outputs/
│   ├── local_lattices/
│   └── explanations/
├── reports/
│   ├── report.pdf
│   └── presentation.pdf
├── requirements.txt
└── README.md
```

## Baseline Pipeline

Let there be a trained black-box model:

```text
f: X -> {0, 1}
```

The model may be complex and non-interpretable. For a selected instance `x`, the
goal is not to explain the whole model globally, but to explain its behavior in
the locality of `x`.

### 1. Local Neighborhood

Generate perturbed samples:

```text
N(x) = {z_1, ..., z_N}
```

The perturbation strategy should respect the data type. For numerical features,
you may use Gaussian noise, bootstrap from nearby training objects, sampling
within quantile intervals, or interpolation with nearest neighbors. For
categorical features, sample valid categories from their empirical distribution
or change only a subset of categorical values. For texts, tag sets, sequences,
and graphs, generate valid local variants: remove or replace tokens, edit small
sequence fragments, keep local graph fragments, or sample nearest training
objects.

Each perturbed sample may receive a proximity weight:

```text
pi_x(z) = exp(-dist(x, z)^2 / sigma^2)
```

This is analogous to the similarity kernel used in LIME.

### 2. Black-Box Pseudo-Labels

For every perturbed sample, obtain:

```text
y_z = f(z)
```

These pseudo-labels are the target of the local explanation. True labels can be
used for additional analysis, but local fidelity is computed with respect to the
black-box model, not with respect to ground truth.

### 3. Local Context

For a binarized variant, build a local formal context:

```text
K_x = (G_x, M_x, I_x)
```

where `G_x` is the set of perturbed samples, possibly including `x` itself,
`M_x` is the set of interpretable binary attributes, and `I_x` is the incidence
relation.

Instead of a binary context, you may build a local pattern structure:

```text
(G_x, (D_x, meet), delta_x)
```

where `D_x` is the selected description space and `meet` gives the common
description of several local objects. Interval vectors are one convenient
example for numerical data: in that case the meet operation returns the smallest
interval containing compared values. For set-valued, textual, sequential, or
graph descriptions, define the corresponding meet operation and generalization
order explicitly.

### 4. Local Concept Lattice

Build the concept lattice or a restricted fragment of it. Constraints may be
based on:

- minimum support;
- maximum intent size;
- minimum purity;
- maximum description complexity: interval width, set size, sequence-pattern
  length, or graph-fragment size;
- top-n concepts by an interestingness measure.

The order between concepts must be preserved. More general concepts cover more
objects and have weaker descriptions. More specific concepts cover fewer objects
and have stronger descriptions.

### 5. Concepts as Local Classifiers

A concept can be treated as a local classifier for class `c` if its extent is
dominated by pseudo-label `c`.

Strict variant:

```text
purity(C) = 1
```

This means that all objects in the concept extent have the same black-box
pseudo-label.

Soft variant:

```text
purity_c(C) >= alpha
```

For example, `alpha = 0.8` or `0.9`. This is useful when the local neighborhood
is noisy or when the explained instance is near the decision boundary.

### 6. Explanation Object

For the explained instance `x`, find concepts that cover `x` or nearby local
samples and support the predicted class `f(x)`.

The explanation should include:

- the black-box prediction;
- selected concepts;
- concept intents;
- concept extents;
- support and purity;
- local fidelity;
- nearest more general and more specific concepts;
- a visualization of the local concept lattice or its fragment.

## Explanation Quality Metrics

### Local Fidelity

The fraction of local samples where the explanation agrees with the black-box
model:

```text
fidelity = mean(explanation(z) == f(z))
```

You may also compute a weighted version using proximity weights `pi_x(z)`.

### Compactness

Possible compactness measures:

- number of selected concepts;
- average intent size;
- number of features used in the explanation;
- size of the displayed lattice fragment.

### Purity

For a concept `C = (A, B)`:

```text
purity_c(C) = |{z in A: f(z) = c}| / |A|
```

Pure concepts are easier to explain, but they may be too specific.

### Coverage

Coverage measures which part of the local neighborhood is explained by the
selected concept set.

### Stability

A practical stability check is to repeat neighborhood generation with different
random seeds and compare:

- whether key attributes in concept intents remain the same;
- whether local fidelity remains close;
- whether the selected class is stable;
- whether the local lattice has similar size and structure.

## Baseline Explanation Methods

Recommended comparisons:

- classical LIME for tabular data;
- SHAP or KernelSHAP;
- local decision tree;
- global decision tree;
- kNN explanation with nearest neighbors.

The comparison should focus on explanation quality, not only on model accuracy.
A black-box model may be highly accurate, while an explanation may still have
low local fidelity or be too large to inspect.

## Interpretability Analysis

In the report, show at least two kinds of examples:

1. An instance with a compact and pure explanation.
2. An instance where the explanation is unstable, too complex, or close to the
   decision boundary.

For every example, describe:

- which features appear in the local explanation;
- which concepts are above and below the selected concept;
- what changes when moving to a more general concept;
- what changes when moving to a more specific concept;
- whether the explanation agrees with domain intuition.

## Counterfactual and Alterfactual Analysis

As an advanced extension, study counterfactual and alterfactual explanations.

Counterfactual question:

```text
Which minimal changes would flip the black-box prediction?
```

In the local lattice, this can be studied through nearby concepts with high
purity for the opposite class.

Alterfactual question:

```text
Which features can be changed without changing the black-box prediction?
```

In the local lattice, this is related to moving toward more general concepts of
the same class. If a condition can be removed while purity remains high, that
feature is probably not necessary for the local decision.

## Typical Mistakes

- Evaluating only black-box accuracy and not computing local fidelity.
- Building a global lattice on the whole dataset instead of a local lattice for
  the explained instance.
- Using ground-truth labels instead of black-box pseudo-labels and then claiming
  to explain the black-box model.
- Generating perturbations that violate feature types or produce impossible
  objects.
- Not fixing random seeds.
- Showing the whole lattice when it is unreadable.
- Calling a feature-importance list a lattice explanation without showing
  extents, intents, and the concept order.
- Comparing PS-LIME and LIME on different instances.
- Ignoring failed explanations.

## Check Questions

1. How is a local explanation different from a global model interpretation?
2. What does PS-LIME explain: true labels or black-box predictions?
3. Why does LIME generate perturbed samples around the explained instance?
4. How do you build a local formal context?
5. What is a pure concept in the local lattice?
6. Why can an overly specific concept be a poor explanation?
7. What does moving upward in the concept lattice mean?
8. What does moving downward in the concept lattice mean?
9. How do you measure local fidelity?
10. How can explanation stability be tested?

## Recommended References

- B. Ganter, R. Wille. *Formal Concept Analysis: Mathematical Foundations*.
- B. Ganter, S. O. Kuznetsov. *Pattern Structures and Their Projections*.
- S. O. Kuznetsov. *Pattern Structures for Analyzing Complex Data*.
- S. O. Kuznetsov. *Fitting Pattern Structures to Knowledge Discovery in Big
  Data*.
- M. T. Ribeiro, S. Singh, C. Guestrin. *"Why Should I Trust You?": Explaining
  the Predictions of Any Classifier*.

## Student implementation / internship example

An independent implementation and experimental study of the proposed
Pattern-Structure-based local explanation approach was developed by
Denis Ushakov during a summer research internship at the HSE
International Laboratory for Intelligent Systems and Structural Analysis.

The implementation includes:
- local neighborhood generation;
- Interval Pattern Structure explanations;
- generalization graphs;
- comparison with a Decision Tree surrogate;
- experiments on the Breast Cancer Wisconsin dataset.

Repository:
https://github.com/DeNNazaVR/AI
