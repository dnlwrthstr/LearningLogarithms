# GUIDELINES.md

## Purpose

Template for generating high-quality educational Jupyter notebooks in the style previously established.

Use this document as the default specification when creating new math / statistics / programming learning notebooks.

---

## Core Output Requirements

- Deliver a **downloadable `.ipynb` notebook**.
- Notebook must run in a standard Python 3 environment.
- Use clean structure, readable markdown, and executable code cells.
- Prefer self-contained notebooks (minimal dependencies).

Recommended libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
```

Use additional libraries only when necessary.

---

## Teaching Style

### Tone

- Clear
- Structured
- Practical
- Precise
- Beginner-friendly but not simplistic

### Explain Concepts in Layers

For each topic:

1. Intuition  
2. Formal definition  
3. Worked examples  
4. Visual explanation  
5. Python implementation  
6. Practice exercises  
7. Summary

---

## Notebook Structure (Default)

# Title

Short description of what the notebook teaches.

## 1. Introduction

- Why topic matters
- Real-world relevance
- What learner will know after completion

## 2. Theory

Use markdown with LaTeX:

```markdown
$$
f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}
$$
```

## 3. Worked Examples

Show step-by-step calculations.

## 4. Python Examples

Executable code cells demonstrating the topic.

## 5. Visualizations

Use plots where useful.

## 6. Exercises

Include short tasks.

## 7. Solutions / Checks

Optional answer cells.

## 8. Conclusion

Concise recap.

---

## Markdown Rules

- Use headings generously.
- Use bullet lists for clarity.
- Keep paragraphs short.
- Use LaTeX for formulas.

Inline math:

```markdown
$ax^2+bx+c$
```

Display math:

```markdown
$$
x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}
$$
```

Currency dollar signs: never write a literal `$` (or `\$`, or `&#36;`) in markdown prose — PyCharm's renderer pairs it with the next math `$` and swallows the text between as LaTeX. Write amounts as code spans (`` `$1` ``, `` `$22,000` ``), and keep the `$` symbol out of LaTeX formulas entirely (write `= 5`, not `= \$5`; the surrounding prose carries the currency).

---

## Code Rules

- Every code cell should run independently where practical.
- Use comments sparingly but clearly.
- Prefer readable variable names.
- Avoid unnecessary abstraction.

Good:

```python
x = np.linspace(-5, 5, 400)
y = x**2
```

Avoid:

```python
a=np.linspace(-5,5,400);b=a**2
```

---

## Visualization Rules

Use matplotlib unless another library is requested.

### Standards

- Figure size readable
- Axis labels
- Title
- Grid when useful
- Legends when multiple lines

Template:

```python
plt.figure(figsize=(8,5))
plt.plot(x, y, label="f(x)")
plt.title("Function Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```

---

## Topic-Specific Enhancements

### Mathematics

Include:

- Symbolic derivations with SymPy
- Numeric approximations
- Graphical intuition

### Statistics

Include:

- Simulations
- Histograms
- Hypothesis tests
- Interpretation of outputs

### Probability

Include:

- Simulations of random experiments
- Distributions and their visualizations
- Worked probability calculations

### Calculus

Include:

- Limits
- Derivatives
- Integrals
- Tangents / areas

---

## Exercise Design

Use three levels:

### Basic

Direct application.

### Intermediate

Multi-step reasoning.

### Advanced

Interpretation or proof-like thinking.

---

## Quality Checklist

Before finalizing:

- Notebook opens correctly
- JSON valid
- All cells render
- Code runs
- Plots display
- Math formatting correct
- Sections logically ordered
- No unexplained jumps in reasoning

---

## Default Closing Message

Use:

> Done — here is the notebook:

> [Download the notebook](sandbox:/mnt/data/<filename>.ipynb)

> It includes:
> - theory
> - worked examples
> - Python code
> - visualizations
> - exercises

---

## Reusable Prompt Template

```text
Create a Jupyter notebook in the established style about [TOPIC].

Include:
- intuition
- formal theory
- step-by-step examples
- Python code
- charts/visuals
- exercises
- summary

Make it downloadable as .ipynb
```

---

## Golden Rule

Every notebook should help the learner **understand**, **see**, and **apply** the concept.

---

## Logarithm Notebooks (this repository)

Everything above applies. In addition, every notebook in this repository:

### Opens with a motivating problem

Logarithms were invented to solve a concrete problem: Napier's astronomers were drowning in
multiplications, and a table that turned multiplication into addition saved them years. Every
notebook starts the same way — with a problem that is hard, slow or impossible *without* the idea,
before the idea is defined. "Why would anyone want this?" is answered in the Introduction, never
postponed to the Conclusion.

### Opens with a Prerequisites list

In the Introduction, list the concepts the notebook relies on, linking earlier notebooks in this
repository and the specific notebooks in the sibling repositories (relative links such as
`../../../LearningCalculus/hypersklill/02_derivaties/05_derivative_of_logarithmic_functions.ipynb`).
CURRICULUM.md names the link for each notebook.

### Shows every result two ways

- **Algebraically**: exponent form $b^y = x$ and log form $y = \log_b x$, side by side, with LaTeX.
- **Geometrically**: a picture. The core picture of the subject is that a logarithm turns a
  *multiplicative* axis into an *additive* one — equal ratios become equal steps. Use it often:
  number lines with log spacing, log-scaled axes, the area under $1/x$, mirror images across $y = x$.

### Conventions

- $\ln x$ is the natural log, $\log_{10} x$ and $\log_2 x$ are always written with their base.
  Plain $\log x$ is avoided in markdown (it means $\ln$ in Python, $\log_{10}$ on calculators,
  $\log_2$ in computer science). Say so whenever it could confuse.
  The one exception is big-O notation, $O(\log n)$, where the base only changes a constant factor
  and is conventionally omitted; say so the first time it appears in a notebook.
- In Python use `math.log(x)` (natural), `math.log(x, b)`, `math.log10`, `math.log2`,
  `math.log1p`, and `np.log`, `np.log10`, `np.log2`, `np.log1p`, `np.logaddexp` for arrays.
  Use `sympy.log` for exact symbolic work (`sp.log(x, b)` for other bases, `sp.expand_log(..., force=True)`).
- Logarithms of real numbers need $x > 0$ and $b > 0$, $b \neq 1$. State the domain when solving
  equations and check candidate solutions against it.
- Log-scaled plots use `ax.set_xscale("log")` / `ax.set_yscale("log")` with a grid on both major and
  minor ticks (`ax.grid(True, which="both", alpha=0.3)`), and the axis label says which scale it is.
- Compare floats with a tolerance (`np.isclose`, `math.isclose`), never `==`: `math.log(1000, 10)`
  is `2.9999999999999996`.
- Exercises include at least one "estimate it first, then compute" question (orders of magnitude,
  number of digits, doubling times) — mental estimation is what logarithms are for.
