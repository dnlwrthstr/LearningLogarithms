# Learning Logarithms

A collection of Jupyter notebooks dedicated to learning logarithms, starting from the problem that
forced their invention — Napier's astronomers drowning in multiplications in 1614 — through $e$,
the natural logarithm as an area, log scales and growth rates, and on to information theory,
likelihood, decibels, music, algorithms, cryptography and the complex logarithm. Each notebook
follows the same layered style as the sibling repositories: motivation, intuition, formal theory,
worked examples, Python code, visualizations, exercises.

It builds on the sibling repositories:

- [LearningCalculus](../LearningCalculus/): functions, limits, derivatives, integrals.
- [LearningProbability](../LearningProbability/): expectation, distributions, maximum likelihood.
- [LearningComplexNumbers](../LearningComplexNumbers/): the notebook format and motivation-first style this series copies.

## Project Structure

- [`CURRICULUM.md`](./CURRICULUM.md): the full module plan, with the prerequisite for each notebook and its status.
- `modules/NN_<topic>/`: one numbered folder per module, notebooks numbered inside.
- `GUIDELINES.md`: the notebook specification, shared with the other Learning* repos, plus a logarithms section at the end.

| # | Module | What it covers |
|---|---|---|
| 00 | [Foundations](./modules/00_foundations/) | why logarithms exist, inverse of exponentiation, the laws, change of base |
| 01 | [The natural logarithm and $e$](./modules/01_the_natural_logarithm/) | compound interest, area under $1/x$, derivatives, series, harmonic numbers |
| 02 | [Graphs, equations and scales](./modules/02_graphs_equations_and_scales/) | graphs and reflection, solving equations, log-log plots, growth rates |
| 03 | [Computing logarithms](./modules/03_computing_logarithms/) | Napier and Briggs tables, the slide rule, how `math.log` works, floating point |
| 04 | [Logarithms in calculus](./modules/04_logarithms_in_calculus/) | logarithmic differentiation, $\int dx/x$, half-life, Stirling, primes |
| 05 | [Information and probability](./modules/05_information_and_probability/) | bits, entropy, log-likelihood, cross-entropy, log-normal and Benford |
| 06 | [Applications](./modules/06_applications/) | decibels and pH, log returns, music, $O(\log n)$ algorithms, discrete logs |
| 07 | [Beyond the reals](./modules/07_beyond_the_reals/) | complex logarithm, matrix logarithm, logarithmic spirals |

**Status:** complete. All 35 notebooks open with a motivating problem and have full theory, worked examples, Python code, visualizations, and three tiers of exercises with worked solutions and numerical checks. [`CURRICULUM.md`](./CURRICULUM.md) links each notebook individually.

## Getting Started

You need a Python 3 environment. Install the dependencies:

```bash
pip install -r requirements.txt
```

Notebook outputs are stripped automatically on commit. After cloning, register the git filter once:

```bash
nbstripout --install --attributes .gitattributes
```

Start the Jupyter Notebook server:

```bash
jupyter notebook
```

## License

This project is for educational purposes.
