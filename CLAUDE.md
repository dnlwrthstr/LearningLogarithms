# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A personal learning project on logarithms, from their invention to their applications, part of the
Learning* series alongside `../LearningCalculus`, `../LearningComplexNumbers`, `../LearningLinearAlgebra`, `../LearningProbability`,
`../LearningNeuralNetworks` and `../LearningMathFinance`. The deliverables are educational Jupyter
notebooks. There is no application code, build system, or test suite.

## Structure

- `CURRICULUM.md`: the module plan. Check it before creating a notebook; keep it in sync (mark notebooks done, add new ones in the right module).
- `modules/NN_<topic>/NN_<notebook>.ipynb`: notebooks, numbered.
- `GUIDELINES.md`: the authoritative spec for every notebook, including the "Logarithm Notebooks" section at the end. Read it before creating or editing a notebook.

## Creating notebooks

Follow GUIDELINES.md. The essentials:

- Standard section order: Introduction → Theory (markdown + LaTeX) → Worked Examples → Python Examples → Visualizations → Exercises (basic/intermediate/advanced) → Solutions → Conclusion.
- Explain each topic in layers: intuition first, then formal definition, worked examples, visuals, code, exercises.
- The Introduction has a **Prerequisites** list linking earlier notebooks here and the sibling-repo notebooks named in CURRICULUM.md (relative links, e.g. `../../../LearningCalculus/hypersklill/02_derivaties/05_derivative_of_logarithmic_functions.ipynb`).
- Open with a motivating problem the idea solves, before defining it.
- Show every result algebraically (exponent form and log form) and as a picture (log-spaced number lines, log axes, area under 1/x).
- Write the base explicitly: $\ln$, $\log_{10}$, $\log_2$ — never a bare $\log$ in markdown (except big-O, $O(\log n)$).
- Default libraries: numpy, matplotlib, sympy, and the standard library's `math`. Keep notebooks self-contained with minimal dependencies.
- Code cells should run independently where practical; readable variable names, no dense one-liners.
- Compare floats with a tolerance, never `==` (`math.log(1000, 10)` is not exactly 3).
- Currency dollar signs: never write a literal `$` in prose (PyCharm pairs it with the next math `$`).
- Before finishing: verify the notebook JSON is valid, all cells render, and code runs.

## Environment

Python 3 with Jupyter. Run notebooks with `jupyter notebook`.

Two Python environments touch the notebooks: Anaconda (numpy 1.x, `/opt/anaconda3/bin/python`) executes them during authoring, and the user's PyCharm virtualenv (numpy 2.x) runs them interactively. Write numpy code that works in both — keep the `np.trapezoid` shim (`if not hasattr(np, "trapezoid"): np.trapezoid = np.trapz`) in the imports cell when integrating, and avoid APIs removed in numpy 2.0 (`np.trapz`, `np.in1d`, `np.row_stack`).

Verify a notebook by executing it in place:

```bash
/opt/anaconda3/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 modules/<module>/<notebook>.ipynb
```

## Git

Notebook outputs are stripped on commit via an nbstripout git filter (`.gitattributes` + local git config). In a fresh clone the filter must be re-registered once with `nbstripout --install --attributes .gitattributes`. Never commit notebooks with outputs; local working copies may keep them.
