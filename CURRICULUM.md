# Curriculum

Eight modules, 35 notebooks, ordered so each one only needs earlier modules plus math from the sibling repos.
Status: ☐ planned · ◐ in progress · ☑ done.

**Prerequisite shorthand** (paths inside the sibling repos):

- **C:** [LearningCalculus](https://github.com/dnlwrthstr/LearningCalculus), `hypersklill/…` or `coursera/…`
- **P:** [LearningProbability](https://github.com/dnlwrthstr/LearningProbability), `coursera/…`
- **F:** [LearningMathFinance](https://github.com/dnlwrthstr/LearningMathFinance), `modules/…`
- **N:** [LearningNeuralNetworks](https://github.com/dnlwrthstr/LearningNeuralNetworks), `prerequisites/…`
- **L:** [LearningLinearAlgebra](https://github.com/dnlwrthstr/LearningLinearAlgebra), `coursera/…`
- **Z:** [LearningComplexNumbers](https://github.com/dnlwrthstr/LearningComplexNumbers), `modules/…`
- **—**: needs only earlier notebooks in this repository.

---

## 00 Foundations

| # | Notebook | Prerequisite | Status |
|---|---|---|---|
| 01 | [Why logarithms exist: Napier, astronomers and turning multiplication into addition](./modules/00_foundations/01_why_logarithms_exist.ipynb) | C: `hypersklill/00_basics/03_functions_in_mathematics` | ☑ |
| 02 | [The logarithm as the inverse of exponentiation: $b^y = x \iff y = \log_b x$](./modules/00_foundations/02_logarithm_as_inverse_of_exponentiation.ipynb) | C: `hypersklill/00_basics/04_properties_of_functions` | ☑ |
| 03 | [The laws of logarithms: product, quotient, power — and why they must hold](./modules/00_foundations/03_laws_of_logarithms.ipynb) | — | ☑ |
| 04 | [Change of base; common, natural and binary logarithms](./modules/00_foundations/04_change_of_base_and_common_bases.ipynb) | — | ☑ |

## 01 The natural logarithm and $e$

| # | Notebook | Prerequisite | Status |
|---|---|---|---|
| 01 | [Compound interest and the number $e$](./modules/01_the_natural_logarithm/01_compound_interest_and_e.ipynb) | C: `hypersklill/01_limits/00_limits_of_sequences`, F: `modules/00_returns_and_compounding/02_continuous_compounding_as_a_limit` | ☑ |
| 02 | [$\ln x$ as the area under the hyperbola $1/x$ (Saint-Vincent and de Sarasa)](./modules/01_the_natural_logarithm/02_ln_as_area_under_the_hyperbola.ipynb) | C: `hypersklill/04_integrals/00_introduction_to_integrals` | ☑ |
| 03 | [Derivatives of $\ln x$ and $e^x$; the inverse-function picture](./modules/01_the_natural_logarithm/03_derivatives_of_ln_and_exp.ipynb) | C: `hypersklill/02_derivaties/05_derivative_of_logarithmic_functions`, `07_derivative_of_the_inverse_function` | ☑ |
| 04 | [Series for the logarithm: $\ln(1+x)$, Mercator and fast-converging variants](./modules/01_the_natural_logarithm/04_series_for_the_logarithm.ipynb) | C: `hypersklill/02_derivaties/09_higher_order_derivatives` | ☑ |
| 05 | [Harmonic numbers, $\ln n$ and the Euler–Mascheroni constant](./modules/01_the_natural_logarithm/05_harmonic_numbers_and_euler_mascheroni.ipynb) | — | ☑ |

## 02 Graphs, equations and scales

| # | Notebook | Prerequisite | Status |
|---|---|---|---|
| 01 | [Graphs of $b^x$ and $\log_b x$: reflection in $y = x$, domain, asymptotes](./modules/02_graphs_equations_and_scales/01_graphs_of_exp_and_log.ipynb) | C: `hypersklill/00_basics/05_properties_of_numerical_functions` | ☑ |
| 02 | [Solving exponential and logarithmic equations (and the extraneous-root trap)](./modules/02_graphs_equations_and_scales/02_solving_exponential_and_log_equations.ipynb) | — | ☑ |
| 03 | [Log scales: semi-log and log-log plots, spotting exponentials and power laws](./modules/02_graphs_equations_and_scales/03_log_scales_and_power_laws.ipynb) | — | ☑ |
| 04 | [Growth rates: $\ln n \ll n^\varepsilon \ll n \ll e^n$, and $n \log n$](./modules/02_graphs_equations_and_scales/04_growth_rates_and_asymptotics.ipynb) | C: `hypersklill/01_limits/03_operations_with_limits` | ☑ |

## 03 Computing logarithms

| # | Notebook | Prerequisite | Status |
|---|---|---|---|
| 01 | [How Napier and Briggs built the tables: repeated square roots](./modules/03_computing_logarithms/01_napier_and_briggs_tables.ipynb) | — | ☑ |
| 02 | [The slide rule: a logarithm you can hold](./modules/03_computing_logarithms/02_the_slide_rule.ipynb) | — | ☑ |
| 03 | [How computers compute $\ln x$: range reduction, series, the AGM](./modules/03_computing_logarithms/03_how_computers_compute_logs.ipynb) | — | ☑ |
| 04 | [Floating point is logarithmic; `log1p`, `logsumexp` and numerical stability](./modules/03_computing_logarithms/04_floating_point_and_numerical_stability.ipynb) | — | ☑ |

## 04 Logarithms in calculus

| # | Notebook | Prerequisite | Status |
|---|---|---|---|
| 01 | [Logarithmic differentiation: $x^x$ and messy products](./modules/04_logarithms_in_calculus/01_logarithmic_differentiation.ipynb) | C: `hypersklill/02_derivaties/03_composite_function_and_derivative` | ☑ |
| 02 | [Integrals that produce logarithms: $\int dx/x$, $\int \tan x\,dx$, partial fractions](./modules/04_logarithms_in_calculus/02_integrals_that_produce_logs.ipynb) | C: `hypersklill/04_integrals/03_u_substitution`, `06_partial_fraction_expansion` | ☑ |
| 03 | [Exponential growth and decay: half-life, doubling time, the rule of 72](./modules/04_logarithms_in_calculus/03_exponential_growth_and_decay.ipynb) | C: `hypersklill/02_derivaties/06_derivative_of_exponential_functions` | ☑ |
| 04 | [Stirling's approximation: $\ln n!$ as a sum of logs](./modules/04_logarithms_in_calculus/04_stirlings_approximation.ipynb) | C: `hypersklill/04_integrals/05_integral_by_parts` | ☑ |
| 05 | [Primes and $\ln$: the prime number theorem $\pi(n) \sim n / \ln n$](./modules/04_logarithms_in_calculus/05_prime_number_theorem.ipynb) | — | ☑ |

## 05 Information and probability

| # | Notebook | Prerequisite | Status |
|---|---|---|---|
| 01 | [Bits, digits and counting: how many questions, how many digits](./modules/05_information_and_probability/01_bits_digits_and_counting.ipynb) | — | ☑ |
| 02 | [Information and entropy: surprise is $-\log_2 p$](./modules/05_information_and_probability/02_entropy_and_information.ipynb) | P: `coursera/02_discribing_distribution/03_expected_value_of_a_function` | ☑ |
| 03 | [Log-likelihood: why we maximize $\sum \ln p$ instead of $\prod p$](./modules/05_information_and_probability/03_log_likelihood_and_mle.ipynb) | P: `coursera/05_point_estimation/01_intro_to_maximum_likelihood` | ☑ |
| 04 | [Cross-entropy, KL divergence and softmax](./modules/05_information_and_probability/04_cross_entropy_kl_and_softmax.ipynb) | N: `prerequisites/07_softmax_and_cross_entropy` | ☑ |
| 05 | [Multiplicative processes: the log-normal distribution and Benford's law](./modules/05_information_and_probability/05_lognormal_and_benfords_law.ipynb) | P: `coursera/01_probability_distribution/10_normal_distribution` | ☑ |

## 06 Applications

| # | Notebook | Prerequisite | Status |
|---|---|---|---|
| 01 | [Decibels, Richter and pH: logarithmic perception (Weber–Fechner)](./modules/06_applications/01_decibels_richter_and_ph.ipynb) | — | ☑ |
| 02 | [Log returns and continuous compounding](./modules/06_applications/02_log_returns_and_continuous_compounding.ipynb) | F: `modules/00_returns_and_compounding/01_simple_vs_log_returns` | ☑ |
| 03 | [Music: octaves, equal temperament and cents](./modules/06_applications/03_music_octaves_and_cents.ipynb) | — | ☑ |
| 04 | [Logarithmic algorithms: binary search, divide and conquer, $O(\log n)$](./modules/06_applications/04_logarithmic_algorithms.ipynb) | — | ☑ |
| 05 | [The discrete logarithm and Diffie–Hellman key exchange](./modules/06_applications/05_discrete_logarithm_and_cryptography.ipynb) | — | ☑ |

## 07 Beyond the reals

| # | Notebook | Prerequisite | Status |
|---|---|---|---|
| 01 | [Logarithms of negative and complex numbers: $\ln(-1) = i\pi$](./modules/07_beyond_the_reals/01_logarithms_of_complex_numbers.ipynb) | Z: `modules/01_polar_form_and_euler/03_eulers_formula`, `modules/03_complex_functions/03_the_complex_logarithm_and_branch_cuts` | ☑ |
| 02 | [The matrix exponential and matrix logarithm](./modules/07_beyond_the_reals/02_matrix_exponential_and_logarithm.ipynb) | L: `coursera/04_eigenvalues_and_eigenvectors/04_calculating_eigenvalues_and_eigenvectors` | ☑ |
| 03 | [Logarithmic spirals: self-similarity in shells, galaxies and hawks](./modules/07_beyond_the_reals/03_logarithmic_spirals.ipynb) | C: `hypersklill/00_basics/07_trigonometric_functions` | ☑ |

---

## Reference courses

- Eli Maor, *e: The Story of a Number* — the historical motivation used throughout.
- Julian Havil, *John Napier: Life, Logic, and Legend* — module 03.
- Thomas Cover & Joy Thomas, *Elements of Information Theory* — module 05.
