# Cooperative Kernel Regression — Distributed and Federated Optimization

This repository contains the implementation and analysis of a numerical project on **distributed optimization**, **federated learning**, and **differential privacy** for **kernel ridge regression with a Nyström approximation**.

The project was developed in the context of the course:

**Numerical Project in Python — M2 Data Science IP Paris × SOD314 ENSTA (2026)**

## Authors

- Mahdi Hadj Taieb
- Rayen Mansour
- Rayen Zargui

## Mentors

- Andrea Simonetto
- Romain Pujol

---

# Overview

The objective of this project is to study how a kernel regression problem can be solved in increasingly constrained settings:

1. **Distributed optimization** over multiple agents connected through a communication graph  
2. **Federated learning**, where agents act as clients performing local training before server aggregation  
3. **Differentially private optimization**, where noise is added to preserve privacy

The project is based on the minimization of a regularized kernel ridge regression objective using a **Nyström approximation**.

---

## Mathematical background

We observe noisy samples

$$
y_i = f(x_i) + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0,\sigma^2),
$$

and approximate the target function using a kernel expansion:

$$
f(x) \approx \sum_{j \in \mathcal{M}} \alpha_j\, k(x,x_j),
\qquad
k(x,x_j)=\exp\bigl(-(x-x_j)^2\bigr).
$$

Using a Nyström approximation with $m$ landmarks, the optimization problem becomes:

```math
\alpha^\star
=
\arg\min_{\alpha \in \mathbb{R}^m}
\frac{\sigma^2}{2}\alpha^\top K_{mm}\alpha
+
\frac{1}{2}\|y-K_{nm}\alpha\|_2^2
+
\frac{\nu}{2}\|\alpha\|_2^2.
```
This problem is studied under three different frameworks.

---

# Project structure

```text
cooperative-kernel-regression/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── Final_Project.ipynb
│
├── report/
│   └── report.pdf
│
└── data/
    └── README.md
