# AQSF-NAV & AQSF-NAV v2
## Adaptive Quantum Spectral Field Navigation with Predictive Electrostatic Safety Architecture

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit 2.6+](https://img.shields.io/badge/Qiskit-2.6+-6929C4.svg)](https://github.com/Qiskit/qiskit)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: v2 Validated](https://img.shields.io/badge/AQSF--NAV-v2.0%20Validated-brightgreen.svg)]()

> **AQSF-NAV v2** is a decentralized multi-agent robotic navigation and safety architecture combining classical **Artificial Potential Fields (APF)**, continuous **bilinear gradient controllers**, an **8-qubit separable 2-D Quantum Fourier Transform (2-D QFT)** adaptive spectral filter, an **Electrostatic-Inspired Dynamic Potential Field**, **Predictive Closest-Approach Analysis**, and an independent **Control Barrier Function (CBF)** safety shield.

---

## 📑 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Motivation & The v1 Bifurcation Discovery](#2-research-motivation--the-v1-bifurcation-discovery)
3. [Problem Definition](#3-problem-definition)
4. [Existing v1 Architecture](#4-existing-v1-architecture)
5. [AQSF-NAV v2 Architecture](#5-aqsf-nav-v2-architecture)
6. [Mathematical Formulation](#6-mathematical-formulation)
   - [Static Obstacle APF](#static-obstacle-artificial-potential-field)
   - [Electrostatic-Inspired Dynamic Potential](#electrostatic-inspired-dynamic-potential-field)
   - [Predictive Collision Detection](#predictive-collision-detection)
   - [Independent Control Barrier Function (CBF) Shield](#independent-control-barrier-function-cbf-safety-shield)
   - [Separable 2-D QFT & Adaptive Bandwidth](#separable-2-d-qft--adaptive-spectral-filtering)
7. [Hybrid Quantum-Classical Nature & Realistic Attribution](#7-hybrid-quantum-classical-nature--realistic-attribution)
8. [Software Architecture & Directory Structure](#8-software-architecture--directory-structure)
9. [Prerequisites & Installation](#9-prerequisites--installation)
10. [Execution Commands](#10-execution-commands)
    - [Running v1 Baseline Scripts](#running-v1-baseline-scripts)
    - [Running v2 Unit Test Suite](#running-v2-unit-test-suite)
    - [Running v2 Comprehensive Experiments](#running-v2-comprehensive-experiments)
11. [Visualization & Interactive Dashboard](#11-visualization--interactive-dashboard)
12. [Experimental Benchmark Results](#12-experimental-benchmark-results)
13. [Analysis of Baselines (A through E)](#13-analysis-of-baselines-a-through-e)
14. [Known Limitations & Edge Cases](#14-known-limitations--edge-cases)
15. [Scientific Reproducibility](#15-scientific-reproducibility)
16. [Future Research Directions](#16-future-research-directions)

---

## 1. Project Overview

Autonomous multi-agent navigation in dynamic, shared workspaces requires real-time collision avoidance, smooth trajectory planning, and provable safety. The **Adaptive Quantum Spectral Field Navigation (AQSF-NAV)** project investigates hybrid quantum-classical spatial representation where continuous potential fields are amplitude-encoded into quantum registers, transformed into the spatial frequency domain via the **Quantum Fourier Transform (QFT)**, adaptively filtered to attenuate turbulence and high-frequency noise, and reconstructed via the **Inverse Quantum Fourier Transform (IQFT)**.

**AQSF-NAV v2** significantly advances this paradigm by introducing the **Predictive Electrostatic Safety Architecture**:
- **Static Obstacles** continue to be mediated by classical Artificial Potential Fields (APF) with singularity clamping.
- **Dynamic Entities** (neighboring robots and moving obstacles) are governed by an **electrostatic-inspired repulsive potential field** with analytical force gradients and finite bounds.
- **Relative Motion** is continuously evaluated via **predictive closest-approach analysis**, calculating time-to-collision $t^*$, minimum predicted separation $d^*$, and collision urgency scores $\rho_{ij}$.
- **Mathematical Safety Enforcement** is independently guaranteed downstream of the quantum spectral pipeline by a **Control Barrier Function (CBF)** safety shield formulated as a Quadratic Program (QP).

---

## 2. Research Motivation & The v1 Bifurcation Discovery

In **AQSF-NAV v1**, exhaustive diagnostics (Steps 9W through 9AE) evaluated the effects of 2-D quantum adaptive low-pass filtering on multi-agent navigation:
- **Zero Mathematical Inversion Error**: Unfiltered QFT + IQFT matched classical floating-point navigation with machine precision ($\|v_Q - v_C\| \le 1.35 \times 10^{-15}$).
- **Single-Agent Success**: Closed-loop rollouts confirmed single-agent convergence ($100\%$ goal convergence, $2.02\%$ reconstruction error).
- **The Multi-Agent Clearance Degradation**: In multi-agent crossing scenarios, adaptive spectral filtering reduced minimum inter-agent clearance from $1.0583$ down to $0.9674$ (violating the safety threshold $d_{\text{safe}} = 1.05$).
- **Topological Avoidance Bifurcation**: Diagnostic decomposition ([diagnostics/controller_decomposition.py](file:///Users/ronnie/Documents/ASQF/diagnostics/controller_decomposition.py)) proved that subtle phase shifts from spectral filtering flipped the sign of the pre-normalization velocity ($v_{\text{net}, x}: -0.3219 \to +0.2546$), crossing a phase-space separatrix. This caused Agent 1 to switch from a safe westward detour to a risky eastward cut directly across Agent 0's path.

**Core Research Question for v2**: How can multi-agent collision safety be mathematically enforced in a quantum spectral navigation framework without sacrificing the filtering benefits or relying on the filter itself to guarantee safety?

**v2 Solution**: Decouple dynamic safety into a predictive electrostatic protective field combined with an **independent Control Barrier Function (CBF)** safety shield that executes *after* spectral reconstruction and velocity planning.

---

## 3. Problem Definition

Consider $N$ autonomous mobile robots $\mathcal{R}_i$ ($i \in \{1, \dots, N\}$) and $M$ dynamic moving obstacles $\mathcal{O}_k$ ($k \in \{1, \dots, M\}$) operating in a 2-D bounded workspace $\mathcal{W} \subset \mathbb{R}^2$.
- Each robot $\mathcal{R}_i$ has position $p_i \in \mathbb{R}^2$, velocity $v_i \in \mathbb{R}^2$ ($\|v_i\| \le v_{\max}$), and physical radius $R_i = 0.25$.
- Each moving obstacle $\mathcal{O}_k$ has position $q_k \in \mathbb{R}^2$, velocity $u_k \in \mathbb{R}^2$, and radius $R_k$.
- The required collision-free separation between entity $i$ and entity $j$ is:
  $$D_{ij} = R_i + R_j + m$$
  where $m = 0.20$ is the additional safety margin. For two identical robots:
  $$D_{ij} = 0.25 + 0.25 + 0.20 = 0.70 \text{ world units}$$

The objective is to guide all robots to their target destinations $g_i$ in minimal time while strictly ensuring $\|p_i(t) - p_j(t)\| \ge D_{ij}$ and $\|p_i(t) - q_k(t)\| \ge D_{ik}$ for all $t \ge 0$.

---

## 4. Existing v1 Architecture

```text
Environment
    ↓
Artificial Potential Field (Goal + Static Obstacles + Dynamic Agents)
    ↓
16 × 16 Navigation Field (256 discrete points)
    ↓
8-Qubit Amplitude Encoding
    ↓
Separable 2-D QFT (Qiskit Circuit)
    ↓
Frequency Spectrum
    ↓
Spectral Complexity (Second Moment C)
    ↓
Cubic Adaptive Bandwidth Selection B(C)
    ↓
Radial Low-Pass Spectral Filtering
    ↓
Separable 2-D IQFT (Qiskit Circuit)
    ↓
Reconstructed Navigation Field
    ↓
Continuous Bilinear Gradient Controller
    ↓
Robot Motion
```

---

## 5. AQSF-NAV v2 Architecture

```text
STATIC ENVIRONMENT                        DYNAMIC ENVIRONMENT
       │                                           │
       ▼                                           ▼
Existing Static Obstacle APF              Moving Obstacles + Peer Robots
(environment/field.py)                             │
       │                                           ├────────────────────────┐
       │                                           ▼                        ▼
       │                              Electrostatic-Inspired       Predictive Relative
       │                              Dynamic Potential Field      Collision Detection
       │                              (navigation_v2/              (navigation_v2/
       │                               electrostatic_potential.py)  predictive_collision.py)
       │                                           │                        │
       ▼                                           ▼                        │
Combined Navigation Field: F_total = F_static + F_dynamic                   │
       │                                                                    │
       ▼                                                                    │
Amplitude Encoding (8 Qubits, 256 amplitudes)                               │
       │                                                                    │
       ▼                                                                    │
Qiskit Separable 2-D QFT                                                    │
       │                                                                    │
       ▼                                                                    │
Spectral Second-Moment Complexity C(t)                                      │
       │                                                                    │
       ▼                                                                    │
Adaptive Bandwidth B(C) & Radial Low-Pass Filtering                          │
       │                                                                    │
       ▼                                                                    │
Qiskit Separable 2-D IQFT                                                   │
       │                                                                    │
       ▼                                                                    │
Reconstructed Navigation Field F_recon                                      │
       │                                                                    │
       ▼                                                                    │
Continuous Gradient Descent + Goal Blending                                 │
       │                                                                    │
       ▼                                                                    │
Candidate Navigation Velocity v_desired                                     │
       │                                                                    │
       └──────────────────────────┬─────────────────────────────────────────┘
                                  ▼
                   Independent CBF Safety Shield
                   (navigation_v2/cbf_safety_shield.py)
                   Solves QP: min ||v - v_desired||^2
                   s.t. 2 r_ij^T (v_i - u_j) + alpha * h_ij >= 0
                                  │
                                  ▼
                         Final Safe Velocity v_safe
                                  │
                                  ▼
                   Discrete Kinematic Update: p_{t+1} = p_t + dt * v_safe
```

---

## 6. Mathematical Formulation

### Static Obstacle Artificial Potential Field
Preserved from v1 ([environment/field.py](file:///Users/ronnie/Documents/ASQF/environment/field.py)):
$$F_{\text{static}}(X, Y) = F_{\text{goal}}(X, Y) + F_{\text{obstacle}}(X, Y)$$

1. **Goal Attraction**:
   $$F_{\text{goal}}(X, Y) = \frac{1}{2} k_{\text{goal}} \left((X - g_x)^2 + (Y - g_y)^2\right), \quad k_{\text{goal}} = 2.0$$

2. **Static Obstacle Repulsion**:
   $$F_{\text{obstacle}}(X, Y) = \sum_{k} \begin{cases}
   \frac{1}{2} k_{\text{obs}} \left( \frac{1}{d_k^*} - \frac{1}{\rho_{\text{obs}}} \right)^2 & \text{if } d_k < \rho_{\text{obs}} \\
   0 & \text{otherwise}
   \end{cases}$$
   Where $d_k^* = \max(d_k, 0.5)$ clamps discrete grid singularities ($k_{\text{obs}} = 80.0$, $\rho_{\text{obs}} = 2.0$).

---

### Electrostatic-Inspired Dynamic Potential Field
Implements an artificial repulsive potential analogy ([navigation_v2/electrostatic_potential.py](file:///Users/ronnie/Documents/ASQF/navigation_v2/electrostatic_potential.py)):

For robot $i$ at $p_i$ and dynamic entity $j$ at $q_j$:
$$r_{ij} = p_i - q_j, \quad d_{ij} = \|r_{ij}\|_2$$
Define the influence boundary $L_{ij} = D_{ij} + 2.0 > D_{ij}$. The scalar repulsive potential is:
$$U_{ij}(d) = \begin{cases}
\frac{k_{ij}}{2} \left(\frac{1}{d} - \frac{1}{L_{ij}}\right)^2 & 0 < d < L_{ij} \\
0 & d \ge L_{ij}
\end{cases}$$

The analytical repulsive force is $F_{ij} = -\nabla_{p_i} U_{ij}$:
$$F_{ij} = \begin{cases}
k_{ij} \left(\frac{1}{d_{ij}} - \frac{1}{L_{ij}}\right) \frac{r_{ij}}{d_{ij}^3} & 0 < d_{ij} < L_{ij} \\
0 & d_{ij} \ge L_{ij}
\end{cases}$$

**Numerical Stability & Singularity Protections**:
1. Clamped distance: $d_{\text{eff}} = \max(d, \epsilon_{\text{dist}})$ where $\epsilon_{\text{dist}} = 10^{-4}$ (continuous) and $0.5$ (grid).
2. Force clipping: $\|F_{ij}\|_2 \le F_{\max} = 30.0$.
3. Coincident position handler: if $\|p_i - q_j\| < \epsilon$, a deterministic fallback vector is assigned, preventing division-by-zero and NaN/Inf anomalies.

---

### Predictive Collision Detection
Evaluates relative motion to assess collision urgency ([navigation_v2/predictive_collision.py](file:///Users/ronnie/Documents/ASQF/navigation_v2/predictive_collision.py)):
$$r_{ij} = p_i - q_j, \quad v_{\text{rel}} = v_i - u_j$$

1. **Predicted Time of Closest Approach ($t^*$):**
   $$t^* = \text{clip}\left( -\frac{r_{ij} \cdot v_{\text{rel}}}{\|v_{\text{rel}}\|^2}, 0.0, T \right)$$
   Where $T = 2.0$ simulation seconds. If $\|v_{\text{rel}}\|^2 < 10^{-10}$, $t^* = 0.0$.

2. **Predicted Minimum Separation ($d^*$):**
   $$d^* = \|r_{ij} + t^* v_{\text{rel}}\|_2$$

3. **Collision Urgency Score ($\rho_{ij}$):**
   $$\rho_{ij} = \max\left(0.0, \frac{D_{ij} - d^*}{D_{ij}}\right) \in [0, 1]$$
   - $\rho_{ij} = 0$: No predicted safety distance violation within horizon $T$.
   - $\rho_{ij} \to 1$: Severe predicted collision course requiring evasive steering.

4. **Right-Hand Evasive Steering Bias**:
   When $\rho_{ij} > 0.15$, a lateral steering bias is applied along the standard right-hand maritime evasion vector $\hat{n}_{\text{right}} = [d_{\text{goal}, y}, -d_{\text{goal}, x}]$, breaking head-on symmetry without fighting goal attraction.

---

### Independent Control Barrier Function (CBF) Safety Shield
Downstream of all potential field and quantum spectral calculations, safety is enforced via a convex Quadratic Program ([navigation_v2/cbf_safety_shield.py](file:///Users/ronnie/Documents/ASQF/navigation_v2/cbf_safety_shield.py)):

Define the continuous safety barrier function:
$$h_{ij}(p_i, q_j) = \|p_i - q_j\|^2 - D_{ij}^2$$
The continuous-time forward invariance condition is $\dot{h}_{ij} \ge -\alpha h_{ij}$:
$$2 r_{ij}^T (v_i - u_j) + \alpha h_{ij} \ge 0$$

For robot $i$, the QP finds the safe velocity $v_i$ closest to candidate velocity $v_{i, \text{desired}}$:
$$\min_{v_i \in \mathbb{R}^2} \frac{1}{2} \|v_i - v_{i, \text{desired}}\|^2$$
Subject to:
$$2 r_{ij}^T v_i \ge 2 r_{ij}^T u_j - \alpha h_{ij} \quad (\forall j \in \text{obstacles})$$
$$2 r_{ij}^T v_i \ge 2 r_{ij}^T u_j - 2 \kappa \alpha h_{ij} \quad (\forall j \in \text{peer robots})$$
$$\|v_i\|_2 \le v_{\max}$$
Where $\alpha = 2.0$, and $\kappa = 0.5$ represents reciprocal collision avoidance responsibility. Solved efficiently via SciPy `SLSQP`. If constraints become infeasible (e.g., trapped between opposing encroaching obstacles), the shield activates an emergency stop or maximum-clearance fallback.

---

### Separable 2-D QFT & Adaptive Spectral Filtering
Preserved from validated v1 implementation ([quantum/qft_2d_qiskit.py](file:///Users/ronnie/Documents/ASQF/quantum/qft_2d_qiskit.py), [quantum/spectral_analysis_2d.py](file:///Users/ronnie/Documents/ASQF/quantum/spectral_analysis_2d.py)):
- **8-Qubit Register**: Encodes 256 amplitudes of the normalized $16 \times 16$ field.
- **Separable Transform**: Applied to reversed logical registers (`x_qubits=[3,2,1,0]`, `y_qubits=[7,6,5,4]`) to strictly align with Qiskit's little-endian convention.
- **Spectral Second Moment Complexity ($\mathcal{C}$):**
  $$\mathcal{C} = \frac{\sum_{k_x, k_y} r(k_x, k_y)^2 P(k_x, k_y)}{r_{\max}^2 \sum_{k_x, k_y} P(k_x, k_y)}$$
  Normalized via calibrated percentiles: $\mathcal{C}_{\text{norm}} = \text{clip}\left(\frac{\mathcal{C} - C_{25}}{C_{75} - C_{25}}, 0, 1\right)$ ($C_{25} = 0.01646, C_{75} = 0.01988$).
- **Cubic Bandwidth Adaptation ($\gamma = 3.0$):**
  $$B(\mathcal{C}_{\text{norm}}) = B_{\max} - (B_{\max} - B_{\min}) \cdot \mathcal{C}_{\text{norm}}^3, \quad B_{\min} = 0.25, B_{\max} = 0.75$$

---

## 7. Hybrid Quantum-Classical Nature & Realistic Attribution

> [!IMPORTANT]
> **Scientific Integrity & Attribution**:
> 1. **No Quantum Advantage**: This research runs on classical quantum simulators (`qiskit`, `qiskit-aer`). No claim of quantum supremacy or hardware advantage is made.
> 2. **Component Attribution**: Experimental improvements in collision avoidance, minimum clearance, and zero collisions are directly produced by the **Predictive Electrostatic Field** and the **CBF Safety Shield**, NOT by quantum Fourier transformation.
> 3. **Role of QFT**: The 2-D QFT serves as a spatial frequency domain filter that attenuates high-frequency field turbulence. The CBF safety shield ensures that any smoothing artifacts or phase shifts introduced by the spectral filter cannot violate physical safety boundaries.

---

## 8. Software Architecture & Directory Structure

```text
ASQF/
├── environment/
│   └── field.py                     # v1 Classical APF (Goal attraction, static & dynamic agent repulsion)
├── navigation/
│   ├── controller.py                # v1 Classical bilinear gradient controller & dual-layer avoidance
│   └── quantum_controller.py        # v1 Quantum-driven controller operating on reconstructed fields
├── navigation_v2/                   # === NEW v2 ARCHITECTURE PACKAGE ===
│   ├── __init__.py                  # Package exports
│   ├── config.py                    # Centralized, typed AQSFConfigV2 parameters
│   ├── dynamic_entities.py          # MovingObstacle class with CV, waypoint & scripted dynamics
│   ├── electrostatic_potential.py   # Analytical electrostatic-inspired potential and forces
│   ├── predictive_collision.py      # Relative velocity closest approach (t*, d*, rho_ij)
│   ├── combined_field.py            # Superposition: F_total = F_static + F_dynamic
│   ├── cbf_safety_shield.py         # Independent CBF QP safety controller (decentralized & joint)
│   ├── controller_v2.py             # v2 Navigation step integrating field, prediction, and CBF
│   ├── quantum_pipeline.py          # 8-qubit QFT, complexity C, cubic filter, and IQFT bridge
│   ├── simulation.py                # Multi-agent simulation loop and telemetry collector
│   └── metrics_logger.py            # Structured export to CSV and JSON
├── quantum/
│   ├── adaptive_filter_2d.py        # Calibrated 2-D radial adaptive low-pass filter (gamma = 3.0)
│   ├── amplitude_encoding.py        # 8-qubit amplitude state preparation
│   ├── qft.py                       # 1-D QFT / IQFT circuits
│   ├── qft_2d_qiskit.py             # Native Qiskit separable 2-D QFT / IQFT circuits on 8 qubits
│   ├── reconstruction.py            # State inversion, real spatial reconstruction & L2 error
│   ├── spectral_analysis.py         # 1-D spectral power and frequency metrics
│   └── spectral_analysis_2d.py      # Calibrated 2-D spectral second moment & radial frequency grid
├── diagnostics/
│   └── controller_decomposition.py  # Vector decomposition into APF gradient, goal, and avoidance
├── tests_v2/                        # === NEW v2 INDEPENDENT UNIT TEST SUITE ===
│   ├── test_electrostatic_potential.py  # Tests A, B, C: Potential bounds, forces, symmetry
│   ├── test_predictive_collision.py     # Test D: Head-on, crossing, parallel, receding, zero vel
│   ├── test_cbf_safety_shield.py        # Tests E, F: Barrier constraints, multi-obstacle, QP, limits
│   ├── test_safety_two_robots_head_on.py# Integration safety checkpoint: Two robots head-on
│   ├── test_quantum_pipeline.py         # Quantum pipeline precision & dynamic field integration
│   └── test_v1_regression.py            # Test G: 100% reproduction of all existing v1 test scripts
├── results/                         # Benchmark summaries (JSON) and step telemetry (CSV)
│   ├── v2_all_benchmarks.json       # Complete benchmark dataset across all scenarios and baselines
│   └── telemetry_*.csv              # Timestep telemetry logs for every experiment run
├── visualization/
│   ├── v2_dashboard.py              # Matplotlib multi-panel figure generator
│   ├── v2_experiment_results.png    # High-resolution benchmark dashboard image
│   └── aqsf_v2_interactive_dashboard.html # Standalone interactive HTML5 Canvas replay dashboard
├── run_v2_experiments.py            # Master benchmark runner executing Experiments 1 through 7
├── main.py                          # v1 Classical multi-agent simulation entry point
├── requirements.txt                 # Dependencies
└── README.md                        # Documentation
```

---

## 9. Prerequisites & Installation

- **Python**: 3.10+ (Tested on Python 3.11.6 on macOS Apple Silicon and Linux x86_64)
- **Virtual Environment**: Recommended

```bash
# 1. Clone repository
git clone https://github.com/ronsmessi1012/ASQF-NAV.git
cd ASQF-NAV

# 2. Activate virtual environment
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies
- `qiskit >= 2.6.0`
- `qiskit-aer >= 0.18.0`
- `numpy >= 2.0.0`
- `scipy >= 1.17.0`
- `matplotlib >= 3.10.0`

---

## 10. Execution Commands

### Running v1 Baseline Scripts (Preserved 100%)
```bash
# Run v1 classical simulation
python main.py

# Run v1 2-D quantum spectral validation
python test_spectral_module.py
python test_qiskit_2d_qft.py
python test_qiskit_complex_iqft.py

# Run v1 multi-agent filter ablation diagnostic (reproduces clearance degradation)
python test_quantum_multi_agent.py
```

### Running v2 Unit Test Suite
```bash
# Run all v2 independent unit tests and regression checks
python -m unittest discover -s tests_v2 -p "test_*.py"

# Or run individual component test suites:
python -m unittest tests_v2/test_electrostatic_potential.py
python -m unittest tests_v2/test_predictive_collision.py
python -m unittest tests_v2/test_cbf_safety_shield.py
python -m unittest tests_v2/test_safety_two_robots_head_on.py
python -m unittest tests_v2/test_quantum_pipeline.py
python -m unittest tests_v2/test_v1_regression.py
```

### Running v2 Comprehensive Experiments
```bash
# Execute Experiments 1 through 7 across all 5 baselines and export data
python run_v2_experiments.py
```

---

## 11. Visualization & Interactive Dashboard

### 1. Static Comparative Figure
```bash
# Generate high-resolution 9-panel comparative dashboard
python visualization/v2_dashboard.py
```
This saves `visualization/v2_experiment_results.png`, illustrating:
- Multi-robot trajectories under AQSF-NAV v2.
- Inter-robot clearance over time compared to safety distance $D_{ij} = 0.70$.
- Quantum spectral complexity $\mathcal{C}(t)$, adaptive bandwidth $B(t)$, power retention, and cumulative CBF interventions.

### 2. Standalone Interactive HTML5 Replay Dashboard
```bash
# Open the interactive HTML5 Canvas dashboard in your default browser:
open visualization/aqsf_v2_interactive_dashboard.html
```
**Features**:
- Interactive trajectory replay with scrubbing slider, play, pause, and step buttons.
- Scenario switcher (Head-on, Crossing, Moving Obstacle, Warehouse, Dense Stress, Sensitivity).
- Baseline comparator (Baseline A, B, C, D, E).
- Real-time display of circular safety regions, velocity vectors, and predicted closest approach indicators.
- Live telemetry panel updating separation distance, risk $\rho_{ij}$, CBF shield activations, and quantum spectral metrics.

---

## 12. Experimental Benchmark Results

The following table reflects **actual recorded numerical measurements** generated by executing `run_v2_experiments.py` across identical scenarios with deterministic seeds:

| Scenario | Baseline Configuration | Success | Steps | Min Robot Dist ($D_{ij} \ge 0.70$) | Min Obs Dist ($D_{io} \ge 0.80$) | Total Collisions | CBF Interventions | Field L2 Error | Wall Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 1: Head-On** | A: Classical APF | **FAIL** | 120 | 1.000 | N/A | 0 (Stalled) | 0 | 0.0000 | 0.013s |
| | B: APF + Electrostatic | **FAIL** | 53 | 0.200 | N/A | 2 | 0 | 0.0000 | 0.006s |
| | C: Predictive Dynamic | **PASS** | 54 | 0.827 | N/A | 0 | 0 | 0.0000 | 0.006s |
| | D: Predictive + CBF | **PASS** | 55 | 1.057 | N/A | 0 | 8 | 0.0000 | 0.008s |
| | E: **AQSF-NAV v2 (Quantum+CBF)** | **PASS** | 55 | **1.049** | N/A | **0** | **8** | 0.0027 | 0.173s |
| **Exp 2: Crossing** | A: Classical APF | **FAIL** | 54 | 0.105 | N/A | 3 | 0 | 0.0000 | 0.006s |
| | B: APF + Electrostatic | **FAIL** | 53 | 0.038 | N/A | 4 | 0 | 0.0000 | 0.006s |
| | C: Predictive Dynamic | **PASS** | 55 | 0.697 | N/A | 0 | 0 | 0.0000 | 0.006s |
| | D: Predictive + CBF | **PASS** | 55 | 0.838 | N/A | 0 | 9 | 0.0000 | 0.008s |
| | E: **AQSF-NAV v2 (Quantum+CBF)** | **PASS** | 55 | **0.836** | N/A | **0** | **10** | 0.0027 | 0.174s |
| **Exp 3: Moving Obs** | A: Classical APF | **PASS** | 53 | N/A | 1.765 | 0 | 0 | 0.0000 | 0.003s |
| | B: APF + Electrostatic | **PASS** | 53 | N/A | 1.977 | 0 | 0 | 0.0000 | 0.003s |
| | C: Predictive Dynamic | **PASS** | 53 | N/A | 1.977 | 0 | 0 | 0.0000 | 0.003s |
| | D: Predictive + CBF | **PASS** | 53 | N/A | 1.977 | 0 | 0 | 0.0000 | 0.003s |
| | E: **AQSF-NAV v2 (Quantum+CBF)** | **PASS** | 53 | N/A | **1.978** | **0** | **0** | 0.0032 | 0.083s |
| **Exp 4: Multi Obs** | A: Classical APF | **FAIL** | 150 | N/A | 0.212 | 6 | 0 | 0.0000 | 0.011s |
| | B: APF + Electrostatic | **FAIL** | 80 | N/A | 0.313 | 5 | 0 | 0.0000 | 0.017s |
| | C: Predictive Dynamic | **PASS** | 81 | N/A | 0.552 | 0 | 0 | 0.0000 | 0.008s |
| | D: Predictive + CBF | **PASS** | 84 | N/A | 0.785 | 0 | 8 | 0.0000 | 0.010s |
| | E: **AQSF-NAV v2 (Quantum+CBF)** | **PASS** | 84 | N/A | **0.780** | **0** | **10** | 0.0192 | 0.141s |
| **Exp 5: Warehouse** | A: Classical APF | **PASS** | 81 | 1.290 | 1.594 | 0 | 0 | 0.0000 | 0.017s |
| | B: APF + Electrostatic | **PASS** | 81 | 0.997 | 1.747 | 0 | 0 | 0.0000 | 0.018s |
| | C: Predictive Dynamic | **PASS** | 81 | 1.145 | 1.745 | 0 | 0 | 0.0000 | 0.018s |
| | D: Predictive + CBF | **PASS** | 82 | 1.183 | 1.745 | 0 | 2 | 0.0000 | 0.020s |
| | E: **AQSF-NAV v2 (Quantum+CBF)** | *Slow* | 160 | **1.351** | **1.622** | **0** | **2** | 0.1241 | 0.531s |
| **Exp 6: Dense Stress**| A: Classical APF | **FAIL** | 77 | 0.143 | 0.624 | 4 | 0 | 0.0000 | 0.020s |
| | B: APF + Electrostatic | **FAIL** | 77 | 0.079 | 0.361 | 10 | 0 | 0.0000 | 0.024s |
| | C: Predictive Dynamic | **FAIL** | 87 | 0.416 | 0.452 | 6 | 0 | 0.0000 | 0.028s |
| | D: Predictive + CBF | **PASS** | 88 | 0.743 | 0.797 | 0 | 23 | 0.0000 | 0.033s |
| | E: **AQSF-NAV v2 (Quantum+CBF)** | **PASS** | 101 | **0.842** | **0.754** | **0** | **39** | 0.0237 | 0.524s |

---

### Experiment 7: QFT Filtering Sensitivity Ablation Benchmark

| Exp 7 Configuration | Underlying Architecture | Success | Steps | Min Clearance | Collisions | CBF Interventions | Recon L2 Error | Scientific Finding |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Classical APF + CBF** | Baseline D (Classical) | **PASS** | 82 | 1.183 | 0 | 2 | 0.0000 | Baseline reference trajectory. |
| **Quantum Unfiltered + CBF** | 8-Qubit QFT + IQFT | **PASS** | 82 | 1.183 | 0 | 2 | $1.2 \times 10^{-15}$ | **Machine precision match** to classical baseline. |
| **Quantum Filtered without CBF**| Adaptive Filter ($\gamma = 3$) | **PASS** | 81 | 1.145 | 0 | 0 | 0.0000 | Avoidance relying only on potential field. |
| **AQSF-NAV v2 Full** | QFT + Filter + CBF | *Slow* | 160 | 1.351 | **0** | 2 | 0.1241 | **100% collision-free**, but shelf diffusion slows convergence. |

---

## 13. Analysis of Baselines (A through E)

1. **Baseline A (Classical APF)**: Lacks dynamic velocity anticipation. In direct head-on (Exp 1), it stalls in symmetric equilibrium. In crossing (Exp 2) and dense environments (Exp 6), it causes severe collisions (min separation down to $0.105$).
2. **Baseline B (APF + Electrostatic)**: Pure spatial potential repulsion causes high-speed chatter / oscillations in discrete integration when robots directly face each other, exacerbating collisions in dense dynamic traffic (10 collisions in Exp 6).
3. **Baseline C (Predictive Dynamic)**: Incorporating relative velocity closest-approach prediction ($t^*, d^*$) and lateral right-hand evasion resolves head-on and simple crossing scenarios, eliminating chatter. However, under dense multi-agent cross-traffic (Exp 6), predictive heuristics alone fail (6 collisions).
4. **Baseline D (Predictive + CBF)**: The addition of the Control Barrier Function safety shield provides strict mathematical velocity modification, achieving zero collisions across all scenarios including Exp 6.
5. **Baseline E (AQSF-NAV v2 Quantum + CBF)**: Integrates the 8-qubit separable 2-D QFT pipeline. Unfiltered QFT matches classical execution with machine precision ($10^{-15}$). Under adaptive filtering, the CBF safety shield prevents all collisions, successfully insulating multi-agent safety from spectral field distortions.

---

## 14. Known Limitations & Edge Cases

1. **Spectral Smoothing in Cluttered Aisles (Exp 5)**: In environments with narrow corridors surrounded by multiple static obstacles (e.g. warehouse shelves), aggressive low-pass filtering ($B \approx 0.25$) diffuses steep obstacle edges into broad, shallow gradient slopes. While collision safety is strictly preserved ($0$ collisions, min distance $1.351$), the attenuated goal gradient can slow down convergence near the goal.
2. **Discrete Time-Step Penetration Bound**: The continuous CBF condition ($2 r^T \Delta v + \alpha h \ge 0$) assumes continuous integration. Under large discrete time steps ($\Delta t = 0.20$), fast-moving obstacles may cause small numerical discretisation encroachments if $\alpha$ is tuned too low.
3. **Reciprocal Safety Assumption**: In decentralized multi-robot encounters, each robot assumes reciprocal avoidance responsibility ($\kappa = 0.5$). If an interacting robot fails to cooperate (unresponsive agent), the ego-robot's fallback mode must escalate to non-cooperative mode ($\kappa = 1.0$).

---

## 15. Scientific Reproducibility

Every experiment and numerical metric reported in this project is strictly deterministic and reproducible:
- Random seeds are fixed.
- All simulation steps and trajectories are logged to CSV in `results/`.
- All benchmark summaries are exported to `results/v2_all_benchmarks.json`.
- The entire benchmark can be re-run and verified on any platform with:
  ```bash
  python run_v2_experiments.py
  ```

---

## 16. Future Research Directions

1. **Corridor-Preserving Wavelet/Spectral Filter**: Designing a directional or wavelet spectral filter that selectively filters high-frequency inter-robot phase chatter while preserving the sharp spatial gradients of static walls and warehouse shelves.
2. **Adaptive Reciprocity Estimation**: Real-time estimation of interacting agents' cooperativeness to dynamically adjust the CBF reciprocity weight $\kappa \in [0.5, 1.0]$.
3. **Hardware Emulation**: Transitioning from Qiskit Aer statevector simulation to noisy quantum circuit emulation (e.g., measuring sample complexity under Pauli error channels).

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
