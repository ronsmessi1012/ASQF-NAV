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
6. [Mathematical Formulation of AQSF-NAV](#mathematical-formulation-of-aqsf-nav)
   - [1. Environment and Robot Representation](#1-environment-and-robot-representation)
   - [2. Original Artificial Potential Field (APF)](#2-original-artificial-potential-field-apf)
   - [3. AQSF-NAV v2: Electrostatic-Inspired Dynamic Protective Field](#3-aqsf-nav-v2-electrostatic-inspired-dynamic-protective-field)
   - [4. Predictive Collision Detection](#4-predictive-collision-detection)
   - [5. Combined Navigation Potential Field](#5-combined-navigation-potential-field)
   - [6. Amplitude Encoding](#6-amplitude-encoding)
   - [7. Two-Dimensional Quantum Fourier Transform](#7-two-dimensional-quantum-fourier-transform)
   - [8. Spectral Complexity](#8-spectral-complexity)
   - [9. Complexity Normalization](#9-complexity-normalization)
   - [10. Adaptive Spectral Bandwidth](#10-adaptive-spectral-bandwidth)
   - [11. Inverse Quantum Fourier Transform](#11-inverse-quantum-fourier-transform)
   - [12. Gradient-Based Navigation](#12-gradient-based-navigation)
   - [13. Predictive Collision Analysis](#13-predictive-collision-analysis)
   - [14. Control Barrier Function (CBF) Safety Shield](#14-control-barrier-function-cbf-safety-shield)
   - [15. Performance Evaluation Metrics](#15-performance-evaluation-metrics)
   - [16. Quantum Resource Metrics](#16-quantum-resource-metrics)
   - [17. Experimental Comparison](#17-experimental-comparison)
   - [18. Scientific Limitations and Assumptions](#18-scientific-limitations-and-assumptions)
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

# MATHEMATICAL FORMULATION OF AQSF-NAV

This section provides the rigorous mathematical formulation governing the hybrid quantum-classical navigation framework, covering both the baseline v1 spectral pipeline and the v2 Predictive Electrostatic Safety Architecture. Every variable, parameter, physical assumption, and governing equation is defined below.

---

## 1. Environment and Robot Representation

The workspace is defined as a bounded continuous two-dimensional Euclidean domain $\mathcal{W} \subset \mathbb{R}^2$. Within this environment, mobile robots, static obstacles, and dynamic moving obstacles operate concurrently.

### 1.1 Robot Kinematics

For an autonomous robot $i \in \{1, \dots, N\}$, its instantaneous state at continuous time $t \ge 0$ is governed by planar kinematics.

**Robot Position:**
$$\mathbf{p}_i(t) = \begin{bmatrix} x_i(t) \\ y_i(t) \end{bmatrix} \in \mathbb{R}^2$$
Where $x_i(t)$ and $y_i(t)$ denote Cartesian coordinates in world units.

**Robot Velocity:**
$$\mathbf{v}_i(t) = \begin{bmatrix} v_{xi}(t) \\ v_{yi}(t) \end{bmatrix} = \frac{d\mathbf{p}_i(t)}{dt} \in \mathbb{R}^2$$
The robot speed is bounded by a maximum actuator limit:
$$\|\mathbf{v}_i(t)\|_2 = \sqrt{v_{xi}^2(t) + v_{yi}^2(t)} \le v_{\max}$$
Where $v_{\max} = 1.0$ world units per second.

**Robot Acceleration:**
$$\mathbf{a}_i(t) = \frac{d\mathbf{v}_i(t)}{dt} = \frac{d^2\mathbf{p}_i(t)}{dt^2} \in \mathbb{R}^2$$

### 1.2 Static Obstacle Representation

A static obstacle $j \in \{1, \dots, M_{\text{static}}\}$ occupies a fixed location in the workspace:
$$\mathbf{o}_j = \begin{bmatrix} x_j^o \\ y_j^o \end{bmatrix} \in \mathbb{R}^2$$
Because the obstacle is stationary, its time derivative is identically zero:
$$\frac{d\mathbf{o}_j}{dt} = \mathbf{0}$$

### 1.3 Moving Obstacle Representation

A moving obstacle $j \in \{1, \dots, M_{\text{moving}}\}$ is an active dynamic agent with time-varying coordinates:
$$\mathbf{q}_j(t) = \begin{bmatrix} x_j^o(t) \\ y_j^o(t) \end{bmatrix} \in \mathbb{R}^2$$
Its instantaneous translational velocity is:
$$\mathbf{u}_j(t) = \frac{d\mathbf{q}_j(t)}{dt} = \begin{bmatrix} u_{xj}(t) \\ u_{yj}(t) \end{bmatrix} \in \mathbb{R}^2$$
Under a constant-velocity (CV) motion model over a finite increment $\Delta t$:
$$\mathbf{q}_j(t + \Delta t) = \mathbf{q}_j(t) + \mathbf{u}_j(t)\Delta t$$
For piecewise waypoint or scripted trajectories, $\mathbf{u}_j(t)$ is updated deterministically at each simulation epoch.

### 1.4 Discrete-Time Position Update

The simulation integrates continuous robot motion using discrete first-order forward Euler integration:
$$\mathbf{p}_i(t + \Delta t) = \mathbf{p}_i(t) + \mathbf{v}_i(t)\Delta t$$
Where $\Delta t = 0.20$ seconds is the constant simulation time step, preserved from the original v1 architecture.

---

## 2. Original Artificial Potential Field (APF)

The static environment in AQSF-NAV is represented via classical Artificial Potential Fields (APF). The static field consists of an attractive well pulling the robot toward its destination and repulsive ridges deflecting the robot away from static boundaries.

### 2.1 Goal Attraction Potential

For robot $i$ navigating toward target destination $\mathbf{g}_i = [g_{xi}, g_{yi}]^T \in \mathbb{R}^2$, the attractive potential is modeled as a parabolic well:
$$U_{\text{goal},i}(\mathbf{p}_i) = \frac{1}{2} K_g \|\mathbf{p}_i - \mathbf{g}_i\|^2$$
Where:
- $K_g$: Positive goal attraction gain coefficient ($K_g = 2.0$).
- $\mathbf{p}_i$: Instantaneous continuous robot coordinate.
- $\mathbf{g}_i$: Designated target destination.

The corresponding attractive force is the negative spatial gradient:
$$\mathbf{F}_{\text{goal},i} = -\nabla_{\mathbf{p}_i} U_{\text{goal},i}$$
Differentiating with respect to $\mathbf{p}_i$:
$$\mathbf{F}_{\text{goal},i} = -K_g (\mathbf{p}_i - \mathbf{g}_i)$$
This force acts as a linear spring pulling robot $i$ toward $\mathbf{g}_i$.

### 2.2 Static Obstacle Repulsion

For static obstacle $j$ at coordinate $\mathbf{o}_j$, the Euclidean distance to robot $i$ is:
$$d_{ij} = \|\mathbf{p}_i - \mathbf{o}_j\|$$
The classical inverse-distance repulsive potential is active only within influence threshold $d_0$:
$$U_{\text{static},ij}(d) = \begin{cases} \displaystyle \frac{1}{2} K_o \left(\frac{1}{d} - \frac{1}{d_0}\right)^2, & 0 < d < d_0 \\[8pt] 0, & d \ge d_0 \end{cases}$$
Where:
- $K_o$: Obstacle repulsion gain coefficient ($K_o = 80.0$).
- $d_0$: Maximum spatial influence distance ($d_0 = \rho_{\text{obs}} = 2.0$).
- $d$: Continuous distance to obstacle center.

To prevent infinite potential singularities when evaluated on a discrete grid, a singularity clamping function $d^* = \max(d, 0.5)$ is applied on the meshgrid.

The repulsive force vector is:
$$\mathbf{F}_{\text{static},ij} = -\nabla_{\mathbf{p}_i} U_{\text{static},ij}$$
For $0 < d < d_0$, applying the chain rule:
$$\mathbf{F}_{\text{static},ij} = K_o \left(\frac{1}{d} - \frac{1}{d_0}\right) \frac{\mathbf{p}_i - \mathbf{o}_j}{d^3}$$
For $d \ge d_0$, $\mathbf{F}_{\text{static},ij} = \mathbf{0}$.

### 2.3 Total Original Static APF

Superimposing the attractive field and all static obstacle barriers yields the complete static scalar potential:
$$U_{\text{static},i} = U_{\text{goal},i} + \sum_{j=1}^{M_{\text{static}}} U_{\text{static},ij}$$
This static APF formulation is preserved exactly from v1 and remains unchanged in v2.

---

## 3. AQSF-NAV v2: Electrostatic-Inspired Dynamic Protective Field

To avoid the multi-agent collision vulnerabilities identified in v1, v2 introduces an electrostatic-inspired dynamic protective field dedicated strictly to moving entities.

### 3.1 Dynamic Entity Representation

Dynamic entities surrounding robot $i$ include:
1. Other active peer robots: $j \in \mathcal{R} \setminus \{i\}$.
2. Dynamic moving obstacles: $j \in \mathcal{O}_{\text{moving}}$.

For dynamic entity $j$:
- $\mathbf{q}_j(t) \in \mathbb{R}^2$: Instantaneous position.
- $\mathbf{u}_j(t) \in \mathbb{R}^2$: Instantaneous velocity.

The relative displacement vector from entity $j$ to robot $i$ is:
$$\mathbf{r}_{ij} = \mathbf{p}_i - \mathbf{q}_j$$
The current Euclidean separation distance is:
$$d_{ij} = \|\mathbf{r}_{ij}\|_2 = \sqrt{(x_i - x_j^o)^2 + (y_i - y_j^o)^2}$$

### 3.2 Effective Safety Distance

The physical collision boundary accounts for body radii and a conservative safety cushion:
$$D_{ij} = R_i + R_j + m$$
Where:
- $R_i$: Physical radius of robot $i$ ($R_i = 0.25$ world units).
- $R_j$: Physical radius of entity $j$ ($R_j = 0.25$ for peer robots; configured per moving obstacle).
- $m$: Additional safety margin ($m = 0.20$ world units).

For two robots:
$$D_{ij} = 0.25 + 0.25 + 0.20 = 0.70 \text{ world units}$$

### 3.3 Dynamic Influence Radius

The dynamic protective field activates before encroaching on the safety boundary:
$$L_{ij} = D_{ij} + \Delta L > D_{ij}$$
Where $\Delta L = 2.0$ world units. Within the shell $D_{ij} < d_{ij} < L_{ij}$, repulsive forces deflect the robot away from oncoming entities.

### 3.4 Electrostatic-Inspired Repulsive Potential

The scalar potential mimics Coulombic electrostatic repulsion between like charges:
$$U_{\text{dynamic},ij}(d) = \begin{cases} \displaystyle \frac{k_{ij}}{2} \left(\frac{1}{d} - \frac{1}{L_{ij}}\right)^2, & 0 < d < L_{ij} \\[8pt] 0, & d \ge L_{ij} \end{cases}$$
Where $k_{ij}$ is the configurable repulsion stiffness:
- $k_{\text{robot}} = 15.0$ for peer robot interactions.
- $k_{\text{obstacle}} = 25.0$ for dynamic moving obstacles.

*Note: This is a mathematical navigation analogy and does not physically generate electric flux or electromagnetic fields.*

### 3.5 Dynamic Repulsive Force

The repulsive force vector is derived analytically as the negative gradient with respect to $\mathbf{p}_i$:
$$\mathbf{F}_{\text{dynamic},ij} = -\nabla_{\mathbf{p}_i} U_{\text{dynamic},ij}$$
For $0 < d_{ij} < L_{ij}$:
$$\boxed{\mathbf{F}_{\text{dynamic},ij} = k_{ij} \left(\frac{1}{d_{ij}} - \frac{1}{L_{ij}}\right) \frac{\mathbf{r}_{ij}}{d_{ij}^{3}}}$$
Outside the influence radius ($d_{ij} \ge L_{ij}$):
$$\mathbf{F}_{\text{dynamic},ij} = \mathbf{0}$$

**Numerical Safeguards**:
- Distance clamping: $d_{\text{clamped}} = \max(d_{ij}, \epsilon)$ with $\epsilon = 10^{-4}$ (continuous dynamics) and $\epsilon = 0.5$ (discrete grid).
- Force saturation: If $\|\mathbf{F}_{\text{dynamic},ij}\|_2 > F_{\max}$, $\mathbf{F} \leftarrow (\mathbf{F} / \|\mathbf{F}\|) F_{\max}$ with $F_{\max} = 30.0$.
- Coincident handler: If $d_{ij} < \epsilon$, a deterministic orthogonal unit vector is assigned, preventing NaN and infinite force spikes.

### 3.6 Total Dynamic Protective Force

For robot $i$, the total dynamic force aggregates all nearby robots and moving obstacles:
$$\mathbf{F}_{\text{dynamic},i} = \sum_{j \in \mathcal{R}, j \neq i} \mathbf{F}_{\text{dynamic},ij} + \sum_{j \in \mathcal{O}_{\text{moving}}} \mathbf{F}_{\text{dynamic},ij}$$

---

## 4. Predictive Collision Detection

Static distance alone cannot foresee whether two converging agents will collide or pass harmlessly. Relative velocity analysis provides forward-looking collision anticipation.

### 4.1 Relative Position

$$\mathbf{r}_{ij} = \mathbf{p}_i - \mathbf{q}_j$$

### 4.2 Relative Velocity

$$\mathbf{v}_{\text{rel},ij} = \mathbf{v}_i - \mathbf{u}_j$$
Where $\mathbf{v}_i$ is the velocity of robot $i$ and $\mathbf{u}_j$ is the velocity of dynamic entity $j$.

### 4.3 Prediction Horizon

Let $T > 0$ denote the forward evaluation horizon ($T = 2.0$ simulation seconds).

### 4.4 Time of Closest Approach

Assuming constant relative velocity over horizon $T$, the squared relative distance at future time $\tau \in [0, T]$ is:
$$D^2(\tau) = \|\mathbf{r}_{ij} + \tau \mathbf{v}_{\text{rel},ij}\|^2 = \|\mathbf{r}_{ij}\|^2 + 2\tau (\mathbf{r}_{ij}^T \mathbf{v}_{\text{rel},ij}) + \tau^2 \|\mathbf{v}_{\text{rel},ij}\|^2$$
Minimizing $D^2(\tau)$ with respect to $\tau$ and clipping to the interval $[0, T]$ yields:
$$\boxed{t_{ij}^{*} = \operatorname{clip}\left(-\frac{\mathbf{r}_{ij}^{T}\mathbf{v}_{\text{rel},ij}}{\|\mathbf{v}_{\text{rel},ij}\|^2}, 0, T\right)}$$
**Special Case**: If $\|\mathbf{v}_{\text{rel},ij}\|^2 < 10^{-10}$ (near-zero relative motion), $t_{ij}^* = 0.0$.

### 4.5 Predicted Minimum Separation

Evaluating separation at the critical epoch $t_{ij}^*$:
$$\boxed{d_{ij}^{*} = \left\|\mathbf{r}_{ij} + t_{ij}^{*}\mathbf{v}_{\text{rel},ij}\right\|_2}$$

### 4.6 Predicted Collision Risk

The predictive risk score quantifies the fractional safety violation expected at closest approach:
$$\boxed{\rho_{ij} = \max\left(0, \frac{D_{ij} - d_{ij}^{*}}{D_{ij}}\right) \in [0, 1]}$$
- $\rho_{ij} = 0$: No predicted safety distance violation within horizon $T$ ($d_{ij}^* \ge D_{ij}$).
- $0 < \rho_{ij} < 1$: Impending safety distance violation.
- $\rho_{ij} \to 1$: Imminent direct collision ($d_{ij}^* \to 0$).

*Interpretation Note: The risk score $\rho_{ij}$ is a geometric heuristic indicating kinematic urgency and must not be interpreted as an uncalibrated probability.*

### 4.7 Combined Dynamic Risk

Spatial potential and predictive urgency remain mathematically decoupled:
$$U_{\text{dynamic},ij} = U_{\text{spatial},ij} + U_{\text{predictive},ij}$$
The spatial electrostatic potential generates smooth physical repulsion across the space, while $\rho_{ij}$ determines lateral evasive bias:
$$\mathbf{v}_{\text{evade}} = \rho_{ij} \cdot 1.2 \cdot \hat{\mathbf{n}}_{\text{right}}$$
Where $\hat{\mathbf{n}}_{\text{right}} = [d_{\text{goal}, y}, -d_{\text{goal}, x}]$ applies right-hand maritime clearance to break head-on symmetry without differentiating velocity terms as position potentials.

---

## 5. Combined Navigation Potential Field

### 5.1 Original Static Field

$$U_{\text{static},i} = U_{\text{goal},i} + \sum_{j \in \mathcal{O}_{\text{static}}} U_{\text{static},ij}$$

### 5.2 Dynamic Field

$$U_{\text{dynamic},i} = \sum_{j \in \mathcal{R}, j \neq i} U_{\text{dynamic},ij} + \sum_{j \in \mathcal{O}_{\text{moving}}} U_{\text{dynamic},ij}$$

### 5.3 Total Navigation Field

$$\boxed{U_{\text{total},i} = U_{\text{static},i} + U_{\text{dynamic},i}}$$
This combined field is discretized onto the $16 \times 16$ spatial grid and serves as the input to the quantum spectral processing pipeline.

---

## 6. Amplitude Encoding

### 6.1 Field Dimensions

The continuous spatial potential field is sampled on a uniform square grid of size $16 \times 16$:
$$N = N_x \times N_y = 16 \times 16 = 256 \text{ grid points}$$

### 6.2 Number of Qubits

The number of qubits $n$ required to encode $N$ amplitudes is:
$$n = \log_2(N)$$
$$\boxed{n = \log_2(256) = 8 \text{ qubits}}$$

### 6.3 Flattening

The two-dimensional potential matrix $U_{\text{total}}(x, y)$ is flattened into a one-dimensional vector $\mathbf{f} \in \mathbb{R}^{256}$ using row-major ordering:
$$\mathbf{f} = \operatorname{vec}(U_{\text{total}}), \quad f_{16y + x} = U_{\text{total}}(x, y)$$

### 6.4 L2 Normalization

The Euclidean norm of the scalar field vector is:
$$\|\mathbf{f}\|_2 = \sqrt{\sum_{j=0}^{255} |f_j|^2}$$
Normalized quantum state amplitudes:
$$\boxed{a_j = \frac{f_j}{\|\mathbf{f}\|_2}}$$
Satisfying the quantum unit-norm axiom:
$$\sum_{j=0}^{255} |a_j|^2 = 1.0$$

### 6.5 Quantum State

The field is prepared as an 8-qubit pure statevector in Hilbert space $\mathcal{H}_2^{\otimes 8}$:
$$\boxed{|\psi\rangle = \sum_{j=0}^{255} a_j |j\rangle}$$
Where $|j\rangle = |y_3 y_2 y_1 y_0 x_3 x_2 x_1 x_0\rangle$ denotes computational basis states.

---

## 7. Two-Dimensional Quantum Fourier Transform

### 7.1 One-Dimensional QFT

The 1-D Quantum Fourier Transform on an $N$-dimensional register maps basis states $|x\rangle$ to frequency superpositions:
$$QFT_N |x\rangle = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} \exp\left(\frac{2\pi i x k}{N}\right) |k\rangle$$

### 7.2 Two-Dimensional QFT

For a 2-D spatial grid, the separable 2-D QFT transforms both coordinates independently:
$$\boxed{S(k_x, k_y) = \frac{1}{\sqrt{N_x N_y}} \sum_{x=0}^{N_x-1} \sum_{y=0}^{N_y-1} U(x, y) \exp\left[2\pi i \left(\frac{k_x x}{N_x} + \frac{k_y y}{N_y}\right)\right]}$$
For $N_x = N_y = 16$:
$$QFT_{2D} = QFT_x \otimes QFT_y$$
In the 8-qubit Qiskit implementation, qubits 0–3 encode the $X$-dimension and qubits 4–7 encode the $Y$-dimension. Due to Qiskit's little-endian indexing, logical registers are applied in reverse order (`x_qubits=[3,2,1,0]`, `y_qubits=[7,6,5,4]`), ensuring machine-precision agreement with 2-D NumPy FFT ($< 1.35 \times 10^{-15}$).

### 7.3 Spectral Power

The spectral power at spatial frequency $(k_x, k_y)$ is:
$$\boxed{P(k_x, k_y) = |S(k_x, k_y)|^2}$$

### 7.4 Parseval's Relation

Because the QFT is a unitary transformation, total power is conserved:
$$\sum_{x=0}^{N_x-1} \sum_{y=0}^{N_y-1} |U(x, y)|^2 = \sum_{k_x=0}^{N_x-1} \sum_{k_y=0}^{N_y-1} |S(k_x, k_y)|^2 = 1.0$$
Providing an exact numerical sanity check on the simulated quantum circuit.

---

## 8. Spectral Complexity

### 8.1 Spatial Frequency Coordinates

Spatial frequencies are measured radially from the DC origin ($k_x = 0, k_y = 0$):
$$r(k_x, k_y) = \sqrt{k_x^2 + k_y^2}$$
Where $k_x, k_y \in [-N/2, N/2 - 1]$ follow standard centered discrete frequency indexing.

### 8.2 Maximum Radial Frequency

$$r_{\max} = \max_{k_x, k_y} r(k_x, k_y) = \sqrt{\left(\frac{N_x}{2}\right)^2 + \left(\frac{N_y}{2}\right)^2} = \sqrt{8^2 + 8^2} = 8\sqrt{2} \approx 11.3137$$

### 8.3 Spectral Second-Moment Complexity

Environmental clutter and high-frequency turbulence are quantified by the **Normalized Spectral Second Moment** ($C$):
$$\boxed{C = \frac{\sum_{k_x, k_y} r(k_x, k_y)^2 P(k_x, k_y)}{r_{\max}^2 \sum_{k_x, k_y} P(k_x, k_y)}}$$
- Lower $C$: Spectral energy is concentrated at low frequencies (smooth, open environments).
- Higher $C$: Substantial energy resides in high spatial frequencies (dense obstacles, sharp gradients).

---

## 9. Complexity Normalization

From the 50-environment calibration study, the 25th and 75th percentiles of $C$ are:
$$C_{25} = 0.01645914$$
$$C_{75} = 0.01987830$$

The standardized unit complexity $C_n \in [0, 1]$ is obtained via percentile clipping:
$$\boxed{C_n = \operatorname{clip}\left(\frac{C - C_{25}}{C_{75} - C_{25}}, 0, 1\right)}$$
*Note: These calibration parameters reflect the v1 static distribution. Dynamic environments with multiple moving agents shift the second moment higher, as documented in v2 experiments.*

---

## 10. Adaptive Spectral Bandwidth

The low-pass filter bandwidth $B$ adapts dynamically to normalized environmental complexity:
$$\boxed{B = B_{\max} - (B_{\max} - B_{\min}) C_n^\gamma}$$
With calibrated parameters:
$$B_{\min} = 0.25, \quad B_{\max} = 0.75, \quad \gamma = 3$$
Yielding the cubic bandwidth response:
$$\boxed{B = 0.75 - 0.50 C_n^3}$$

### 10.1 Frequency Cutoff

$$k_c = B \cdot r_{\max}$$

### 10.2 Radial Filtering Mask

$$M(k_x, k_y) = \begin{cases} 1, & r(k_x, k_y) \le k_c \\ 0, & r(k_x, k_y) > k_c \end{cases}$$

### 10.3 Filtered Spectrum

$$\boxed{S'(k_x, k_y) = M(k_x, k_y) S(k_x, k_y)}$$

### 10.4 Retained Spectral Power

$$\boxed{P_{\text{retained}} = \frac{\sum_{k_x, k_y} |S'(k_x, k_y)|^2}{\sum_{k_x, k_y} |S(k_x, k_y)|^2}}$$

### 10.5 Retained Component Fraction

$$\boxed{R_{\text{components}} = \frac{\sum_{k_x, k_y} M(k_x, k_y)}{N_x N_y} = \frac{\sum M}{256}}$$

---

## 11. Inverse Quantum Fourier Transform

### 11.1 Inverse QFT Circuit

The inverse transform restores frequency states back to spatial computational basis states:
$$QFT_N^{-1} |k\rangle = \frac{1}{\sqrt{N}} \sum_{x=0}^{N-1} \exp\left(-\frac{2\pi i x k}{N}\right) |x\rangle$$

### 11.2 Reconstructed Field

$$\boxed{U'(x, y) = IQFT_{2D}\left(S'(k_x, k_y)\right)}$$

### 11.3 Reconstruction Error

The relative $L_2$ spatial field distortion introduced by spectral filtering is:
$$\boxed{E_{\text{reconstruction}} = \frac{\|U - U'\|_2}{\|U\|_2} = \frac{\sqrt{\sum_{x,y} |U(x,y) - U'(x,y)|^2}}{\sqrt{\sum_{x,y} |U(x,y)|^2}}}$$

### 11.4 Imaginary Residual

Physical navigation requires real-valued potentials. The residual imaginary component from complex IQFT inversion is verified to be negligible ($< 10^{-15}$):
$$\boxed{E_{\text{imag}} = \|\operatorname{Im}(U')\|_2 \le 1.91 \times 10^{-16}}$$
The real component $\operatorname{Re}(U'(x, y))$ is scaled by $\|\mathbf{f}\|_2$ and passed to the controller.

---

## 12. Gradient-Based Navigation

### 12.1 Potential Gradient

At continuous coordinate $\mathbf{p} = (x, y)$, the spatial gradient $\nabla U'$ is sampled using continuous bilinear interpolation over the discrete reconstructed grid:
$$\nabla U' = \begin{bmatrix} \frac{\partial U'}{\partial x} \\ \frac{\partial U'}{\partial y} \end{bmatrix} = \operatorname{BilinearInterpolate}\left(\nabla U'_{\text{grid}}, x, y\right)$$

### 12.2 Navigation Direction

$$\boxed{\mathbf{d}_{APF} = -\nabla U'}$$

### 12.3 Normalized APF Direction

$$\hat{\mathbf{d}}_{APF} = \begin{cases} \displaystyle \frac{-\nabla U'}{\|\nabla U'\|_2}, & \|\nabla U'\|_2 > 10^{-10} \\[8pt] \mathbf{0}, & \|\nabla U'\|_2 \le 10^{-10} \end{cases}$$

### 12.4 Goal Direction

$$\hat{\mathbf{d}}_{\text{goal}} = \begin{cases} \displaystyle \frac{\mathbf{g}_i - \mathbf{p}_i}{\|\mathbf{g}_i - \mathbf{p}_i\|_2}, & \|\mathbf{g}_i - \mathbf{p}_i\|_2 > 10^{-10} \\[8pt] \mathbf{0}, & \|\mathbf{g}_i - \mathbf{p}_i\|_2 \le 10^{-10} \end{cases}$$

### 12.5 Combined Desired Direction

Blending field gradient descent and line-of-sight goal attraction:
$$\boxed{\mathbf{v}_{\text{base}} = 0.65 \hat{\mathbf{d}}_{APF} + 0.35 \hat{\mathbf{d}}_{\text{goal}}}$$
Candidate desired velocity normalized to maximum robot speed $v_{\max}$:
$$\mathbf{v}_{\text{desired}} = \left(\frac{\mathbf{v}_{\text{base}} + \mathbf{v}_{\text{evade}}}{\|\mathbf{v}_{\text{base}} + \mathbf{v}_{\text{evade}}\|_2}\right) v_{\max}$$

### 12.6 Position Update

$$\boxed{\mathbf{p}_i(t + \Delta t) = \mathbf{p}_i(t) + \Delta t \, \mathbf{v}_{\text{safe}, i}(t)}$$

---

## 13. Predictive Collision Analysis

Summary of relative motion equations for robot $i$ and dynamic entity $j$:

### 13.1 Relative Position

$$\mathbf{r}_{ij} = \mathbf{p}_i - \mathbf{q}_j$$

### 13.2 Relative Velocity

$$\mathbf{v}_{\text{rel},ij} = \mathbf{v}_i - \mathbf{u}_j$$

### 13.3 Time of Closest Approach

$$\boxed{t_{ij}^{*} = \operatorname{clip}\left(-\frac{\mathbf{r}_{ij}^{T}\mathbf{v}_{\text{rel},ij}}{\|\mathbf{v}_{\text{rel},ij}\|^2}, 0, T\right)}$$
If $\|\mathbf{v}_{\text{rel},ij}\|^2 < 10^{-10}$, then $t_{ij}^* = 0.0$.

### 13.4 Predicted Minimum Distance

$$\boxed{d_{ij}^{*} = \|\mathbf{r}_{ij} + t_{ij}^{*}\mathbf{v}_{\text{rel},ij}\|_2}$$

### 13.5 Collision Risk

$$\boxed{\rho_{ij} = \max\left(0, \frac{D_{ij} - d_{ij}^{*}}{D_{ij}}\right)}$$

---

## 14. Control Barrier Function (CBF) Safety Shield

### 14.1 Safety Function

For robot $i$ and dynamic entity $j$:
$$\boxed{h_{ij}(\mathbf{p}_i, \mathbf{q}_j) = \|\mathbf{p}_i - \mathbf{q}_j\|^2 - D_{ij}^2}$$
The safe set is defined as the zero-superlevel set $\mathcal{C} = \{\mathbf{p}_i, \mathbf{q}_j \mid h_{ij} \ge 0\}$.

### 14.2 Time Derivative

$$\dot{h}_{ij} = \frac{\partial h_{ij}}{\partial \mathbf{p}_i} \dot{\mathbf{p}}_i + \frac{\partial h_{ij}}{\partial \mathbf{q}_j} \dot{\mathbf{q}}_j = 2(\mathbf{p}_i - \mathbf{q}_j)^T (\mathbf{v}_i - \mathbf{u}_j) = 2 \mathbf{r}_{ij}^T (\mathbf{v}_i - \mathbf{u}_j)$$

### 14.3 Control Barrier Condition

By Nagumo's theorem and continuous-time barrier theory, forward invariance of set $\mathcal{C}$ is guaranteed if:
$$\boxed{\dot{h}_{ij} + \alpha h_{ij} \ge 0}$$
Expanding with relative state vectors:
$$\boxed{2\mathbf{r}_{ij}^{T}(\mathbf{v}_i - \mathbf{u}_j) + \alpha h_{ij} \ge 0}$$
Where $\alpha > 0$ is a positive linear class-$\mathcal{K}$ parameter ($\alpha = 2.0$).

For two active peer robots both capable of control action:
$$\boxed{2\mathbf{r}_{ij}^{T}(\mathbf{v}_i - \mathbf{v}_j) + \alpha h_{ij} \ge 0}$$

### 14.4 Quadratic Programming Velocity Correction

The safety shield projects candidate velocity $\mathbf{v}_{\text{desired}}$ onto the feasible safe velocity polytope via Quadratic Programming:
$$\boxed{\min_{\mathbf{v} \in \mathbb{R}^2} \frac{1}{2} \|\mathbf{v} - \mathbf{v}_{\text{desired}}\|^2}$$
Subject to:
$$2\mathbf{r}_{ij}^{T} (\mathbf{v} - \mathbf{u}_j) + \alpha h_{ij} \ge 0 \quad (\forall j \in \mathcal{O}_{\text{moving}})$$
$$2\mathbf{r}_{ij}^{T} \mathbf{v} \ge 2\mathbf{r}_{ij}^{T} \mathbf{v}_j - 2\kappa \alpha h_{ij} \quad (\forall j \in \mathcal{R} \setminus \{i\})$$
$$\|\mathbf{v}\|_2 \le v_{\max}$$
Where $\kappa = 0.5$ allocates equal reciprocal avoidance responsibility in decentralized mode. Solved in $< 1 \text{ ms}$ via SciPy `SLSQP`.

### 14.5 Safety Constraint Feasibility

Define the barrier residual for constraint $j$:
$$s_{ij} = 2\mathbf{r}_{ij}^{T}(\mathbf{v}_i - \mathbf{u}_j) + \alpha h_{ij}$$
Feasibility is verified up to numerical tolerance: $s_{ij} \ge -\epsilon_{\text{tol}}$ ($\epsilon_{\text{tol}} = 10^{-6}$). If opposing obstacles make the constraint polytope empty (infeasible trap), the solver reports an infeasible alert and commands a maximum-clearance deceleration or emergency stop ($v \to 0$).

---

## 15. Performance Evaluation Metrics

### 15.1 Path Length

$$L_i = \sum_{t=0}^{T_{\text{steps}}-1} \|\mathbf{p}_i(t+1) - \mathbf{p}_i(t)\|_2$$

### 15.2 Total Multi-Agent Path Length

$$L_{\text{total}} = \sum_{i=1}^{N} L_i$$

### 15.3 Minimum Inter-Agent Distance

$$\boxed{d_{\min}^{\text{robots}} = \min_{t} \min_{i \neq j} \|\mathbf{p}_i(t) - \mathbf{p}_j(t)\|_2}$$

### 15.4 Minimum Robot-to-Moving-Obstacle Distance

$$\boxed{d_{\min}^{\text{obstacles}} = \min_{t} \min_{i, j} \|\mathbf{p}_i(t) - \mathbf{q}_j(t)\|_2}$$

### 15.5 Collision Count

Let physical collision occur if separation breaches physical boundary $R_i + R_j$:
$$I_{ij}(t) = \begin{cases} 1, & d_{ij}(t) < R_i + R_j \\ 0, & \text{otherwise} \end{cases}$$
Collisions are counted as distinct continuous collision episodes.

### 15.6 Goal Success Rate

$$\boxed{\text{SuccessRate} = \frac{N_{\text{successful robots}}}{N_{\text{total robots}}} \times 100\%}$$

### 15.7 Average Completion Time

$$T_{\text{avg}} = \frac{1}{N_{\text{successful robots}}} \sum_{i=1}^{N_{\text{success}}} T_i$$

### 15.8 Safety Intervention Rate

$$\text{InterventionRate} = \frac{N_{\text{CBF interventions}}}{N_{\text{control steps}}}$$

### 15.9 Average Reconstruction Error

$$E_{\text{avg}} = \frac{1}{T_{\text{steps}}} \sum_{t=1}^{T_{\text{steps}}} E_{\text{reconstruction}}(t)$$

### 15.10 Average Spectral Complexity

$$C_{\text{avg}} = \frac{1}{T_{\text{steps}}} \sum_{t=1}^{T_{\text{steps}}} C(t)$$

### 15.11 Average Adaptive Bandwidth

$$B_{\text{avg}} = \frac{1}{T_{\text{steps}}} \sum_{t=1}^{T_{\text{steps}}} B(t)$$

### 15.12 Gradient Direction Similarity

Angular fidelity between classical potential gradient $\nabla U_{\text{classical}}$ and quantum reconstructed gradient $\nabla U_{\text{reconstructed}}$:
$$\boxed{\cos\theta = \frac{\nabla U_{\text{classical}}^T \nabla U_{\text{reconstructed}}}{\|\nabla U_{\text{classical}}\|_2 \|\nabla U_{\text{reconstructed}}\|_2}}$$
Where norms $< 10^{-4}$ evaluate to $\cos\theta = 1.0$.

---

## 16. Quantum Resource Metrics

### Number of Qubits

$$n = \log_2(N_x N_y) = \log_2(16 \times 16) = \log_2(256) = 8 \text{ qubits}$$

### State Dimension

$$D = 2^n = 2^8 = 256 \text{ orthogonal basis states}$$

### Statevector Memory

For a double-precision complex statevector ($16$ bytes per complex amplitude):
$$M_{\text{statevector}} = 16 \times 2^n = 16 \times 256 = 4096 \text{ bytes (4.0 KiB)}$$
Excludes classical simulator metadata and runtime cache overhead.

### Circuit Metrics

For the 8-qubit separable 2-D QFT native circuit:
- Qubits: 8
- Circuit depth: 9 layers
- Total quantum gates: 25 (8 Hadamards, 12 Controlled-Phase rotations, 5 SWAP gates)
- State preparation: Exact amplitude initialization via `QuantumCircuit.initialize()`
- Simulation execution time: $\approx 1.9 \text{ ms}$ per 2D QFT step on CPU.

---

## 17. Experimental Comparison

All benchmarks compare the 5 standardized configurations:
- **Baseline A**: Classical APF only.
- **Baseline B**: Classical APF + Electrostatic-inspired dynamic field.
- **Baseline C**: Classical APF + Dynamic field + Predictive collision detection.
- **Baseline D**: Classical APF + Dynamic field + Predictive collision detection + CBF shield.
- **Baseline E**: AQSF-NAV v2 with 8-Qubit QFT Adaptive Spectral Filtering and CBF shield.

Full empirical comparison metrics across all 7 experimental scenarios are recorded in [Section 12](#12-experimental-benchmark-results).

---

## 18. Scientific Limitations and Assumptions

1. **Hybrid Framework**: The architecture is hybrid quantum-classical. Spatial fields are encoded and processed quantum-spectrally, while physical motion integration and CBF safety optimization are executed classically.
2. **Simulation vs Hardware**: All QFT and IQFT operations run on classical statevector simulators (`qiskit`, `qiskit-aer`). Physical quantum hardware execution was not used.
3. **No Quantum Advantage**: No quantum supremacy or computational advantage is claimed over classical FFTs for 256-element grids.
4. **Artificial Mathematical Analogy**: The electrostatic-inspired potential is an artificial mathematical repulsive analogy, not a physical electric field.
5. **Constant Relative Velocity Assumption**: Closest-approach calculations ($t^*, d^*$) assume relative velocity is piecewise constant over prediction horizon $T = 2.0 \text{ s}$.
6. **Velocity Estimation Dependence**: Predictive accuracy depends on accurate velocity state estimates for moving obstacles.
7. **Control Authority and Feasibility**: Continuous-time CBF guarantees assume adequate control authority ($\|v\| \le v_{\max}$) and empty intersection of unsafe sets.
8. **Discrete Discretization Effects**: Continuous-time CBF conditions can experience minor penetration bounds under discrete-time Euler integration ($\Delta t = 0.20 \text{ s}$), absorbed by margin $m = 0.20$.
9. **Gradient Modification via Filtering**: Radial low-pass spectral filtering modifies local potential gradients and flattens obstacle edges.
10. **Global Power vs Local Safety**: High global spectral power retention ($> 97\%$) does not guarantee local clearance margin maintenance; independent CBF enforcement is required.
11. **Potential Field Basins**: Highly cluttered static layouts (such as narrow warehouse aisles) can create shallow gradient basins when filtered, slowing convergence.
12. **Reproducibility Commitment**: All results, tables, and telemetry are deterministically recorded without fabricated data.

---

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
