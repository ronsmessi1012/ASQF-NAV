# ASQF-NAV

> **Autonomous Multi-Agent Navigation Framework with Potential Fields, Dynamic Collision Avoidance, and Quantum Amplitude Encoding**

**ASQF-NAV** is a multi-agent simulation and path planning research platform combining classical **Artificial Potential Fields (APF)**, continuous **bilinear gradient controllers with dual-layer collision avoidance**, and **Quantum Computing** (via [Qiskit](https://github.com/Qiskit/qiskit)) for collision-free multi-agent navigation in complex grid environments.

---

## 📁 Project Structure

```text
ASQF/
├── environment/
│   └── field.py              # Artificial Potential Field (goal attraction, obstacle & dynamic agent repulsion)
├── navigation/
│   └── controller.py         # Multi-agent continuous motion controller with early avoidance & hard safety projection
├── quantum/
│   ├── amplitude_encoding.py # 8-qubit quantum amplitude state preparation for 16x16 potential fields
│   └── test_amplitude.py     # Verification script for quantum amplitude encoding
├── experiment/               # Benchmark scenarios, evaluation suites, and parameter sweeps
├── results/                  # Output metrics, performance logs, and simulation artifacts
├── visualization/            # Matplotlib plotting routines, circuit & trajectory renderers
├── main.py                   # Multi-agent dynamic closed-loop simulation & trajectory visualizer
├── test_amplitude.py         # Quantum statevector encoding test runner
├── requirements.txt          # Python dependencies
├── .gitignore                # Git exclusion rules
└── README.md                 # Project documentation
```

### Module Overview

- **`environment/field.py`**: Computes the 2D interaction potential field for agents on a discrete $16 \times 16$ grid.
  - Attractive potential pulling agents toward target goals.
  - Repulsive potential pushing agents away from static obstacles within an influence radius.
  - Inter-agent dynamic repulsive potential treating other moving robots as dynamic obstacles.
  - Distance clamping ($d^* = \max(d, 0.5)$) to eliminate numerical division-by-zero singularities.
- **`navigation/controller.py`**: Closed-loop multi-agent continuous motion planning and velocity generation.
  - **Bilinear Field Interpolation**: Evaluates discrete grid potential and gradients smoothly at continuous floating-point robot positions.
  - **Gradient-Based Steering**: Blends normalized negative potential gradients with direct goal headings.
  - **Early Avoidance Steering** ($d < 1.5$): Smoothly increases outward separation velocity as robots approach each other.
  - **Hard Safety Projection** ($d \le 1.05$): Orthogonally projects out any velocity component pointing toward another robot and adds a radial separation force.
- **`quantum/amplitude_encoding.py`**: Encodes the entire $16 \times 16$ ($N = 256$) scalar field into an $n = 8$ qubit quantum statevector using Qiskit's `initialize()` operator, with exact fidelity verification.
- **`experiment/`**: Automated test harnesses to evaluate path efficiency, collision rates, and computational scaling.
- **`results/`**: Output directory for storing trajectory benchmarks, CSV metric data, and generated figures.
- **`visualization/`**: Plotting utilities for field heatmaps, multi-agent trajectories, and quantum circuit layouts.

---

## 📐 Potential Field Formulation

The total potential $F_{\text{total}}(x, y)$ perceived by an agent is modeled as:

$$F_{\text{total}}(x, y) = F_{\text{goal}}(x, y) + F_{\text{obstacle}}(x, y) + F_{\text{agent}}(x, y)$$

1. **Goal Attraction ($F_{\text{goal}}$):**
   $$F_{\text{goal}}(x, y) = \frac{1}{2} k_{\text{goal}} \left((x - g_x)^2 + (y - g_y)^2\right)$$
   *(Parameters: $k_{\text{goal}} = 2.0$)*

2. **Obstacle Repulsion ($F_{\text{obstacle}}$):**
   $$F_{\text{obstacle}}(x, y) = \sum_{i} \begin{cases} 
   \frac{1}{2} k_{\text{obs}} \left( \frac{1}{d_i^*} - \frac{1}{\rho_{\text{obs}}} \right)^2 & \text{if } d_i < \rho_{\text{obs}} \\
   0 & \text{otherwise}
   \end{cases}$$
   Where $d_i^* = \max(d_i, 0.5)$ prevents division-by-zero singularities.  
   *(Parameters: $k_{\text{obs}} = 80.0$, $\rho_{\text{obs}} = 2.0$)*

3. **Dynamic Agent Repulsion ($F_{\text{agent}}$):**
   $$F_{\text{agent}}(x, y) = \sum_{j \neq \text{self}} \begin{cases}
   \frac{1}{2} k_{\text{agent}} \left( \frac{1}{d_j^*} - \frac{1}{\rho_{\text{agent}}} \right)^2 & \text{if } d_j < \rho_{\text{agent}} \\
   0 & \text{otherwise}
   \end{cases}$$
   *(Parameters: $k_{\text{agent}} = 30.0$, $\rho_{\text{agent}} = 2.0$)*

---

## 🧭 Multi-Agent Motion Controller & Safety Layers

The robot controller operates in continuous coordinate space $(x, y) \in [0, 15] \times [0, 15]$:

1. **Continuous Gradient Sampling:**
   Spatial gradients $(\nabla_x F, \nabla_y F)$ are computed from the discrete field via central differences and bilinearly interpolated at continuous robot coordinates:
   $$\nabla F(p) = \text{BilinearInterpolate}\left(\nabla F_{\text{grid}}, p_x, p_y\right)$$

2. **Base Velocity Blending:**
   $$v_{\text{base}} = 0.65 \cdot \left(-\frac{\nabla F}{\|\nabla F\|}\right) + 0.35 \cdot \hat{d}_{\text{goal}}$$

3. **Dual-Layer Inter-Agent Collision Avoidance:**
   For relative position vector $r = p_{\text{self}} - p_{\text{other}}$ and separation $d = \|r\|$:
   - **Layer 1: Early Avoidance Steering** ($d_{\text{safe}} < d < d_{\text{avoid}}$, where $d_{\text{avoid}} = 1.5, d_{\text{safe}} = 1.05$):
     $$w_{\text{avoid}} = \frac{d_{\text{avoid}} - d}{d_{\text{avoid}} - d_{\text{safe}}}, \quad v \leftarrow v + w_{\text{avoid}} \cdot \hat{r}$$
   - **Layer 2: Hard Safety Projection** ($d \le d_{\text{safe}}$):
     If $v \cdot \hat{r} < 0$ (moving toward the other robot):
     $$v \leftarrow v - (v \cdot \hat{r}) \hat{r} + (d_{\text{safe}} - d) \cdot \hat{r}$$

4. **Integration & Termination:**
   Position updates follow $p_{t+1} = p_t + \Delta t \cdot v$ with $\Delta t = 0.20$ and $v_{\text{max}} = 1.0$. The agent terminates when $\|p - g\| \le 0.45$.

---

## ⚛️ Quantum Amplitude Encoding

To interface the spatial environment with quantum algorithms and variational circuits:
- The continuous scalar potential field $F \in \mathbb{R}^{16 \times 16}$ is flattened into a vector of $N = 256$ elements.
- The vector is normalized under the $L_2$ norm: $\|c\|_2 = 1.0$.
- Because $256 = 2^8$, the entire potential landscape is mapped into the state amplitudes of an **8-qubit quantum register**:

$$|\psi\rangle = \sum_{i=0}^{255} c_i |i\rangle, \quad \sum_{i=0}^{255} |c_i|^2 = 1$$

- State preparation is implemented using Qiskit's `QuantumCircuit.initialize()` and verified against exact simulated `Statevector` outputs:
  - **Classical $\rightarrow$ Quantum Amplitude Reconstruction Error:** $0.0$
  - **Normalization Error:** $\approx 1.11 \times 10^{-16}$

---

## 🛠️ Prerequisites

- **Python:** 3.10+ (Tested on Python 3.11 on macOS Apple Silicon / Intel and Linux)
- **Git**

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ronsmessi1012/ASQF-NAV.git
   cd ASQF-NAV
   ```

2. **Set up a virtual environment:**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

| Package | Version | Purpose |
| :--- | :--- | :--- |
| `numpy` | `^2.0` | Numerical grids, bilinear interpolation, vector algebra |
| `matplotlib` | `^3.10` | 2D field visualization, trajectory traces, and scatter markers |
| `qiskit` | `^2.6` | Quantum circuit construction, state initialization, and quantum gates |
| `qiskit-aer` | `^0.18` | High-performance quantum statevector and circuit simulations |
| `scipy` | `^1.17` | Scientific computing and optimization routines |

---

## 💻 Running the Simulations

### 1. Multi-Agent Navigation Demo

Execute `main.py` to run the closed-loop multi-agent simulation with crossing trajectories and obstacle avoidance:

```bash
python3.11 main.py
```

#### Console Output

```text
Both robots reached their goals in 93 steps.

Final Results
Robot 0: Start=(2.0, 2.0), Final=(12.980, 12.702), Goal=[13. 13.], Distance=0.299
Robot 1: Start=(3.0, 13.0), Final=(12.950, 2.093), Goal=[13.  2.], Distance=0.105

Minimum inter-agent distance: 1.058
Minimum distance occurred at step: 59
```

#### Visualization

The script opens an interactive Matplotlib figure showing:
- **Potential Field Heatmap:** Continuous interaction field surface $F(x,y)$.
- **Static Obstacles:** Red squares at `(8.0, 9.0)` and `(5.0, 11.0)`.
- **Robot Trajectories:** Curves tracing both Robot 0 and Robot 1 traversing the map without colliding.
- **Start & Goal Markers:** Circle start markers and star goal targets for each agent.
- **Safety Margin:** Confirms inter-agent separation remains above the safety threshold ($1.058 > 1.0$).

---

### 2. Quantum Amplitude Encoding Verification

Test the 8-qubit quantum state encoding of the interaction potential field:

```bash
python3.11 test_amplitude.py
```

#### Console Output

```text
========== AMPLITUDE ENCODING ==========
Classical field shape: (16, 16)
Number of amplitudes: 256
Number of qubits: 8
Quantum state dimension: 256
Classical vector norm: 1.0
Quantum state norm: 1.0
Amplitude reconstruction error: 0.0
Normalization error: 1.1102230246251565e-16

Quantum circuit:
     »
q_0: »
q_1: »
...
q_7: »
```

---

## 🧪 Roadmap

- [x] Classical 2D Artificial Potential Field (APF) with obstacle and dynamic agent repulsion (`environment/field.py`).
- [x] Continuous bilinear interpolation & numerical gradient sampling (`navigation/controller.py`).
- [x] Dual-layer multi-agent collision avoidance (early avoidance steering + hard safety projection).
- [x] Multi-agent closed-loop navigation rollout and dynamic trajectory visualization (`main.py`).
- [x] 8-qubit quantum amplitude encoding of $16 \times 16$ spatial fields (`quantum/amplitude_encoding.py`).
- [ ] Quantum Variational Circuit (VQC) / Parameterized Quantum Policy optimization in `quantum/`.
- [ ] Multi-scenario benchmark suites and automated metric logging in `experiment/` & `results/`.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
