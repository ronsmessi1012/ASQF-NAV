# ASQF-NAV

> **Autonomous Navigation & Simulation Framework with Potential Fields, Continuous Gradient Control, and Quantum Optimization**

**ASQF-NAV** is a multi-agent simulation and path planning research platform combining classical **Artificial Potential Fields (APF)**, continuous **bilinear gradient controllers**, and **Quantum Computing** (via [Qiskit](https://github.com/Qiskit/qiskit)) for collision-free multi-agent navigation in complex grid environments.

---

## 📁 Project Structure

```text
ASQF/
├── environment/
│   └── field.py              # Artificial Potential Field (goal attraction, obstacle & agent repulsion)
├── navigation/
│   └── controller.py         # Continuous motion controller, bilinear interpolation & local-minimum escape
├── quantum/                  # Quantum circuits, variational layers, and Qiskit-Aer simulation
├── experiment/               # Benchmark scenarios, evaluation suites, and parameter sweeps
├── results/                  # Output metrics, performance logs, and simulation artifacts
├── visualization/            # Matplotlib plotting routines, circuit & trajectory renderers
├── main.py                   # Closed-loop simulation demo & trajectory visualizer
├── requirements.txt          # Python dependencies
├── .gitignore                # Git exclusion rules
└── README.md                 # Project documentation
```

### Module Overview

- **`environment/field.py`**: Computes the 2D interaction potential field for agents on a discrete $16 \times 16$ grid.
  - Attractive potential pulling agents toward target goals.
  - Repulsive potential pushing agents away from static obstacles within a defined influence radius.
  - Inter-agent dynamic repulsive potential to prevent collisions between moving agents.
  - Distance clamping ($d^* = \max(d, 0.5)$) to eliminate numerical division-by-zero singularities.
- **`navigation/controller.py`**: Closed-loop continuous motion planning and robot velocity generation.
  - **Bilinear Field Interpolation**: Evaluates discrete grid potential and gradients smoothly at continuous floating-point robot positions.
  - **Gradient-Based Steering**: Computes velocities using normalized negative potential gradients combined with goal heading.
  - **Local-Minimum Escape**: Detects field stagnation ($\|\nabla F\| < 0.15$) and injects an orthogonal tangential velocity along the equipotential contour toward the goal.
  - **Continuous Step & Bounds**: Advances agent positions with configurable step size ($\Delta t = 0.20$) while clipping within grid bounds.
- **`quantum/`**: Quantum circuit definitions, state encoding, variational quantum layers (VQC), and Qiskit-Aer simulation backends.
- **`experiment/`**: Automated test harnesses to evaluate path efficiency, collision rates, and computational scaling.
- **`results/`**: Output directory for storing trajectory benchmarks, CSV metric data, and generated figures.
- **`visualization/`**: Plotting utilities for field heatmaps, agent trajectories, and quantum circuit layouts.

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

3. **Inter-Agent Repulsion ($F_{\text{agent}}$):**
   $$F_{\text{agent}}(x, y) = \sum_{j \neq \text{self}} \begin{cases}
   \frac{1}{2} k_{\text{agent}} \left( \frac{1}{d_j^*} - \frac{1}{\rho_{\text{agent}}} \right)^2 & \text{if } d_j < \rho_{\text{agent}} \\
   0 & \text{otherwise}
   \end{cases}$$
   *(Parameters: $k_{\text{agent}} = 30.0$, $\rho_{\text{agent}} = 2.0$)*

---

## 🧭 Navigation & Control Architecture

The robot controller operates in continuous coordinate space $(x, y) \in [0, 15] \times [0, 15]$:

1. **Continuous Gradient Sampling:**
   Using central differences on the potential field grid, the spatial gradient $(\nabla_x F, \nabla_y F)$ is computed and bilinearly interpolated at continuous robot coordinates:
   $$\nabla F(p) = \text{BilinearInterpolate}\left(\nabla F_{\text{grid}}, p_x, p_y\right)$$

2. **Velocity Blending:**
   Under standard APF conditions:
   $$v = 0.65 \cdot \left(-\frac{\nabla F}{\|\nabla F\|}\right) + 0.35 \cdot \hat{d}_{\text{goal}}$$

3. **Equipotential Tangent Escape:**
   When encountering a local minimum ($\|\nabla F\| < 0.15$ while $\|p - g\| > 0.45$):
   $$\tau = \left(-\frac{\partial F}{\partial y}, \frac{\partial F}{\partial x}\right), \quad v = 0.45 \cdot \hat{d}_{\text{goal}} + 0.55 \cdot \hat{\tau}$$
   (ensuring $\hat{\tau} \cdot \hat{d}_{\text{goal}} \ge 0$).

4. **Integration & Termination:**
   Position updates follow $p_{t+1} = p_t + \Delta t \cdot v$ with $\Delta t = 0.20$ and $v_{\text{max}} = 1.0$. The navigation terminates when $\|p - g\| \le 0.45$.

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
| `qiskit` | `^2.6` | Quantum circuit construction, transpilation, and quantum gates |
| `qiskit-aer` | `^0.18` | High-performance quantum simulation backends |
| `scipy` | `^1.17` | Scientific computing and optimization routines |

---

## 💻 Running the Simulation

Execute `main.py` to run the closed-loop navigation simulation:

```bash
python3.11 main.py
```

### Console Output

```text
Goal reached in 78 steps.

Trajectory:
Step   0: (2.000, 2.000)
Step  10: (3.414, 3.414)
Step  20: (4.828, 4.828)
Step  30: (6.243, 6.243)
Step  40: (7.787, 7.439)
Step  50: (9.544, 8.078)
Step  60: (10.872, 9.566)
Step  70: (11.928, 11.265)
Starting position: (2.0, 2.0)
Final position: [12.76895673 12.62606621]
Goal: [13. 13.]
Final distance: 0.4395537184949548
```

### Visualization

The script opens an interactive Matplotlib figure showing:
- **Potential Field Heatmap:** Scaled potential values $F(x,y)$ rendered via `viridis`.
- **Static Obstacles:** Red squares located at `(8.0, 9.0)` and `(5.0, 11.0)`.
- **Robot Trajectory:** Smooth white curve tracing the agent's path from start to goal.
- **Start Position:** Cyan circle marker at `(2.0, 2.0)`.
- **Goal Target:** Yellow star marker at `(13.0, 13.0)`.

---

## 🧪 Roadmap

- [x] Classical 2D Artificial Potential Field (APF) with obstacle and agent repulsion (`environment/field.py`).
- [x] Continuous bilinear interpolation & numerical gradient sampling (`navigation/controller.py`).
- [x] Tangential equipotential steering for local-minima avoidance.
- [x] Closed-loop navigation rollout and dynamic trajectory visualization (`main.py`).
- [ ] Quantum variational circuit (VQC) / QRL policy integration in `quantum/`.
- [ ] Multi-scenario benchmark suites and automated metric logging in `experiment/` & `results/`.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
