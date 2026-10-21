# ASQF-NAV

> **Autonomous Navigation & Simulation Framework with Potential Fields and Quantum Optimization**

**ASQF-NAV** is a multi-agent simulation and path planning research platform combining classical **Artificial Potential Fields (APF)** with **Quantum Computing** (via [Qiskit](https://github.com/Qiskit/qiskit)) for collision-free multi-agent navigation in complex grid environments.

---

## 📁 Project Structure

```text
ASQF/
├── environment/
│   └── field.py          # Artificial Potential Field (goal attraction, obstacle & agent repulsion)
├── navigation/           # Path planning, agent policies, and trajectory optimizers
├── quantum/              # Quantum circuits, variational layers, and Qiskit-Aer simulation
├── experiment/           # Benchmark scenarios, evaluation suites, and parameter sweeps
├── results/              # Output metrics, performance logs, and simulation artifacts
├── visualization/        # Matplotlib plotting routines, circuit & trajectory renderers
├── main.py               # Main demo script: calculates and renders interaction potential fields
├── requirements.txt      # Python dependencies
├── .gitignore            # Git exclusion rules
└── README.md             # Project documentation
```

### Module Overview

- **`environment/field.py`**: Computes the complete 2D interaction potential field for agents on a discrete grid. Handles:
  - Attractive potential pulling agents toward target goals.
  - Repulsive potential pushing agents away from static obstacles within a defined influence radius.
  - Inter-agent dynamic repulsive potential to prevent collisions between moving agents.
  - Distance clamping ($d^* = \max(d, 0.5)$) to eliminate numerical singularities.
- **`navigation/`**: Houses classical trajectory planning and quantum-assisted navigation policies.
- **`quantum/`**: Quantum circuit definitions, state encoding, variational quantum layers (VQC), and Qiskit-Aer simulation backends.
- **`experiment/`**: Automated test harnesses to evaluate path efficiency, collision rates, and computational scaling.
- **`results/`**: Directory for storing experiment logs, CSV metric data, and generated figures.
- **`visualization/`**: Plotting utilities for field heatmaps, agent trajectories, and quantum circuit layouts.

---

## 📐 Potential Field Formulation

The total potential $F_{\text{total}}(x, y)$ perceived by an agent is modeled as:

$$F_{\text{total}}(x, y) = F_{\text{goal}}(x, y) + F_{\text{obstacle}}(x, y) + F_{\text{agent}}(x, y)$$

1. **Goal Attraction ($F_{\text{goal}}$):**
   $$F_{\text{goal}}(x, y) = \frac{1}{2} k_{\text{goal}} \left((x - g_x)^2 + (y - g_y)^2\right)$$
   *(Parameters: $k_{\text{goal}} = 1.0$)*

2. **Obstacle Repulsion ($F_{\text{obstacle}}$):**
   $$F_{\text{obstacle}}(x, y) = \sum_{i} \begin{cases} 
   \frac{1}{2} k_{\text{obs}} \left( \frac{1}{d_i^*} - \frac{1}{\rho_{\text{obs}}} \right)^2 & \text{if } d_i < \rho_{\text{obs}} \\
   0 & \text{otherwise}
   \end{cases}$$
   Where $d_i^* = \max(d_i, 0.5)$ prevents division-by-zero singularities.  
   *(Parameters: $k_{\text{obs}} = 80.0$, $\rho_{\text{obs}} = 3.0$)*

3. **Inter-Agent Repulsion ($F_{\text{agent}}$):**
   $$F_{\text{agent}}(x, y) = \sum_{j \neq \text{self}} \begin{cases}
   \frac{1}{2} k_{\text{agent}} \left( \frac{1}{d_j^*} - \frac{1}{\rho_{\text{agent}}} \right)^2 & \text{if } d_j < \rho_{\text{agent}} \\
   0 & \text{otherwise}
   \end{cases}$$
   *(Parameters: $k_{\text{agent}} = 30.0$, $\rho_{\text{agent}} = 2.5$)*

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
| `numpy` | `^2.0` | Grid generation, vectorized Euclidean distance, matrix operations |
| `matplotlib` | `^3.10` | 2D field visualization, scatter plots, trajectory rendering |
| `qiskit` | `^2.6` | Quantum circuit construction, transpilation, and quantum gates |
| `qiskit-aer` | `^0.18` | High-performance quantum statevector and QASM simulation |
| `scipy` | `^1.17` | Scientific computing and optimization routines |

---

## 💻 Running the Simulation

Execute `main.py` to calculate the interaction potential field for Agent 0 and display the 2D visualization:

```bash
python3.11 main.py
```

### Expected Console Output

```text
Grid shape: (16, 16)
Minimum field value: 2.15
Maximum field value: 169.0
Mean field value: 53.425768050120816
```

### Visualization

The script opens an interactive Matplotlib window showing:
- **Potential Surface:** A `viridis` colormap displaying the continuous scalar field $F(x,y)$.
- **Obstacles:** Red squares located at `(8, 8)` and `(5, 11)`.
- **Agents:** Labeled circles (`A0` at `(2, 2)`, `A1` at `(13, 13)`).
- **Goals:** Labeled yellow stars (`G0` at `(13, 13)`, `G1` at `(2, 2)`).
- **Colorbar:** Scaled potential values $F(x,y)$ guiding gradient descent trajectories.

---

## 🧪 Next Steps & Roadmap

- [x] Classical 2D Artificial Potential Field (APF) with obstacle and agent repulsion.
- [x] Multi-agent scenario visualization on a $16 \times 16$ grid.
- [ ] Local minima resolution via quantum variational optimization in `quantum/`.
- [ ] Agent path integration and navigation rollout loop in `navigation/`.
- [ ] Multi-scenario benchmarking and trajectory logging in `experiment/` & `results/`.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
