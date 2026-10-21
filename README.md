# ASQF

> **A Quantum-Enhanced Simulation & Navigation Framework**

ASQF is a Python-based research and development platform that combines quantum computing (via [Qiskit](https://github.com/Qiskit/qiskit)) with simulated environments, autonomous agent navigation, and visualization tools.

---

## 📁 Project Structure

```text
ASQF/
├── environment/      # Simulation environment, world dynamics, and state representation
├── navigation/       # Path planning, agent navigation, and decision-making logic
├── quantum/          # Quantum circuits, gates, variational layers, and simulation backends
├── experiment/       # Experiment pipelines, benchmarking, and evaluation scripts
├── results/          # Output metrics, performance logs, and simulation artifacts
├── visualization/    # Trajectory plots, circuit diagrams, and data visualization
├── main.py           # Application entry point and demo execution
├── requirements.txt  # Python package dependencies
├── .gitignore        # Git ignore rules for Python, virtual environments, and OS files
└── README.md         # Project documentation
```

### Module Overview

- **`quantum/`**: Core quantum routines built with Qiskit and Qiskit-Aer. Includes quantum circuit designs, entanglement protocols, and quantum measurement routines.
- **`environment/`**: Defines state spaces, dynamic models, and obstacle configurations for testing agent behavior.
- **`navigation/`**: Houses classical and quantum-assisted navigation policies, path planning algorithms, and agent control routines.
- **`experiment/`**: Experiment harnesses to run comparative benchmarks, sensitivity tests, and parameter sweeps.
- **`visualization/`**: Plotting utilities for trajectories, circuit outputs, state vectors, and evaluation graphs using Matplotlib.
- **`results/`**: Destination for generated logs, CSV outputs, and figures.

---

## 🛠️ Prerequisites

- **Operating System:** macOS (Apple Silicon / Intel), Linux, or Windows
- **Python:** 3.10+ (Tested with Python 3.11)

---

## 🚀 Installation & Setup

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd ASQF
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Create a virtual environment
   python3.11 -m venv venv

   # Activate the virtual environment
   # On macOS / Linux:
   source venv/bin/activate

   # On Windows (Command Prompt):
   # venv\Scripts\activate.bat
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

| Package | Purpose |
| :--- | :--- |
| `qiskit` | Quantum circuit creation, manipulation, and transpilation |
| `qiskit-aer` | High-performance quantum circuit simulation backend |
| `numpy` | Numerical operations and matrix manipulation |
| `scipy` | Scientific computing and optimization routines |
| `matplotlib` | 2D plotting, circuit visualization, and trajectory figures |

---

## 💻 Quickstart

Run the entry point script to verify your Qiskit and simulation setup:

```bash
python main.py
```

Expected output:
```text
     ┌───┐     
q_0: ┤ H ├──■──
     └───┘┌─┴─┐
q_1: ─────┤ X ├
          └───┘
```

---

## 🧪 Development Workflow

1. **Adding Quantum Circuits:** Define parameterized or variational quantum circuits in `quantum/`.
2. **Configuring Environments:** Implement grid or continuous state spaces in `environment/`.
3. **Running Experiments:** Execute reproducible experiments from `experiment/` and save artifacts to `results/`.
4. **Visualizing Results:** Use helper routines in `visualization/` to plot agent trajectories and quantum circuit diagrams.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
