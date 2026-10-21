# ASQF-NAV

> **Adaptive Spectral Quantum Filtering for Multi-Agent Navigation with Artificial Potential Fields, Dynamic Collision Avoidance, and Quantum Spectral Transforms**

**ASQF-NAV** is a multi-agent simulation and path planning research platform combining classical **Artificial Potential Fields (APF)**, continuous **bilinear gradient controllers with dual-layer collision avoidance**, and **Quantum Computing** (via [Qiskit](https://github.com/Qiskit/qiskit)) featuring **8-Qubit Amplitude Encoding**, the **Quantum Fourier Transform (QFT)**, **Quantum Spectral Complexity Analysis**, the **Adaptive Spectral Quantum Filter (ASQF)**, and **Quantum-Guided Continuous Motion Control** for collision-free navigation in complex environments.

---

## 📁 Project Structure

```text
ASQF/
├── environment/
│   └── field.py                  # Artificial Potential Field (goal attraction, obstacle & dynamic agent repulsion)
├── navigation/
│   ├── controller.py             # Classical multi-agent continuous motion controller with early avoidance
│   └── quantum_controller.py     # Quantum-driven motion controller operating on reconstructed quantum fields
├── quantum/
│   ├── adaptive_filter.py        # Adaptive Spectral Quantum Filter (ASQF) low-pass bandwidth control
│   ├── amplitude_encoding.py     # 8-qubit quantum amplitude state preparation for 16x16 potential fields
│   ├── qft.py                    # Quantum Fourier Transform (QFT) & Inverse QFT (IQFT) circuit implementations
│   ├── reconstruction.py         # Inverse QFT state inversion, 2D field reconstruction, and relative L2 error analysis
│   ├── spectral_analysis.py      # Spectral power, frequency metrics, and spatial complexity analysis
│   └── test_amplitude.py         # Verification script for quantum amplitude encoding
├── experiment/                   # Benchmark scenarios, evaluation suites, and parameter sweeps
├── results/                      # Output metrics, performance logs, and simulation artifacts
├── visualization/                # Matplotlib plotting routines, circuit & trajectory renderers
├── main.py                       # Multi-agent dynamic closed-loop simulation & trajectory visualizer
├── test_amplitude.py             # Quantum statevector encoding test runner
├── test_qft.py                   # Step 4: QFT, spectral complexity, adaptive filtering & reconstruction test runner
├── test_quantum_navigation.py    # Step 8: Quantum navigation pipeline & velocity generation test runner
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git exclusion rules
└── README.md                     # Project documentation
```

### Module Overview

- **`environment/field.py`**: Computes the 2D interaction potential field for agents on a discrete $16 \times 16$ grid.
  - Attractive potential pulling agents toward target goals.
  - Repulsive potential pushing agents away from static obstacles within an influence radius.
  - Inter-agent dynamic repulsive potential treating other moving robots as dynamic obstacles.
  - Distance clamping ($d^* = \max(d, 0.5)$) to eliminate numerical division-by-zero singularities.
- **`navigation/controller.py`**: Classical continuous motion planning and velocity generation.
  - **Bilinear Field Interpolation**: Evaluates discrete grid potential and gradients smoothly at continuous floating-point coordinates.
  - **Early Avoidance Steering** ($d < 1.5$): Smoothly increases outward separation velocity as robots approach each other.
  - **Hard Safety Projection** ($d \le 1.05$): Orthogonally projects out any velocity component pointing toward another robot.
- **`navigation/quantum_controller.py`**: Bridges quantum processing with continuous robot steering.
  - Executes the full quantum pipeline: Amplitude Encoding $\rightarrow$ QFT $\rightarrow$ Spectral Complexity $\rightarrow$ ASQF Filtering $\rightarrow$ IQFT Reconstruction.
  - Bilinearly interpolates continuous spatial gradients directly from the reconstructed quantum field.
  - Computes unit-normalized velocity commands steering agents toward goals through filtered potential valleys.
- **`quantum/adaptive_filter.py`**: Implements the **Adaptive Spectral Quantum Filter (ASQF)**. Inversely scales the filter bandwidth based on environmental spectral complexity, selectively suppresses high-frequency potential turbulence, and renormalizes the quantum state.
- **`quantum/amplitude_encoding.py`**: Encodes the entire $16 \times 16$ ($N = 256$) scalar field into an $n = 8$ qubit quantum statevector using Qiskit's `initialize()` operator, with exact fidelity verification.
- **`quantum/qft.py`**: Implements the Quantum Fourier Transform (QFT) and Inverse QFT (IQFT) on $n$ qubits using Hadamard gates, controlled phase rotations ($CP$), and SWAP inversion cascades.
- **`quantum/reconstruction.py`**: Inverts filtered quantum statevectors back into spatial coordinates using Inverse QFT circuits, computes the real-valued $16 \times 16$ field, verifies residual imaginary bounds, and quantifies relative $L_2$ error.
- **`quantum/spectral_analysis.py`**: Analyzes the QFT frequency spectrum to compute spectral power distributions and **Spectral Complexity** ($\mathcal{C}_{\text{spectral}}$), quantifying spatial clutter directly in the quantum frequency domain.
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

## 🧭 Navigation & Control Architecture

### 1. Classical Motion Planning & Dual-Layer Safety
Operates in continuous coordinate space $(x, y) \in [0, 15] \times [0, 15]$:
- **Continuous Gradient Sampling:** Spatial gradients $(\nabla_x F, \nabla_y F)$ are computed from the discrete field via central differences and bilinearly interpolated at continuous robot coordinates:
  $$\nabla F(p) = \text{BilinearInterpolate}\left(\nabla F_{\text{grid}}, p_x, p_y\right)$$
- **Base Velocity Blending:**
  $$v_{\text{base}} = 0.65 \cdot \left(-\frac{\nabla F}{\|\nabla F\|}\right) + 0.35 \cdot \hat{d}_{\text{goal}}$$
- **Dual-Layer Inter-Agent Collision Avoidance:**
  - **Layer 1: Early Avoidance Steering** ($d_{\text{safe}} < d < d_{\text{avoid}}$, where $d_{\text{avoid}} = 1.5, d_{\text{safe}} = 1.05$):
    $$w_{\text{avoid}} = \frac{d_{\text{avoid}} - d}{d_{\text{avoid}} - d_{\text{safe}}}, \quad v \leftarrow v + w_{\text{avoid}} \cdot \hat{r}$$
  - **Layer 2: Hard Safety Projection** ($d \le d_{\text{safe}}$):
    If $v \cdot \hat{r} < 0$: $v \leftarrow v - (v \cdot \hat{r}) \hat{r} + (d_{\text{safe}} - d) \cdot \hat{r}$.

### 2. Quantum-Guided Motion Control
In [navigation/quantum_controller.py](file:///Users/ronnie/Documents/ASQF/navigation/quantum_controller.py), continuous velocity planning is executed directly on the **quantum-filtered reconstructed field** $F_{\text{quantum}}$:
- High-frequency potential ripples that cause agent entrapment in local minima are filtered by the ASQF low-pass spectral mask.
- The quantum velocity $v_{\text{quantum}}$ combines the negative gradient of the filtered field with direct goal guidance:
  $$v_{\text{quantum}} = v_{\text{max}} \cdot \frac{0.65 \cdot \hat{v}_{-\nabla F_{\text{quantum}}} + 0.35 \cdot \hat{d}_{\text{goal}}}{\|0.65 \cdot \hat{v}_{-\nabla F_{\text{quantum}}} + 0.35 \cdot \hat{d}_{\text{goal}}\|}$$
- Produces a smooth, globally-consistent heading toward the goal while preserving obstacle repulsion.

---

## ⚛️ Quantum State Preparation, QFT & Adaptive Filtering Pipeline

```text
Classical 2D Field (16x16)
       │
       ▼
8-Qubit Amplitude Encoding (|ψ⟩, 256 amplitudes)
       │
       ▼
Quantum Fourier Transform (QFT |ψ⟩ = |ψ̃⟩)
       │
       ▼
Spectral Complexity Analysis (C_spectral = 1.13%)
       │
       ▼
Adaptive Spectral Quantum Filter (ASQF Bandwidth = 74.4%)
       │
       ▼
Inverse QFT Circuit (IQFT |ψ̃_filtered⟩)
       │
       ▼
Filtered Classical 2D Potential Field (L2 Relative Error = 6.45%)
       │
       ▼
Quantum Velocity Planning (v = [0.512, 0.859])
```

### 1. 8-Qubit Amplitude Encoding
To map the continuous $16 \times 16$ spatial field ($N = 256$) into quantum state space:
- The potential field $F \in \mathbb{R}^{16 \times 16}$ is flattened and normalized under the $L_2$ norm: $\|c\|_2 = 1.0$.
- Because $256 = 2^8$, the normalized field is mapped onto an **8-qubit register**:

$$|\psi\rangle = \sum_{i=0}^{255} c_i |i\rangle, \quad \sum_{i=0}^{255} |c_i|^2 = 1$$

- State preparation is realized with Qiskit's `QuantumCircuit.initialize()`:
  - **Reconstruction Error:** $0.0$
  - **Normalization Error:** $\approx 1.11 \times 10^{-16}$

### 2. Quantum Fourier Transform (QFT)
The discrete Quantum Fourier Transform maps the spatial representation $|\psi\rangle$ into the frequency/momentum basis $|\tilde{\psi}\rangle$:

$$\text{QFT} |j\rangle = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} \omega^{j k} |k\rangle, \quad \omega = \exp\left(\frac{2\pi i}{N}\right), \quad N = 256$$

- **Circuit Architecture**:
  - Hadamard gate on each qubit $j$: creates superpositions.
  - Controlled Phase gates $CP(\theta_{k, j})$ between qubits $j$ and $k > j$: rotates phases by $\theta_{k, j} = \frac{\pi}{2^{k - j}}$.
  - SWAP cascades: reverses qubit order to preserve standard binary significance.
- **Inverse QFT (IQFT)**:
  - Reverses SWAP layers and applies conjugate phase rotations $CP(-\theta_{k, j})$ and inverse Hadamards.
- **Unitary Conservation & Numerical Precision**:
  - $\text{IQFT} \cdot \text{QFT} = \mathbb{I}$
  - State reconstruction error: $\| |\psi\rangle - \text{IQFT}(\text{QFT}|\psi\rangle) \|_2 = \mathbf{1.56 \times 10^{-15}}$ (exact machine precision).
  - Total spectral energy is strictly conserved ($\sum |\tilde{c}_k|^2 \approx 1.0$).

### 3. Spectral Complexity & Environment Clutter
Using [quantum/spectral_analysis.py](file:///Users/ronnie/Documents/ASQF/quantum/spectral_analysis.py), the frequency distribution reveals structural spatial information about the environment:
- **Spectral Power**: $P(k) = |\tilde{c}_k|^2$ for frequency mode $k$.
- **Spectral Complexity ($\mathcal{C}_{\text{spectral}}$)**: Defined as the fraction of energy in high-frequency spatial modes ($|f_k| > 0.25 N = 64$):

$$\mathcal{C}_{\text{spectral}} = \frac{\sum_{|f_k| > 64} P(k)}{\sum_{k=0}^{255} P(k)}$$

- **Navigational Significance**:
  - Smooth attractor basins concentrate $\approx 98.87\%$ of spectral power in low-frequency harmonics.
  - Localized obstacle boundaries, narrow passages, and inter-agent boundaries manifest in the high-frequency band ($\approx 1.13\%$).
  - Provides a global, quantum-evaluable complexity index to quantify navigation difficulty and clutter density.

### 4. Adaptive Spectral Quantum Filter (ASQF)
The core **ASQF** algorithm dynamically adapts a spectral low-pass filter to environmental complexity:
- **Adaptive Bandwidth ($B$):**
  $$B(\mathcal{C}) = B_{\text{min}} + (B_{\text{max}} - B_{\text{min}}) \cdot (1 - \mathcal{C})$$
  *(Default parameters: $B_{\text{min}} = 0.25$, $B_{\text{max}} = 0.75$)*
- **Cutoff Frequency ($f_{\text{cutoff}}$):**
  $$f_{\text{cutoff}} = \left(\frac{N}{2}\right) \cdot B(\mathcal{C})$$
- **Spectral Masking & Renormalization:**
  $$\tilde{c}_k^{\text{filtered}} = \begin{cases} \frac{\tilde{c}_k}{\mathcal{N}} & \text{if } |f_k| \le f_{\text{cutoff}} \\ 0 & \text{otherwise} \end{cases}$$
  where $\mathcal{N} = \sqrt{\sum_{|f_k| \le f_{\text{cutoff}}} |\tilde{c}_k|^2}$ guarantees that the filtered quantum state strictly respects the quantum normalization postulate ($\|\tilde{\psi}_{\text{filtered}}\|_2 = 1.0$).
- **Filtering Response:**
  - In open/smooth regions ($\mathcal{C} = 0.0113$), the filter retains $B = 0.744$ bandwidth ($f_{\text{cutoff}} = 95.28$, retaining $191/256 = 74.61\%$ of spectral components).
  - In high-clutter/dense regions ($\mathcal{C} \to 1.0$), the filter automatically narrows down to $B = 0.25$, filtering high-frequency field ripples that cause local agent traps.

### 5. Field Reconstruction & Relative $L_2$ Error
Using [quantum/reconstruction.py](file:///Users/ronnie/Documents/ASQF/quantum/reconstruction.py), the filtered state is transformed back to physical space:
- **State Inversion**: $|\psi_{\text{recon}}\rangle = \text{IQFT} |\tilde{\psi}_{\text{filtered}}\rangle$.
- **Physical Verification**: Residual imaginary components are strictly bounded ($\max |\text{Im}(\psi_{\text{recon}})| = 0.0121 \ll 1$), confirming a physical, real potential field.
- **Relative $L_2$ Error**:
  $$\epsilon_{\text{rel}} = \frac{\| c_{\text{original}} - c_{\text{filtered}} \|_2}{\| c_{\text{original}} \|_2} = \mathbf{6.45\%}$$
  This indicates that ASQF retains **$93.55\%$** of the global navigation landscape while eliminating high-frequency potential irregularities.

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
| `qiskit` | `^2.6` | Quantum circuit construction, QFT/IQFT gates, state initialization |
| `qiskit-aer` | `^0.18` | High-performance quantum statevector and circuit simulations |
| `scipy` | `^1.17` | Scientific computing and optimization routines |

---

## 💻 Running the Simulations & Validations

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

The script displays an interactive Matplotlib figure showing:
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
```

---

### 3. End-to-End Quantum Pipeline Validation (QFT, ASQF & Reconstruction)

Run the complete pipeline: Amplitude Encoding $\rightarrow$ QFT $\rightarrow$ Spectral Complexity $\rightarrow$ ASQF Filtering $\rightarrow$ IQFT Reconstruction:

```bash
python3.11 test_qft.py
```

#### Console Output

```text
========== SPECTRAL ANALYSIS ==========
Number of spectral components: 256
Total spectral power: 0.9999999999999982
High-frequency spectral power: 0.011308812510864054
Spectral complexity: 0.011308812510864075
Spectral complexity (%): 1.1308812510864075
Maximum imaginary component: 0.012136756891117855

========== RECONSTRUCTION ==========
Original field shape: (16, 16)
Reconstructed field shape: (16, 16)
Reconstructed state norm: 0.9999999999999996
Minimum reconstructed value: -0.0002902245456402888
Maximum reconstructed value: 0.1569685404838594
Mean reconstructed value: 0.05064990704727313
Relative reconstruction error: 0.06454817754849157
Relative reconstruction error (%): 6.454817754849157

========== ADAPTIVE FILTER ==========
Spectral complexity: 0.011308812510864075
Spectral complexity (%): 1.1308812510864075
Adaptive bandwidth: 0.744345593744568
Cutoff frequency: 95.2762359993047
Retained spectral components: 191
Total spectral components: 256
Retained fraction: 0.74609375
Filtered spectrum norm: 1.0000000000000004

========== STEP 4: QFT VALIDATION ==========
Number of qubits: 8
State dimension: 256
Original state norm: 0.9999999999999999
QFT state norm: 0.9999999999999989
Reconstructed state norm: 0.9999999999999983
QFT -> IQFT reconstruction error: 1.5632581197446034e-15
Total spectral power: 0.9999999999999982
Maximum spectral power: 0.6512498428429626
Minimum spectral power: 2.2118479466521852e-07
```

---

### 4. Step 8: Quantum Navigation Pipeline & Velocity Planning

Test continuous velocity planning computed on the quantum-filtered potential field:

```bash
python3.11 test_quantum_navigation.py
```

#### Console Output

```text
========== STEP 8: QUANTUM NAVIGATION ==========
Original field shape: (16, 16)
Original field norm: 2067.30422476496
Maximum imaginary component: 0.012136756891117855

---------- Quantum Processing ----------
Spectral complexity: 0.011308812510864075
Adaptive bandwidth: 0.744345593744568
Retained fraction: 0.74609375
Original normalized norm: 0.9999999999999999
Reconstructed state norm: 0.9999999999999996
Reconstructed field shape: (16, 16)
Reconstructed field minimum: -0.0002902245456402888
Reconstructed field maximum: 0.1569685404838594

---------- Navigation ----------
Robot position: [2. 2.]
Goal: [13. 13.]
Quantum velocity: [0.51153638 0.85926162]
Velocity magnitude: 1.0
```

---

## 🧪 Roadmap

- [x] Classical 2D Artificial Potential Field (APF) with obstacle and dynamic agent repulsion (`environment/field.py`).
- [x] Continuous bilinear interpolation & numerical gradient sampling (`navigation/controller.py`).
- [x] Dual-layer multi-agent collision avoidance (early avoidance steering + hard safety projection).
- [x] Multi-agent closed-loop navigation rollout and dynamic trajectory visualization (`main.py`).
- [x] 8-qubit quantum amplitude encoding of $16 \times 16$ spatial fields (`quantum/amplitude_encoding.py`).
- [x] Quantum Fourier Transform (QFT & IQFT) with unitary spectral verification (`quantum/qft.py`).
- [x] Quantum spectral power distribution and spatial complexity analysis (`quantum/spectral_analysis.py`).
- [x] Adaptive Spectral Quantum Filter (ASQF) with complexity-driven bandwidth scaling (`quantum/adaptive_filter.py`).
- [x] Inverse QFT quantum field reconstruction and relative $L_2$ error quantification (`quantum/reconstruction.py`).
- [x] Quantum-guided continuous motion controller and velocity planning (`navigation/quantum_controller.py`).
- [x] Step 8: Quantum navigation pipeline validation (`test_quantum_navigation.py`).
- [ ] Multi-agent quantum closed-loop rollout benchmarking against classical APF baseline in `experiment/`.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
