# ASQF-NAV

> **Adaptive Spectral Quantum Filtering for Multi-Agent Navigation with Artificial Potential Fields, Dynamic Collision Avoidance, and Calibrated 2-D Quantum Spectral Transforms**

**ASQF-NAV** is an advanced multi-agent simulation and path planning research platform combining classical **Artificial Potential Fields (APF)**, continuous **bilinear gradient controllers with dual-layer collision avoidance**, and **Quantum Computing** (via [Qiskit](https://github.com/Qiskit/qiskit)). 

The architecture features **8-Qubit Amplitude Encoding**, a **Separable 2-D Quantum Fourier Transform (2-D QFT)** on an 8-qubit register, **Calibrated Spectral Complexity Analysis (Second Moment)**, a **Cubic Adaptive Spectral Quantum Filter (ASQF)**, and a **Comprehensive Diagnostic & Ablation Benchmark Suite** for high-fidelity multi-agent navigation.

---

## 📁 Project Structure

```text
ASQF/
├── environment/
│   └── field.py                     # Artificial Potential Field (goal attraction, static & dynamic agent repulsion)
├── navigation/
│   ├── controller.py                # Classical continuous motion controller with dual-layer collision avoidance
│   └── quantum_controller.py        # Quantum-driven motion controller operating on reconstructed quantum fields
├── quantum/
│   ├── adaptive_filter.py           # 1-D Adaptive Spectral Quantum Filter baseline
│   ├── adaptive_filter_2d.py        # 2-D Radial Adaptive Spectral Quantum Filter with cubic response (γ = 3.0)
│   ├── amplitude_encoding.py        # 8-qubit quantum amplitude state preparation for 16x16 potential fields
│   ├── qft.py                       # 1-D Quantum Fourier Transform (QFT) & Inverse QFT (IQFT) circuit implementation
│   ├── qft_2d_qiskit.py             # Separable 2-D QFT & IQFT Qiskit circuits on 8 qubits (little-endian mapped)
│   ├── reconstruction.py            # Inverse QFT state inversion, 2D field reconstruction, and relative L2 error analysis
│   ├── spectral_analysis.py         # 1-D spectral power, frequency metrics, and spatial complexity analysis
│   └── spectral_analysis_2d.py      # Calibrated 2-D spectral analysis, radial frequency grid & second moment metric
├── visualization/
│   └── vector_widget.html           # Interactive HTML5 canvas tool for vector field inspection & heading analysis
├── main.py                          # Classical multi-agent dynamic simulation & trajectory visualizer
├── vector_demo.html                 # Standalone interactive vector steering visualizer
├── test_amplitude.py                # Quantum statevector encoding test runner
├── test_qft.py                      # Step 4: 1-D QFT, spectral complexity, adaptive filtering & reconstruction test runner
├── test_classical_vs_quantum.py     # Comparative benchmark harness (Classical APF vs. 1-D Quantum ASQF)
│
├── # --- 2-D Quantum Spectral Pipeline & Calibration Suite ---
├── test_2d_qft.py                   # 2-D mathematical QFT verification vs. NumPy 2D FFT
├── test_qiskit_2d_qft.py            # Qiskit 8-qubit separable 2-D QFT circuit precision validation (< 10^-15 error)
├── test_qiskit_mapping.py           # Verification of little-endian coordinate mapping on Qiskit registers
├── test_qiskit_bit_order.py         # Bit-reversal swap verification for 2-D QFT circuits
├── test_qiskit_1d_qft.py            # 1-D Qiskit QFT gate verification
├── test_qiskit_complex_iqft.py      # Complex-valued filtered state encoding and IQFT circuit inversion
├── test_spectral_metrics.py         # Monotonicity comparison: threshold ratios vs. spectral second moment
├── test_complexity_calibration.py   # Statistical calibration of spectral complexity across 50 random environments
├── test_response_curves.py          # Bandwidth response curve evaluation (γ = 1.0, 1.5, 2.0, 3.0)
├── test_calibrated_2d_filter.py     # Verification of calibrated 2-D radial spectral filter
├── test_spectral_module.py          # End-to-end unit test suite for the complete 2-D spectral module
│
├── # --- Navigation & Multi-Agent Diagnostic Suite (Step 9) ---
├── test_quantum_navigation_step.py  # Step 9W: Single-step velocity and gradient consistency test
├── test_single_agent_trajectory.py  # Step 9X: Closed-loop single-agent navigation rollout with 2-D quantum filtering
├── test_quantum_multi_agent.py      # Step 9Y: 3-way multi-agent filter ablation benchmark (Classical / Unfiltered / Filtered)
├── test_filter_local_sensitivity.py # Step 9Z: Critical-region gradient sensitivity and collision-margin diagnostic
│
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git exclusion rules
└── README.md                        # Project documentation
```

### Module Overview

- **[environment/field.py](file:///Users/ronnie/Documents/ASQF/environment/field.py)**: Generates the 2D interaction potential field for agents on a discrete $16 \times 16$ grid with singularity clamping ($d^* = \max(d, 0.5)$).
- **[navigation/controller.py](file:///Users/ronnie/Documents/ASQF/navigation/controller.py)**: Classical continuous motion planning with bilinear field interpolation and dual-layer inter-agent collision avoidance (early avoidance steering + hard safety projection).
- **[navigation/quantum_controller.py](file:///Users/ronnie/Documents/ASQF/navigation/quantum_controller.py)**: Continuous velocity planning directly evaluated from reconstructed quantum fields.
- **[quantum/qft_2d_qiskit.py](file:///Users/ronnie/Documents/ASQF/quantum/qft_2d_qiskit.py)**: Separable 2-D QFT and IQFT implemented as native Qiskit circuits on an 8-qubit register with exact little-endian coordinate alignment.
- **[quantum/spectral_analysis_2d.py](file:///Users/ronnie/Documents/ASQF/quantum/spectral_analysis_2d.py)**: 2-D radial spatial frequency grid generation, spectral power distribution, and the **Spectral Second Moment** complexity metric calibrated over 50 randomized environments.
- **[quantum/adaptive_filter_2d.py](file:///Users/ronnie/Documents/ASQF/quantum/adaptive_filter_2d.py)**: Radial low-pass adaptive filter with cubic response ($\gamma = 3.0$) providing smooth bandwidth adaptation from $B_{\max} = 0.75$ down to $B_{\min} = 0.25$.
- **[quantum/amplitude_encoding.py](file:///Users/ronnie/Documents/ASQF/quantum/amplitude_encoding.py)**: Maps normalized $16 \times 16$ scalar fields ($N = 256$) into normalized 8-qubit quantum states.
- **[quantum/reconstruction.py](file:///Users/ronnie/Documents/ASQF/quantum/reconstruction.py)**: Inverse QFT state transformation, imaginary component bound verification, and relative $L_2$ error calculation.

---

## 📐 Potential Field & Classical Control Formulation

The total potential $F_{\text{total}}(x, y)$ perceived by an agent is modeled as:

$$F_{\text{total}}(x, y) = F_{\text{goal}}(x, y) + F_{\text{obstacle}}(x, y) + F_{\text{agent}}(x, y)$$

1. **Goal Attraction ($F_{\text{goal}}$):**
   $$F_{\text{goal}}(x, y) = \frac{1}{2} k_{\text{goal}} \left((x - g_x)^2 + (y - g_y)^2\right), \quad k_{\text{goal}} = 2.0$$

2. **Obstacle Repulsion ($F_{\text{obstacle}}$):**
   $$F_{\text{obstacle}}(x, y) = \sum_{i} \begin{cases} 
   \frac{1}{2} k_{\text{obs}} \left( \frac{1}{d_i^*} - \frac{1}{\rho_{\text{obs}}} \right)^2 & \text{if } d_i < \rho_{\text{obs}} \\
   0 & \text{otherwise}
   \end{cases}$$
   Where $d_i^* = \max(d_i, 0.5)$ eliminates division-by-zero singularities ($k_{\text{obs}} = 80.0$, $\rho_{\text{obs}} = 2.0$).

3. **Dynamic Agent Repulsion ($F_{\text{agent}}$):**
   $$F_{\text{agent}}(x, y) = \sum_{j \neq \text{self}} \begin{cases}
   \frac{1}{2} k_{\text{agent}} \left( \frac{1}{d_j^*} - \frac{1}{\rho_{\text{agent}}} \right)^2 & \text{if } d_j < \rho_{\text{agent}} \\
   0 & \text{otherwise}
   \end{cases}$$
   Where $k_{\text{agent}} = 30.0$, $\rho_{\text{agent}} = 2.0$.

### Continuous Gradient Sampling & Dual-Layer Collision Avoidance

- **Bilinear Field Interpolation**: Spatial gradients $(\nabla_x F, \nabla_y F)$ are computed from discrete grid differences and bilinearly interpolated at continuous robot coordinates $p = (x, y)$:
  $$\nabla F(p) = \text{BilinearInterpolate}\left(\nabla F_{\text{grid}}, p_x, p_y\right)$$
- **Base Velocity Blending**:
  $$v_{\text{base}} = 0.65 \cdot \left(-\frac{\nabla F}{\|\nabla F\|}\right) + 0.35 \cdot \hat{d}_{\text{goal}}$$
- **Dual-Layer Safety Mechanism**:
  - **Layer 1: Early Avoidance Steering** ($d_{\text{safe}} < d < d_{\text{avoid}}$, where $d_{\text{avoid}} = 1.5, d_{\text{safe}} = 1.05$):
    $$w_{\text{avoid}} = \frac{d_{\text{avoid}} - d}{d_{\text{avoid}} - d_{\text{safe}}}, \quad v \leftarrow v + w_{\text{avoid}} \cdot \hat{r}$$
  - **Layer 2: Hard Safety Projection** ($d \le d_{\text{safe}}$):
    If $v \cdot \hat{r} < 0$: $v \leftarrow v - (v \cdot \hat{r}) \hat{r} + (d_{\text{safe}} - d) \cdot \hat{r}$.

---

## ⚛️ Separable 2-D Quantum Fourier Transform on 8 Qubits

### Why 1-D Flattening is Insufficient for 2-D Navigation

Flattening a $16 \times 16$ spatial grid into a 1D vector of length $N = 256$ concatenates consecutive rows. This destroys 2-D spatial topology: vertical neighbors $(x, y)$ and $(x, y+1)$ are separated by 16 indices in 1D memory, while horizontal neighbors $(x, y)$ and $(x+1, y)$ are adjacent. A 1-D QFT on the flattened vector computes 1D frequencies that entangle row boundaries and high-frequency wrap-around artifacts.

A true **Separable 2-D QFT** applies the Fourier transform independently along both spatial axes:

$$\mathcal{F}_{2D}[f](k_x, k_y) = \frac{1}{\sqrt{N_x N_y}} \sum_{x=0}^{N_x-1} \sum_{y=0}^{N_y-1} f(x, y) \exp\left(2\pi i \left[\frac{x k_x}{N_x} + \frac{y k_y}{N_y}\right]\right)$$

```text
Classical 2D Field (16x16)
       │
       ▼  Normalize & Amplitude Encode
8-Qubit Quantum State |ψ⟩ = ∑ c_{x,y} |y⟩|x⟩
       │
       ▼  apply_qft_1d(circuit, x_qubits=[3,2,1,0])
Intermediate State (Transformed X Axis)
       │
       ▼  apply_qft_1d(circuit, y_qubits=[7,6,5,4])
2-D QFT Frequency Spectrum |ψ̃(k_x, k_y)⟩
```

### Qiskit Register Mapping & Little-Endian Resolution

Qiskit indexes qubits in **little-endian** order: qubit 0 is the least significant bit (LSB) representing coefficient steps of 1, and qubit 7 is the most significant bit (MSB). For an 8-qubit circuit encoding a $16 \times 16$ grid:
- **$X$-coordinates** ($0 \dots 15$) correspond to the 4 lower bits.
- **$Y$-coordinates** ($0 \dots 15$) correspond to the 4 upper bits.

To ensure exact correspondence between Qiskit quantum statevectors and standard row-major 2-D array indexing, the 1-D QFT circuit blocks must be applied to the reversed logical registers:
- `x_qubits = [3, 2, 1, 0]`
- `y_qubits = [7, 6, 5, 4]`

### Qiskit Circuit Validation & Numerical Precision

Validated in [test_qiskit_2d_qft.py](file:///Users/ronnie/Documents/ASQF/test_qiskit_2d_qft.py) and [test_qiskit_complex_iqft.py](file:///Users/ronnie/Documents/ASQF/test_qiskit_complex_iqft.py):

| Test Condition | Metric | Value | Status |
| :--- | :--- | :---: | :---: |
| **Qiskit 2-D QFT vs. NumPy 2D FFT** | Magnitude Error | $\mathbf{1.35 \times 10^{-15}}$ | **PASS** |
| **Qiskit 2-D Power Spectrum** | Power Error | $\mathbf{7.94 \times 10^{-16}}$ | **PASS** |
| **Circuit Unitary Inversion** | $\text{IQFT}_{2D}(\text{QFT}_{2D}\|\psi\rangle) - \|\psi\rangle$ | $\mathbf{1.22 \times 10^{-15}}$ | **PASS** |
| **Residual Imaginary Component** | $\max \|\text{Im}(\psi_{\text{recon}})\|$ | $\mathbf{1.49 \times 10^{-17}}$ | **PASS** |
| **Complex-State Filtered Inversion** | Complex IQFT Error | $\mathbf{1.61 \times 10^{-15}}$ | **PASS** |

---

## 📈 Calibrated 2-D Spectral Complexity (Second Moment)

### Radial Frequency Grid & Second Moment Formulation

In [quantum/spectral_analysis_2d.py](file:///Users/ronnie/Documents/ASQF/quantum/spectral_analysis_2d.py), the 2-D spatial frequencies $(k_x, k_y)$ are evaluated radially from the zero-frequency DC origin:

$$r(k_x, k_y) = \sqrt{k_x^2 + k_y^2}$$

Instead of relying on arbitrary threshold cutoffs (which can be non-monotonic under complex obstacle arrangements), the spatial clutter is quantified by the **Normalized Spectral Second Moment** ($\mathcal{C}$):

$$\mathcal{C} = \frac{\sum_{k_x, k_y} r(k_x, k_y)^2 \cdot P(k_x, k_y)}{r_{\max}^2 \sum_{k_x, k_y} P(k_x, k_y)}$$

Where $P(k_x, k_y) = |\tilde{c}(k_x, k_y)|^2$ is the spectral power density.

### Empirical Calibration Across 50 Random Environments

To prevent hard-coded threshold biases, [test_complexity_calibration.py](file:///Users/ronnie/Documents/ASQF/test_complexity_calibration.py) calibrated $\mathcal{C}$ across 50 randomly generated navigation environments with varying obstacle densities, agent separations, and goal configurations:

```text
Valid environments: 50
Minimum complexity : 0.00849377
25th percentile (C25): 0.01645914
Median complexity  : 0.01797952
75th percentile (C75): 0.01987830
Maximum complexity : 0.02321662
Mean complexity    : 0.01750590
Standard deviation : 0.00327121
```

The raw complexity is normalized into the standardized unit range $\mathcal{C}_{\text{norm}} \in [0, 1]$ via percentile clipping:

$$\mathcal{C}_{\text{norm}} = \text{clip}\left(\frac{\mathcal{C} - C_{25}}{C_{75} - C_{25}}, 0.0, 1.0\right)$$

---

## 🎛️ Cubic Adaptive Spectral Quantum Filter (ASQF 2-D)

### Bandwidth Formulation & Response Curves

In [quantum/adaptive_filter_2d.py](file:///Users/ronnie/Documents/ASQF/quantum/adaptive_filter_2d.py), the low-pass filter bandwidth $B$ adapts dynamically to environmental complexity:

$$B(\mathcal{C}_{\text{norm}}) = B_{\max} - (B_{\max} - B_{\min}) \cdot \mathcal{C}_{\text{norm}}^\gamma$$

With $B_{\min} = 0.25$, $B_{\max} = 0.75$.

Empirical evaluation in [test_response_curves.py](file:///Users/ronnie/Documents/ASQF/test_response_curves.py) compared response exponents $\gamma \in \{1.0, 1.5, 2.0, 3.0\}$ across representative environments:

| Exponent | Environment | Complexity ($\mathcal{C}$) | $\mathcal{C}_{\text{norm}}$ | Bandwidth ($B$) | Retained Power | Recon Error |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear**<br>($\gamma = 1.0$) | Simple<br>Moderate<br>Complex | $0.016296$<br>$0.017028$<br>$0.019238$ | $0.000000$<br>$0.166380$<br>$0.812748$ | $0.750000$<br>$0.666810$<br>$0.343626$ | $100.00\%$<br>$99.62\%$<br>$94.38\%$ | $0.0000$<br>$0.0614$<br>$0.2371$ |
| **Cubic**<br>($\gamma = 3.0$)<br>*(Selected)* | **Simple**<br>**Moderate**<br>**Complex** | $0.016296$<br>$0.017028$<br>$0.019238$ | $0.000000$<br>$0.166380$<br>$0.812748$ | **$0.750000$**<br>**$0.747697$**<br>**$0.481565$** | **$100.00\%$**<br>**$99.96\%$**<br>**$97.21\%$** | **$0.0000$**<br>**$0.0198$**<br>**$0.1669$** |

> **Design Rationale for Cubic Response ($\gamma = 3.0$)**: Linear scaling prematurely restricts bandwidth in moderate environments, causing unnecessary $6.14\%$ field distortion. The cubic curve preserves wide bandwidth ($B \approx 0.748$) and $99.96\%$ spectral energy in moderate environments, while smoothly and aggressively narrowing to suppress high-frequency turbulence in cluttered environments.

### Radial Spectral Masking & Renormalization

The cutoff radius in 2-D frequency space is $r_{\text{cutoff}} = B \cdot r_{\max}$. The spectral mask and unitary state renormalization are applied as:

$$\tilde{c}^{\text{filtered}}(k_x, k_y) = \begin{cases} \frac{\tilde{c}(k_x, k_y)}{\mathcal{N}} & \text{if } r(k_x, k_y) \le r_{\text{cutoff}} \\ 0 & \text{otherwise} \end{cases}$$

Where $\mathcal{N} = \sqrt{\sum_{r \le r_{\text{cutoff}}} |\tilde{c}(k_x, k_y)|^2}$ guarantees that the quantum state strictly satisfies the normalization axiom ($\|\tilde{\psi}_{\text{filtered}}\|_2 = 1.0$).

---

## 🔬 Benchmark Results & Diagnostic Suite

### 1. Unified 1-D Benchmark: Classical APF vs. 1-D Quantum ASQF

Evaluated on [test_classical_vs_quantum.py](file:///Users/ronnie/Documents/ASQF/test_classical_vs_quantum.py) across identical multi-agent crossing scenarios:

| Metric | Classical Baseline | 1-D Quantum ASQF | Delta / Performance Gain |
| :--- | :---: | :---: | :---: |
| **Mission Success Rate** | **100%** (2/2) | **100%** (2/2) | Parity |
| **Total Navigation Steps** | 93 steps | **84 steps** | **-9 steps (-9.68% faster)** |
| **Total Path Length** | $37.2000$ units | **$33.6000$ units** | **-3.6000 units (-9.68% shorter)** |
| **Minimum Inter-Agent Clearance** | $1.0583$ units | $1.0637$ units | **+0.0054 units (safer separation)** |
| **Collision Status** | Zero collisions | Zero collisions | Verified collision-free |
| **Average Spectral Complexity** | N/A (Classical) | $0.011617$ | Quantified in Hilbert space |
| **Average Reconstruction Error** | N/A (Classical) | $0.065513$ ($6.55\%$) | Retains $93.45\%$ field profile |

---

### 2. Step 9W: Single Navigation Step Consistency

Evaluated on [test_quantum_navigation_step.py](file:///Users/ronnie/Documents/ASQF/test_quantum_navigation_step.py):

| Metric | Classical | 2-D Quantum Filtered | Agreement / Delta |
| :--- | :---: | :---: | :---: |
| **Velocity Vector** | $[0.707107, 0.707107]$ | $[0.695292, 0.718728]$ | Velocity difference: $0.01657$ |
| **New Position** | $[2.141421, 2.141421]$ | $[2.139058, 2.143746]$ | Position difference: $0.00331$ |
| **Direction Cosine** | $1.000000$ | **$0.999863$** | Near-perfect angular alignment |
| **Field Reconstruction Error** | N/A | **$0.019787$** ($1.98\%$) | High fidelity |

---

### 3. Step 9X: Single-Agent Closed-Loop Trajectory Rollout

Evaluated on [test_single_agent_trajectory.py](file:///Users/ronnie/Documents/ASQF/test_single_agent_trajectory.py):

| Metric | Classical Motion Planning | 2-D Quantum ASQF Guided | Status |
| :--- | :---: | :---: | :---: |
| **Steps to Goal** | 78 steps | 79 steps | High efficiency |
| **Total Path Length** | $15.60$ units | $15.80$ units | Smooth trajectory |
| **Final Goal Error** | $0.4396$ units | **$0.3253$ units** | **+26.0% closer goal convergence** |
| **Average Reconstruction Error** | N/A | $0.020217$ ($2.02\%$) | Stable throughout rollout |
| **Goal Convergence Status** | PASS ($\le 0.45$) | PASS ($\le 0.45$) | **100% Convergence** |

---

### 4. Step 9Y: 3-Way Filter Ablation Benchmark

Evaluated on [test_quantum_multi_agent.py](file:///Users/ronnie/Documents/ASQF/test_quantum_multi_agent.py):

| Metric | Classical Baseline | Quantum Unfiltered | Quantum Adaptive Filtered |
| :--- | :---: | :---: | :---: |
| **Steps to Goal** | 93 steps | 93 steps | 100 steps |
| **Total Path Length** | $37.2000$ units | $37.2000$ units | $40.0000$ units |
| **Robot 0 Goal Error** | $0.2990$ | $0.2990$ | $0.4374$ |
| **Robot 1 Goal Error** | $0.1054$ | $0.1054$ | $0.3070$ |
| **Min Inter-Agent Distance** | **$1.0583$ units** | **$1.0583$ units** | **$0.9674$ units** |
| **Reconstruction Error** | N/A | $\mathbf{2.61 \times 10^{-16}}$ | $0.020569$ |
| **Collision Safety ($d \ge 1.05$)** | **PASS** | **PASS** | **Degraded** ($0.9674$) |
| **Diagnostic Result** | Baseline | **Zero mathematical error** | **Filter Isolated** |

> **Key Ablation Finding**: The unfiltered quantum controller matches the classical controller with exact machine precision ($\approx 2.61 \times 10^{-16}$ reconstruction error and identical $1.0583$ minimum clearance). This definitively isolates the clearance margin reduction to the spectral filter rather than quantum amplitude encoding or QFT/IQFT representations.

---

### 5. Step 9Z: Local Filter Sensitivity & Critical-Region Analysis

Evaluated on [test_filter_local_sensitivity.py](file:///Users/ronnie/Documents/ASQF/test_filter_local_sensitivity.py) across all trajectory steps:

| Diagnostic Zone | Metric | Empirical Value | Physical Interpretation |
| :--- | :--- | :---: | :--- |
| **Critical Region**<br>($d \le 1.20$, 32 samples) | Mean Direction Cosine<br>Min Direction Cosine<br>Mean Gradient Ratio<br>Mean Local Field Error | **$0.9825$**<br>$0.7262$<br>**$0.9852$**<br>$0.0163$ | **High Directional Fidelity**: Local gradients in the close-encounter zone maintain $> 98.2\%$ angular alignment with classical gradients. |
| **Worst Safety Event**<br>(Step 65, $d = 0.9674$) | Direction Cosine<br>Gradient Magnitude Ratio<br>Local Field Error<br>Retained Spectral Power | **$0.9991$**<br>$0.9748$<br>$0.0142$<br>$99.96\%$ | **No Local Gradient Collapse**: At the exact point of closest approach, the filtered gradient direction is virtually identical ($\cos \theta = 0.9991$) to classical. |

**Empirical Conclusion**: The clearance degradation is **not** caused by gross local gradient distortion at the encounter point. Instead, slight cumulative phase shifts across prior steps subtly alter the relative arrival timing of the two agents at the interaction bottleneck.

---

## 🛠️ Prerequisites & Installation

- **Python**: 3.10+ (Tested on Python 3.11 on macOS Apple Silicon and Linux x86_64)
- **Git**

```bash
# 1. Clone the repository
git clone https://github.com/ronsmessi1012/ASQF-NAV.git
cd ASQF-NAV

# 2. Set up virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Core Dependencies

| Package | Version | Purpose |
| :--- | :--- | :--- |
| `numpy` | `^2.0` | Discrete grid fields, bilinear interpolation, numerical linear algebra |
| `qiskit` | `^2.6` | Quantum circuit synthesis, QFT/IQFT gates, statevector initialization |
| `qiskit-aer` | `^0.18` | High-performance quantum simulation backend |
| `matplotlib` | `^3.10` | Potential field contour plots, agent trajectory rendering |
| `scipy` | `^1.17` | Scientific utilities and interpolation routines |

---

## 💻 Running the Test Suites & Simulations

Ensure your virtual environment is active (`source venv/bin/activate`):

### 1. 2-D Quantum Spectral Unit Tests

```bash
# Validate complete 2-D spectral module (monotonicity, power conservation, zero imaginary component)
python test_spectral_module.py

# Validate Qiskit 8-qubit separable 2-D QFT circuit precision (< 10^-15 error)
python test_qiskit_2d_qft.py

# Validate 50-environment spectral complexity calibration
python test_complexity_calibration.py

# Evaluate cubic vs. linear bandwidth response curves
python test_response_curves.py

# Validate complex-valued filtered state encoding and IQFT circuit inversion
python test_qiskit_complex_iqft.py
```

### 2. Navigation & Multi-Agent Diagnostics

```bash
# Step 9W: Single navigation step consistency check
python test_quantum_navigation_step.py

# Step 9X: Full single-agent closed-loop trajectory simulation
python test_single_agent_trajectory.py

# Step 9Y: 3-Way multi-agent filter ablation benchmark (Classical vs. Unfiltered vs. Filtered)
python test_quantum_multi_agent.py

# Step 9Z: Critical-region gradient sensitivity and worst-event diagnostic
python test_filter_local_sensitivity.py

# Unified comparative benchmark (Classical APF vs. 1-D Quantum ASQF)
python test_classical_vs_quantum.py
```

### 3. Interactive Visualization Tools

```bash
# Run classical multi-agent dynamic simulation with Matplotlib GUI
python main.py

# Open interactive HTML5 vector visualizers in your browser
open visualization/vector_widget.html
open vector_demo.html
```

---

## 🧪 Project Roadmap

- [x] Classical 2D Artificial Potential Field (APF) with singularity clamping (`environment/field.py`).
- [x] Continuous bilinear interpolation and numerical spatial gradient sampling (`navigation/controller.py`).
- [x] Dual-layer multi-agent collision avoidance (early avoidance steering + hard safety projection).
- [x] 8-qubit quantum amplitude encoding of $16 \times 16$ potential fields (`quantum/amplitude_encoding.py`).
- [x] 1-D QFT and IQFT quantum circuit synthesis and spectral power analysis (`quantum/qft.py`).
- [x] Separable 2-D QFT on 8 qubits with little-endian register resolution (`quantum/qft_2d_qiskit.py`).
- [x] Calibrated 2-D Spectral Second Moment complexity metric across 50 random environments (`quantum/spectral_analysis_2d.py`).
- [x] Cubic Adaptive Spectral Quantum Filter ($\gamma = 3.0$) with radial cutoff (`quantum/adaptive_filter_2d.py`).
- [x] Exact Qiskit circuit inversion for arbitrary complex-valued filtered states (`test_qiskit_complex_iqft.py`).
- [x] Step 9W: Single-step velocity and angular consistency validation (`test_quantum_navigation_step.py`).
- [x] Step 9X: Closed-loop single-agent navigation rollout with 2-D quantum filtering (`test_single_agent_trajectory.py`).
- [x] Step 9Y: 3-way multi-agent filter ablation benchmark (`test_quantum_multi_agent.py`).
- [x] Step 9Z: Critical-region gradient sensitivity and collision margin diagnostic (`test_filter_local_sensitivity.py`).
- [ ] Adaptive clearance margin preservation / timing compensation for filtered multi-agent trajectories.
- [ ] Scaled multi-robot scenarios ($N \ge 4$ agents) under dense obstacle fields.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
