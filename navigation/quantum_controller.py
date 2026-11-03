import numpy as np

from quantum.amplitude_encoding import encode_field
from quantum.qft import apply_qft
from quantum.spectral_analysis import spectral_complexity
from quantum.adaptive_filter import adaptive_spectral_filter
from quantum.reconstruction import reconstruct_field


MAX_SPEED = 1.0


def quantum_reconstruct_field(field):
    """
    Run the complete quantum spectral pipeline:

        Field
        -> Amplitude Encoding
        -> QFT
        -> Spectral Complexity
        -> Adaptive Filtering
        -> IQFT
        -> Reconstructed Field
    """

    # -------------------------------------------------
    # 1. Amplitude encoding
    # -------------------------------------------------

    circuit, amplitudes = encode_field(field)

    n_qubits = circuit.num_qubits

    # -------------------------------------------------
    # 2. QFT
    # -------------------------------------------------

    apply_qft(circuit, n_qubits)

    from qiskit.quantum_info import Statevector

    qft_state = Statevector.from_instruction(circuit)

    spectrum = np.asarray(qft_state.data)

    # -------------------------------------------------
    # 3. Spectral complexity
    # -------------------------------------------------

    complexity = spectral_complexity(spectrum)

    # -------------------------------------------------
    # 4. Adaptive spectral filtering
    # -------------------------------------------------

    (
        filtered_spectrum,
        bandwidth,
        cutoff,
        retained_fraction,
        mask
    ) = adaptive_spectral_filter(
        spectrum,
        complexity
    )

    # -------------------------------------------------
    # 5. Inverse QFT
    # -------------------------------------------------

    reconstructed_field, reconstructed_state = (
        reconstruct_field(
            filtered_spectrum,
            field.shape
        )
    )

    return {
        "field": field,
        "amplitudes": amplitudes,
        "spectrum": spectrum,
        "filtered_spectrum": filtered_spectrum,
        "reconstructed_field": reconstructed_field,
        "reconstructed_state": reconstructed_state,
        "complexity": complexity,
        "bandwidth": bandwidth,
        "cutoff": cutoff,
        "retained_fraction": retained_fraction,
        "mask": mask
    }


def compute_quantum_velocity(
    reconstructed_field,
    position,
    goal
):
    """
    Compute navigation velocity using the
    reconstructed quantum field.
    """

    x, y = position

    rows, cols = reconstructed_field.shape

    # -------------------------------------------------
    # Continuous bilinear interpolation
    # -------------------------------------------------

    x = np.clip(x, 0.0, cols - 1.001)
    y = np.clip(y, 0.0, rows - 1.001)

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))

    x1 = min(x0 + 1, cols - 1)
    y1 = min(y0 + 1, rows - 1)

    dx = x - x0
    dy = y - y0

    # -------------------------------------------------
    # Numerical gradient
    # -------------------------------------------------

    gy, gx = np.gradient(reconstructed_field)

    gx_value = (
        (1 - dx) * (1 - dy) * gx[y0, x0]
        + dx * (1 - dy) * gx[y0, x1]
        + (1 - dx) * dy * gx[y1, x0]
        + dx * dy * gx[y1, x1]
    )

    gy_value = (
        (1 - dx) * (1 - dy) * gy[y0, x0]
        + dx * (1 - dy) * gy[y0, x1]
        + (1 - dx) * dy * gy[y1, x0]
        + dx * dy * gy[y1, x1]
    )

    gradient = np.array([
        gx_value,
        gy_value
    ])

    # -------------------------------------------------
    # Gradient descent
    # -------------------------------------------------

    gradient_norm = np.linalg.norm(gradient)

    if gradient_norm < 1e-12:
        gradient_velocity = np.zeros(2)
    else:
        gradient_velocity = -gradient / gradient_norm

    # -------------------------------------------------
    # Goal direction
    # -------------------------------------------------

    goal_vector = np.asarray(goal) - np.asarray(position)

    goal_distance = np.linalg.norm(goal_vector)

    if goal_distance < 1e-12:
        goal_direction = np.zeros(2)
    else:
        goal_direction = goal_vector / goal_distance

    # -------------------------------------------------
    # Combine field and goal direction
    # -------------------------------------------------

    velocity = (
        0.65 * gradient_velocity
        + 0.35 * goal_direction
    )

    velocity_norm = np.linalg.norm(velocity)

    if velocity_norm < 1e-12:
        return np.zeros(2)

    velocity = velocity / velocity_norm

    return MAX_SPEED * velocity