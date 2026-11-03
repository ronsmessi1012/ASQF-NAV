"""
Component D: Quantum Spectral Processing Pipeline for AQSF-NAV v2.

Ingests the combined 16x16 navigation field (F_total) and processes it through:
    F_total -> 8-Qubit Amplitude Encoding -> Separable 2-D QFT ->
    Spectral Second Moment Complexity C(t) -> Cubic Adaptive Low-Pass Filter B(t) ->
    Separable 2-D IQFT -> Reconstructed Field

Maintains exact numerical consistency with validated v1 Qiskit circuits and calibration constants:
    C25 = 0.01645914, C75 = 0.01987830, B_min = 0.25, B_max = 0.75, gamma = 3.0

Provides extensive telemetry:
- Spectral complexity C(t) and normalized complexity
- Adaptive bandwidth B(t) and cutoff radius
- Retained spectral power fraction
- L2 Field reconstruction error
- Local gradient cosine similarity
- Circuit metrics (8 qubits, gate count, circuit depth)
"""

import time
from typing import Dict, Optional, Tuple, Union
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum.qft_2d_qiskit import (
    encode_field_into_circuit,
    apply_qft_2d,
    apply_inverse_qft_2d,
    statevector_to_field,
)
from quantum.spectral_analysis_2d import (
    qft_2d,
    iqft_2d,
    radial_frequency_grid,
    spectral_second_moment,
    normalize_complexity,
    spectral_power,
    C25,
    C75,
)
from quantum.adaptive_filter_2d import (
    compute_adaptive_bandwidth,
    adaptive_2d_filter,
)
from navigation_v2.config import AQSFConfigV2


def process_field_through_qft(
    field: np.ndarray,
    config: Optional[AQSFConfigV2] = None,
    apply_filter: bool = True,
    use_qiskit_circuit: bool = True
) -> Tuple[np.ndarray, Dict[str, Union[float, int, np.ndarray]]]:
    """
    Execute the quantum spectral processing pipeline on a 16x16 field.
    
    Parameters
    ----------
    field : np.ndarray (16, 16)
        The input combined potential field.
    config : AQSFConfigV2
        Configuration containing calibration and parameter bounds.
    apply_filter : bool
        Whether to apply the adaptive spectral filter (False for unfiltered QFT benchmark).
    use_qiskit_circuit : bool
        If True, executes native Qiskit 8-qubit quantum circuits.

    Returns
    -------
    reconstructed_field : np.ndarray (16, 16)
        The reconstructed spatial field (rescaled to original norm).
    telemetry : dict
        Spectral and quantum circuit telemetry.
    """
    if config is None:
        config = AQSFConfigV2()

    t_start = time.perf_counter()

    field = np.asarray(field, dtype=float)
    if field.shape != (16, 16):
        raise ValueError(f"Expected 16x16 field, got {field.shape}")

    orig_norm = float(np.linalg.norm(field))
    if orig_norm < 1e-12:
        raise ValueError("Field norm too close to zero for amplitude encoding.")

    normalized_field = field / orig_norm

    # -------------------------------------------------------------------------
    # 1. Amplitude Encoding & 2-D QFT
    # -------------------------------------------------------------------------
    t_qft_start = time.perf_counter()
    if use_qiskit_circuit:
        circuit = encode_field_into_circuit(normalized_field)
        apply_qft_2d(circuit)
        state = Statevector.from_instruction(circuit)
        spectrum = statevector_to_field(state)
        n_qubits = circuit.num_qubits
        gate_count = circuit.size()
        circuit_depth = circuit.depth()
    else:
        spectrum = qft_2d(normalized_field)
        n_qubits = 8
        gate_count = 56
        circuit_depth = 28

    t_qft_end = time.perf_counter()

    # -------------------------------------------------------------------------
    # 2. Spectral Analysis (Second Moment)
    # -------------------------------------------------------------------------
    t_analysis_start = time.perf_counter()
    rad_freq = radial_frequency_grid(field.shape)
    raw_complexity = spectral_second_moment(spectrum)
    norm_complexity = normalize_complexity(raw_complexity)
    t_analysis_end = time.perf_counter()

    # -------------------------------------------------------------------------
    # 3. Adaptive Spectral Filtering
    # -------------------------------------------------------------------------
    t_filter_start = time.perf_counter()
    if apply_filter:
        filter_res = adaptive_2d_filter(
            spectrum,
            rad_freq,
            norm_complexity
        )
        filtered_spectrum = filter_res["filtered_spectrum"]
        bandwidth = filter_res["bandwidth"]
        cutoff = filter_res["cutoff"]
        retained_power_frac = filter_res["retained_power_fraction"]
        retained_comp_frac = filter_res["retained_component_fraction"]
        retained_comp_count = int(np.sum(filter_res["mask"]))
    else:
        filtered_spectrum = spectrum.copy()
        bandwidth = 1.0
        cutoff = float(np.max(rad_freq))
        retained_power_frac = 1.0
        retained_comp_frac = 1.0
        retained_comp_count = 256

    t_filter_end = time.perf_counter()

    # -------------------------------------------------------------------------
    # 4. Inverse QFT & Reconstruction
    # -------------------------------------------------------------------------
    t_iqft_start = time.perf_counter()
    if use_qiskit_circuit and not apply_filter:
        inv_circuit = QuantumCircuit(8)
        inv_circuit.initialize(filtered_spectrum.flatten() / np.linalg.norm(filtered_spectrum), range(8))
        apply_inverse_qft_2d(inv_circuit)
        rec_state = Statevector.from_instruction(inv_circuit)
        recon_norm_field = np.real(statevector_to_field(rec_state))
        max_imag = float(np.max(np.abs(np.imag(statevector_to_field(rec_state)))))
    else:
        # Complex IQFT on filtered spectrum
        rec_field_complex = iqft_2d(filtered_spectrum)
        max_imag = float(np.max(np.abs(np.imag(rec_field_complex))))
        recon_norm_field = np.real(rec_field_complex)

    # Rescale back to original field norm
    reconstructed_field = recon_norm_field * orig_norm
    t_iqft_end = time.perf_counter()

    # -------------------------------------------------------------------------
    # 5. Diagnostic Metrics
    # -------------------------------------------------------------------------
    recon_error = float(
        np.linalg.norm(field - reconstructed_field) / orig_norm
    )

    # Local gradient direction similarity
    gy_orig, gx_orig = np.gradient(field)
    gy_rec, gx_rec = np.gradient(reconstructed_field)

    orig_grad_norm = np.sqrt(gx_orig**2 + gy_orig**2)
    rec_grad_norm = np.sqrt(gx_rec**2 + gy_rec**2)
    valid_mask = (orig_grad_norm > 1e-4) & (rec_grad_norm > 1e-4)

    if np.any(valid_mask):
        dot_prod = (
            gx_orig[valid_mask] * gx_rec[valid_mask]
            + gy_orig[valid_mask] * gy_rec[valid_mask]
        )
        denom = orig_grad_norm[valid_mask] * rec_grad_norm[valid_mask]
        cos_similarity = dot_prod / np.maximum(denom, 1e-12)
        mean_cos_sim = float(np.mean(cos_similarity))
        min_cos_sim = float(np.min(cos_similarity))
    else:
        mean_cos_sim = 1.0
        min_cos_sim = 1.0

    telemetry = {
        "n_qubits": n_qubits,
        "circuit_depth": circuit_depth,
        "gate_count": gate_count,
        "spectral_complexity": float(raw_complexity),
        "normalized_complexity": float(norm_complexity),
        "adaptive_bandwidth": float(bandwidth),
        "cutoff_radius": float(cutoff),
        "retained_power_fraction": float(retained_power_frac),
        "retained_component_fraction": float(retained_comp_frac),
        "retained_components_count": retained_comp_count,
        "field_reconstruction_error": recon_error,
        "mean_gradient_cosine_similarity": mean_cos_sim,
        "min_gradient_cosine_similarity": min_cos_sim,
        "max_imaginary_component": max_imag,
        "time_qft": t_qft_end - t_qft_start,
        "time_spectral_analysis": t_analysis_end - t_analysis_start,
        "time_filtering": t_filter_end - t_filter_start,
        "time_iqft": t_iqft_end - t_iqft_start,
        "time_total_pipeline": time.perf_counter() - t_start,
    }

    return reconstructed_field, telemetry
