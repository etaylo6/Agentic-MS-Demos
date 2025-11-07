import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union


UNCERTAINTY_WEIGHT = 1.0
ERROR_WEIGHT = 0.05


# ---------------------------------------------------------------------------
# Uncertainty propagation utilities
# ---------------------------------------------------------------------------

def _as_scalar(value: Any) -> float:
    """Coerce numpy scalars/arrays to float while rejecting non-scalar outputs."""
    if np.isscalar(value):
        return float(value)
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("Expected scalar output when propagating uncertainty.")
    return float(array.reshape(()))


def propagate_uncertainty(
    func,
    values: Mapping[str, float],
    uncertainties: Mapping[str, float],
    *,
    covariance: Optional[Union[np.ndarray, Mapping[Tuple[str, str], float]]] = None,
    relative_step: float = 1e-6,
    absolute_step: float = 1e-9,
    return_gradient: bool = False,
) -> Dict[str, Any]:
    """
    Propagate input uncertainties through a scalar function using a first-order approximation.

    Parameters
    ----------
    func : callable
        Scalar function to evaluate. Must accept keyword arguments corresponding to ``values``.
    values : Mapping[str, float]
        Nominal values for each parameter of ``func``.
    uncertainties : Mapping[str, float]
        One-sigma uncertainties (standard deviations) for the same parameters.
    covariance : array-like or Mapping[(str, str), float], optional
        Full covariance information. When omitted, independent inputs are assumed.
    relative_step : float, optional
        Relative perturbation size used for numerical differentiation.
    absolute_step : float, optional
        Minimum perturbation magnitude used for numerical differentiation.
    return_gradient : bool, optional
        When True, include the computed gradient in the result.

    Returns
    -------
    dict
        Contains ``value`` (nominal function value), ``std`` (standard deviation),
        ``variance`` and optionally ``gradient`` (mapping parameter -> partial derivative).
    """
    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping from parameter name to nominal value.")
    if not isinstance(uncertainties, Mapping):
        raise TypeError("uncertainties must be a mapping from parameter name to sigma.")

    ordered_names = list(values.keys())
    if not ordered_names:
        raise ValueError("At least one parameter is required for uncertainty propagation.")

    value_map: Dict[str, float] = {}
    sigma_map: Dict[str, float] = {}
    for name in ordered_names:
        try:
            value_map[name] = float(values[name])
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise KeyError(f"Missing nominal value for parameter '{name}'.") from exc
        sigma_map[name] = float(uncertainties.get(name, 0.0))

    def call(current: Mapping[str, float]) -> float:
        try:
            return _as_scalar(func(**current))
        except TypeError as err:
            raise TypeError(
                f"Failed to call {func.__name__} with keyword arguments {current.keys()}."
            ) from err

    nominal_value = call(value_map)

    gradient = np.zeros(len(ordered_names), dtype=float)
    for idx, name in enumerate(ordered_names):
        sigma = sigma_map[name]
        if sigma == 0.0:
            gradient[idx] = 0.0
            continue
        base = value_map[name]
        step = max(abs(base) * relative_step, sigma * relative_step, absolute_step)
        forward = dict(value_map)
        backward = dict(value_map)
        forward[name] = base + step
        backward[name] = base - step
        f_plus = call(forward)
        f_minus = call(backward)
        gradient[idx] = (f_plus - f_minus) / (2.0 * step)

    sigma_vector = np.array([sigma_map[name] for name in ordered_names], dtype=float)
    if covariance is None:
        variance = float(np.sum((gradient * sigma_vector) ** 2))
    elif isinstance(covariance, Mapping):
        variance = 0.0
        for i, name_i in enumerate(ordered_names):
            for j, name_j in enumerate(ordered_names):
                if (name_i, name_j) in covariance:
                    cov_ij = float(covariance[(name_i, name_j)])
                elif (name_j, name_i) in covariance:
                    cov_ij = float(covariance[(name_j, name_i)])
                elif name_i == name_j:
                    cov_ij = sigma_map[name_i] ** 2
                else:
                    cov_ij = 0.0
                variance += gradient[i] * gradient[j] * cov_ij
        variance = float(variance)
    else:
        cov_arr = np.asarray(covariance, dtype=float)
        expected_shape = (len(ordered_names), len(ordered_names))
        if cov_arr.shape != expected_shape:
            raise ValueError(
                f"Covariance matrix must have shape {expected_shape}, got {cov_arr.shape}."
            )
        variance = float(gradient @ cov_arr @ gradient.T)

    variance = max(variance, 0.0)
    std_dev = float(np.sqrt(variance))

    result: Dict[str, Any] = {
        "value": nominal_value,
        "std": std_dev,
        "variance": variance,
    }
    if return_gradient:
        result["gradient"] = {name: gradient[idx] for idx, name in enumerate(ordered_names)}
    return result


# ---------------------------------------------------------------------------
# Beam theory models
# ---------------------------------------------------------------------------

def timoshenko_w(P: float, E: float, I: float, L: float, k: float, G: float, A: float) -> float:
    """Timoshenko-Ehrenfest beam deflection under end load."""
    delta_bending = P * L**3 / (3.0 * E * I)
    delta_shear = P * L / (k * G * A)
    return delta_bending + delta_shear


def euler_bernoulli_w(P: float, L: float, E: float, I: float) -> float:
    """Euler-Bernoulli beam deflection under end load."""
    return P * L**3 / (3.0 * E * I)


# ---------------------------------------------------------------------------
# Model comparison structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelMetrics:
    name: str
    prediction: float
    uncertainty: float
    uncertainty_fraction: float
    relative_error: float
    cost: float


@dataclass(frozen=True)
class DOERecord:
    k_u: float
    G_u: float
    timoshenko: ModelMetrics
    euler_bernoulli: ModelMetrics
    selected_model: str


def evaluate_models(
    values: Mapping[str, float],
    uncertainties: Mapping[str, float],
) -> Tuple[ModelMetrics, ModelMetrics]:
    """Evaluate both beam theories and compute cost metrics."""
    value_map = dict(values)
    unc_map = dict(uncertainties)

    te_params = {name: value_map[name] for name in ("P", "E", "I", "L", "k", "G", "A")}
    te_unc = {name: unc_map.get(name, 0.0) for name in te_params}
    te_result = propagate_uncertainty(timoshenko_w, te_params, te_unc)
    te_prediction = te_result["value"]
    te_uncertainty = te_result["std"]
    te_uncertainty_fraction = te_uncertainty / max(abs(te_prediction), 1e-12)
    te_metrics = ModelMetrics(
        name="Timoshenko",
        prediction=te_prediction,
        uncertainty=te_uncertainty,
        uncertainty_fraction=te_uncertainty_fraction,
        relative_error=0.0,
        cost=UNCERTAINTY_WEIGHT * te_uncertainty_fraction,
    )

    eb_params = {name: value_map[name] for name in ("P", "L", "E", "I")}
    eb_unc = {name: unc_map.get(name, 0.0) for name in eb_params}
    eb_result = propagate_uncertainty(euler_bernoulli_w, eb_params, eb_unc)
    eb_prediction = eb_result["value"]
    eb_uncertainty = eb_result["std"]
    eb_uncertainty_fraction = eb_uncertainty / max(abs(eb_prediction), 1e-12)
    rel_error = abs(eb_prediction - te_prediction) / max(abs(te_prediction), 1e-12)
    eb_metrics = ModelMetrics(
        name="Euler-Bernoulli",
        prediction=eb_prediction,
        uncertainty=eb_uncertainty,
        uncertainty_fraction=eb_uncertainty_fraction,
        relative_error=rel_error,
        cost=UNCERTAINTY_WEIGHT * eb_uncertainty_fraction + ERROR_WEIGHT * rel_error,
    )

    return te_metrics, eb_metrics


def evaluate_scenario(
    values: Mapping[str, float],
    base_uncertainties: Mapping[str, float],
    k_sigma: float,
    G_sigma: float,
) -> DOERecord:
    """Evaluate a single design point for k_u and G_u."""
    uncertainties = dict(base_uncertainties)
    uncertainties["k"] = k_sigma
    uncertainties["G"] = G_sigma

    te_metrics, eb_metrics = evaluate_models(values, uncertainties)
    selected = te_metrics.name if te_metrics.cost <= eb_metrics.cost else eb_metrics.name

    return DOERecord(
        k_u=k_sigma,
        G_u=G_sigma,
        timoshenko=te_metrics,
        euler_bernoulli=eb_metrics,
        selected_model=selected,
    )


def run_design_of_experiments(
    values: Mapping[str, float],
    base_uncertainties: Mapping[str, float],
    k_u_values: Sequence[float],
    G_u_values: Sequence[float],
) -> Sequence[DOERecord]:
    """Perform a grid search over k_u and G_u uncertainties."""
    records = []
    for G_sigma in G_u_values:
        for k_sigma in k_u_values:
            records.append(evaluate_scenario(values, base_uncertainties, k_sigma, G_sigma))
    return records


def print_selection_matrix(
    records: Sequence[DOERecord],
    k_u_values: Sequence[float],
    G_u_values: Sequence[float],
) -> None:
    """Render a matrix showing which model is selected for each DOE scenario."""
    idx = 0
    header = ["G_u \\ k_u"] + [f"{k:.3f}" for k in k_u_values]
    print("\nModel selection matrix")
    print("-" * (len(header) * 10))
    print(" | ".join(header))

    for G_sigma in G_u_values:
        row = [f"{G_sigma/1e9:.2f}e9"]
        for _ in k_u_values:
            record = records[idx]
            choice = "TE" if record.selected_model == "Timoshenko" else "EB"
            row.append(choice)
            idx += 1
        print(" | ".join(row))
    print("-" * (len(header) * 10))


def print_sample_metrics(record: DOERecord, title: str) -> None:
    """Display detailed metrics for a single DOE record."""
    print(f"\n{title} (k_u={record.k_u:.4f}, G_u={record.G_u:.2e})")
    for metrics in (record.timoshenko, record.euler_bernoulli):
        print(
            f"  {metrics.name:16s} "
            f"prediction={metrics.prediction:.6f} m, "
            f"uncertainty={metrics.uncertainty:.6f} m "
            f"({metrics.uncertainty_fraction:.6f} rel), "
            f"relative_error={metrics.relative_error:.6f}, "
            f"cost={metrics.cost:.6f}"
        )
    print(f"  Selected model: {record.selected_model}")


def main() -> None:
    """Reproduce the model-selection experiment described in the paper section."""
    radius = 0.10  # meters
    area = np.pi * radius**2

    values = {
        "P": 1000.0,      # N
        "E": 200e9,       # Pa
        "I": 0.002,       # m^4
        "L": 5.0,         # m
        "k": 0.85,        # shear correction
        "G": 79.3e9,      # Pa
        "A": area,        # m^2
    }

    base_uncertainties = {
        "P": 20.0,                 # N
        "E": 5.0e9,                # Pa
        "I": 1.0e-4,               # m^4
        "L": 0.002,                # m
        "A": area * 0.02,          # m^2
        "k": 0.0,                  # placeholder, overwritten in DOE
        "G": 0.0,                  # placeholder, overwritten in DOE
    }

    k_u_values = np.linspace(0.0, 0.5, 11)     # shear-factor uncertainty sweep
    G_u_values = np.linspace(0.0, 40e9, 11)    # shear-modulus uncertainty sweep

    records = run_design_of_experiments(values, base_uncertainties, k_u_values, G_u_values)

    selection_counts = Counter(record.selected_model for record in records)
    baseline_record = records[0]

    print("Beam model selection based on uncertainty-weighted cost (e = U_T + E_T)")
    print(f"  Uncertainty weight: {UNCERTAINTY_WEIGHT:.2f}")
    print(f"  Relative error weight: {ERROR_WEIGHT:.2f}")
    print(f"Slenderness ratio (L/radius): {values['L'] / radius:.1f}")
    print(f"Total DOE scenarios: {len(records)}")
    for model_name, count in selection_counts.items():
        print(f"  {model_name:16s}: {count} selections")

    print_sample_metrics(baseline_record, "Baseline scenario")
    worst_te = max(records, key=lambda record: record.timoshenko.cost)
    print_sample_metrics(worst_te, "High-uncertainty scenario")
    print_selection_matrix(records, k_u_values, G_u_values)


if __name__ == "__main__":
    main()

