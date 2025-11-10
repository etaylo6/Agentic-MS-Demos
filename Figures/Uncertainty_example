import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union


UNCERTAINTY_WEIGHT = 1.0
ERROR_WEIGHT = 0.05
SLENDERNESS_THRESHOLD = 10.0


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
    absolute_error: float
    cost: float


@dataclass(frozen=True)
class DOERecord:
    length: float
    slenderness_ratio: float
    eb_allowed: bool
    relative_uncertainty: float
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
        absolute_error=0.0,
        cost=UNCERTAINTY_WEIGHT * te_uncertainty_fraction,
    )

    eb_params = {name: value_map[name] for name in ("P", "L", "E", "I")}
    eb_unc = {name: unc_map.get(name, 0.0) for name in eb_params}
    eb_result = propagate_uncertainty(euler_bernoulli_w, eb_params, eb_unc)
    eb_prediction = eb_result["value"]
    eb_uncertainty = eb_result["std"]
    eb_uncertainty_fraction = eb_uncertainty / max(abs(eb_prediction), 1e-12)
    abs_error = abs(eb_prediction - te_prediction)
    eb_metrics = ModelMetrics(
        name="Euler-Bernoulli",
        prediction=eb_prediction,
        uncertainty=eb_uncertainty,
        uncertainty_fraction=eb_uncertainty_fraction,
        absolute_error=abs_error,
        cost=UNCERTAINTY_WEIGHT * eb_uncertainty_fraction + ERROR_WEIGHT * abs_error,
    )

    return te_metrics, eb_metrics


def evaluate_scenario(
    values: Mapping[str, float],
    base_uncertainties: Mapping[str, float],
    length: float,
    relative_uncertainty: float,
) -> DOERecord:
    """Evaluate a single design point for the shared relative uncertainty."""
    updated_values = dict(values)
    updated_values["L"] = length

    uncertainties = dict(base_uncertainties)
    k_sigma = updated_values["k"] * relative_uncertainty
    G_sigma = updated_values["G"] * relative_uncertainty
    uncertainties["k"] = k_sigma
    uncertainties["G"] = G_sigma

    te_metrics, eb_metrics = evaluate_models(updated_values, uncertainties)

    radius = np.sqrt(updated_values["A"] / np.pi)
    slenderness_ratio = updated_values["L"] / max(radius, 1e-12)
    eb_allowed = slenderness_ratio >= SLENDERNESS_THRESHOLD

    if not eb_allowed:
        eb_metrics = replace(eb_metrics, cost=float("inf"))

    selected = (
        te_metrics.name
        if (not eb_allowed or te_metrics.cost <= eb_metrics.cost)
        else eb_metrics.name
    )

    return DOERecord(
        length=length,
        slenderness_ratio=slenderness_ratio,
        eb_allowed=eb_allowed,
        relative_uncertainty=relative_uncertainty,
        k_u=k_sigma,
        G_u=G_sigma,
        timoshenko=te_metrics,
        euler_bernoulli=eb_metrics,
        selected_model=selected,
    )


def run_design_of_experiments(
    values: Mapping[str, float],
    base_uncertainties: Mapping[str, float],
    length_values: Sequence[float],
    relative_uncertainties: Sequence[float],
) -> Sequence[DOERecord]:
    """Perform a grid search over length and shared relative uncertainties."""
    records = []
    for length in length_values:
        for rel_unc in relative_uncertainties:
            records.append(
                evaluate_scenario(values, base_uncertainties, length, rel_unc)
            )
    return records


def plot_selection_matrix(
    records: Sequence[DOERecord],
    length_values: Sequence[float],
    relative_uncertainties: Sequence[float],
) -> None:
    """Display a publication-quality heatmap of model selection choices."""
    if len(length_values) == 0 or len(relative_uncertainties) == 0:
        return

    selection = np.zeros((len(length_values), len(relative_uncertainties)), dtype=int)
    slenderness_by_length = np.full(len(length_values), np.nan)

    for i, length in enumerate(length_values):
        for j, rel_unc in enumerate(relative_uncertainties):
            record = next(
                (
                    rec
                    for rec in records
                    if np.isclose(rec.length, length) and np.isclose(rec.relative_uncertainty, rel_unc)
                ),
                None,
            )
            if record is None:
                raise ValueError(
                    f"No DOE record found for length={length} and relative uncertainty={rel_unc}."
                )
            if np.isnan(slenderness_by_length[i]):
                slenderness_by_length[i] = record.slenderness_ratio

            selection[i, j] = 0 if record.selected_model == "Timoshenko" else 1

    # Fill any remaining slenderness entries using available records (defensive)
    for i, length in enumerate(length_values):
        if not np.isnan(slenderness_by_length[i]):
            continue
        record = next(
            (rec for rec in records if np.isclose(rec.length, length)),
            None,
        )
        if record is not None:
            slenderness_by_length[i] = record.slenderness_ratio

    def _cell_edges(array: Sequence[float]) -> np.ndarray:
        arr = np.asarray(array, dtype=float)
        if arr.size == 1:
            delta = max(abs(arr[0]) * 0.05, 0.05)
            return np.array([arr[0] - delta, arr[0] + delta])
        diffs = np.diff(arr)
        start = arr[0] - diffs[0] / 2.0
        end = arr[-1] + diffs[-1] / 2.0
        centers = arr[:-1] + diffs / 2.0
        return np.concatenate(([start], centers, [end]))

    clemson_deep_purple = "#381460"
    clemson_lavender = "#CFBEF3"
    cmap = ListedColormap([clemson_deep_purple, clemson_lavender])
    legend_handles = [
        Patch(facecolor=cmap(0), label="Timoshenko-Ehrenfest"),
        Patch(facecolor=cmap(1), label="Euler-Bernoulli"),
    ]

    percent_values = np.asarray(relative_uncertainties) * 100.0
    x_edges = _cell_edges(percent_values)
    y_edges = _cell_edges(slenderness_by_length)

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=200)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    _ = ax.pcolormesh(x_edges, y_edges, selection, cmap=cmap, vmin=-0.5, vmax=1.5, shading="auto")

    x_ticks = np.linspace(percent_values[0], percent_values[-1], 6)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{val:.0f}%" for val in x_ticks])
    ax.set_xlabel("Shared uncertainty σ (k, G) [% of nominal]", labelpad=10)

    y_min = float(np.nanmin(slenderness_by_length))
    y_max = float(np.nanmax(slenderness_by_length))
    y_ticks = np.arange(np.ceil(y_min), np.floor(y_max) + 1, 5)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{val:.0f}" for val in y_ticks])
    ax.set_ylabel("Slenderness ratio, S [–]", labelpad=12)

    ax.tick_params(axis="both", colors="#342a44", length=5, width=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend = ax.legend(handles=legend_handles, loc="upper right", frameon=False, title="Model")
    legend.remove()

    te_mask = selection == 0
    eb_mask = selection == 1
    if te_mask.any():
        te_y, te_x = np.where(te_mask)
        te_x_mid = percent_values[te_x].min() + 0.25 * (percent_values[te_x].max() - percent_values[te_x].min())
        te_y_mid = 35.0
        ax.text(
            te_x_mid,
            te_y_mid,
            "Timoshenko-Ehrenfest",
            color="white",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
    if eb_mask.any():
        eb_y, eb_x = np.where(eb_mask)
        eb_x_mid = percent_values[eb_x].mean()
        eb_y_mid = slenderness_by_length[eb_y].mean()
        ax.text(
            eb_x_mid,
            eb_y_mid,
            "Euler-Bernoulli",
            color="#1b102d",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.show()


def print_sample_metrics(record: DOERecord, title: str) -> None:
    """Display detailed metrics for a single DOE record."""
    print(
        f"\n{title} (L={record.length:.2f} m, S={record.slenderness_ratio:.1f}, "
        f"σ_rel={record.relative_uncertainty*100:.1f}%, k_u={record.k_u:.4f}, "
        f"G_u={record.G_u:.2e})"
    )
    for metrics in (record.timoshenko, record.euler_bernoulli):
        print(
            f"  {metrics.name:16s} "
            f"prediction={metrics.prediction:.6f} m, "
            f"uncertainty={metrics.uncertainty:.6f} m "
            f"({metrics.uncertainty_fraction:.6f} rel), "
            f"absolute_error={metrics.absolute_error:.6f}, "
            f"cost={metrics.cost:.6f}"
        )
    print(f"  Selected model: {record.selected_model}")
    if not record.eb_allowed:
        print(f"  Euler-Bernoulli disallowed (S < {SLENDERNESS_THRESHOLD:.0f})")


def main() -> None:
    """Reproduce the model-selection experiment described in the paper section."""
    radius = 0.10  # meters
    area = np.pi * radius**2

    values = {
        "P": 1000.0,      # N
        "E": 200e9,       # Pa
        "I": 0.002,       # m^4
        "L": 5.0,         # m (baseline length)
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

    slenderness_values = np.arange(5, 51, 1, dtype=float)  # whole-number slenderness
    length_values = slenderness_values * radius  # convert to beam lengths
    relative_uncertainties = np.linspace(0.0, 0.4, 50)  # shared sigma fractions (0% to 40%)

    records = run_design_of_experiments(
        values,
        base_uncertainties,
        length_values,
        relative_uncertainties,
    )

    selection_counts = Counter(record.selected_model for record in records)
    baseline_record = next(
        record
        for record in records
        if np.isclose(record.length, values["L"])
        and np.isclose(record.relative_uncertainty, relative_uncertainties[0])
    )

    slenderness_values = [record.slenderness_ratio for record in records]

    print("Beam model selection based on uncertainty-weighted cost (e = U_T + E_T)")
    print(f"  Uncertainty weight: {UNCERTAINTY_WEIGHT:.2f}")
    print(f"  Absolute error weight: {ERROR_WEIGHT:.2f}")
    print(
        f"  Slenderness range: "
        f"{min(slenderness_values):.1f} ≤ S ≤ {max(slenderness_values):.1f} "
        f"(threshold = {SLENDERNESS_THRESHOLD:.0f})"
    )
    print(
        f"  Shared k/G uncertainty sweep: "
        f"{relative_uncertainties[0]*100:.0f}% → {relative_uncertainties[-1]*100:.0f}% of nominal"
    )
    print(f"Total DOE scenarios: {len(records)}")
    for model_name, count in selection_counts.items():
        print(f"  {model_name:16s}: {count} selections")

    print_sample_metrics(baseline_record, "Baseline scenario")
    worst_te = max(records, key=lambda record: record.timoshenko.cost)
    print_sample_metrics(worst_te, "High-uncertainty scenario")
    plot_selection_matrix(records, length_values, relative_uncertainties)


if __name__ == "__main__":
    main()

