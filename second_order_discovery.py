"""Residual SGS discovery helpers for the PACE convection notebook."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from SPIDER.commons.library import Observable
from SPIDER.continuous.process_library_terms import SRDataset


def bare_column(lib, name):
    columns = [i for i, term in enumerate(lib.terms)
               if len(term.primes) == 1
               and term.primes[0].derivand.string == name
               and term.primes[0].nderivs == 0]
    if len(columns) != 1:
        raise ValueError(f"Expected one bare {name} column, found {columns}")
    return columns[0]


def build_correction_dataset(first_order_srd, resolved_data, *,
                             num_processors, parallel=True):
    """Evaluate resolved predictors and baseline columns on the same rows.

    resolved_data must use exactly the crop of first_order_srd, not merely
    arrays with a coincidentally matching shape. No new domains are sampled.
    """
    if set(resolved_data) != {"u", "p", "b"}:
        raise ValueError("Expected resolved fields u, p, b")
    for name, values in resolved_data.items():
        shape = tuple(first_order_srd.world_size) + ((2,) if name == "u" else ())
        if values.shape != shape:
            raise ValueError(f"{name}: expected aligned shape {shape}, got {values.shape}")
    observables = [Observable(string="u", rank=1),
                   Observable(string="p", rank=0),
                   Observable(string="b", rank=0)]
    dataset = SRDataset(
        world_size=np.array(first_order_srd.world_size, copy=True),
        data_dict={**resolved_data, **first_order_srd.data_dict},
        observables=observables, dxs=list(first_order_srd.dxs),
        irreps=first_order_srd.irreps,
    )
    # Two observables plus four spatial derivatives admit Hessian products.
    dataset.make_libraries(max_complexity=6, max_observables=2,
                           max_dx=4, max_dt=0)
    for irrep, lib in dataset.libs.items():
        # Only append the undifferentiated measured and first-order fields.
        # They never participate in generated products or derivatives.
        source_terms = list(first_order_srd.libs[irrep].terms)
        if any(len(term.primes) != 1 or term.primes[0].nderivs != 0
               for term in source_terms):
            raise ValueError("Expected a two-field first-order dataset")
        lib.terms = list(lib.terms) + source_terms
    dataset.observables += list(first_order_srd.observables)
    dataset.domains = deepcopy(first_order_srd.domains)
    dataset.domain_size = list(first_order_srd.domain_size)
    dataset.pad = first_order_srd.pad
    dataset.weights, dataset.tensor_weight_basis = deepcopy(
        (first_order_srd.weights, first_order_srd.tensor_weight_basis))
    dataset.weight_dxs = list(first_order_srd.weight_dxs)
    dataset.set_LT_scale(first_order_srd.xscale, first_order_srd.tscale)
    dataset.make_library_matrices(parallel=parallel,
                                  num_processors=num_processors, debug=False)
    return dataset


def is_gradient_baseline(term, target_name):
    """Recognize the resolved equivalent of the explicitly evaluated G field."""
    primes = term.primes
    names = sorted(prime.derivand.string for prime in primes)
    expected = ["u", "u"] if target_name == "tau" else ["b", "u"]
    if names != expected or any(prime.derivative.torder != 0
                               or prime.derivative.xorder != 1 for prime in primes):
        return False
    left, right = (prime.derivative.x_derivatives[0] for prime in primes)
    observable_indices = [index for prime in primes for index in prime.derivand.indices]
    return left == right and left not in observable_indices


@dataclass
class ResidualProblem:
    Q: np.ndarray
    weights: np.ndarray
    terms: list
    measured: np.ndarray
    baseline: np.ndarray
    coefficient: float
    target_name: str
    gradient_name: str
    removed: list
    residual_is_zero: bool


def residual_problem(lib, target_name, gradient_name, coefficient, *,
                     include_first_order_terms=False, tolerance=1e-8):
    """Subtract a frozen baseline in physical units, then scale and clean.

    Column zero always represents R = target - coefficient * gradient.
    Its term object is retained only for tensor indexing; reports label it R.
    """
    coefficient = float(coefficient)
    if not np.isfinite(coefficient) or not 0 < tolerance < 1:
        raise ValueError("Invalid baseline coefficient or tolerance")
    target_col = bare_column(lib, target_name)
    gradient_col = bare_column(lib, gradient_name)
    raw = np.asarray(lib.Q, dtype=float)
    weights = np.asarray(lib.col_weights, dtype=float)
    if (not np.isfinite(raw).all() or not np.isfinite(weights).all()
            or np.any(weights <= 0)):
        raise ValueError("Nonfinite library or invalid characteristic sizes")
    measured = raw[:, target_col].copy()
    baseline = coefficient * raw[:, gradient_col]
    residual = measured - baseline
    residual_norm = np.linalg.norm(residual)
    reference_norm = max(np.linalg.norm(measured), np.linalg.norm(baseline))
    is_zero = residual_norm <= tolerance * max(reference_norm, np.finfo(float).tiny)
    # Keep the original target's physical units, but scale its magnitude to R.
    residual_weight = weights[target_col] * (
        residual_norm / max(np.linalg.norm(measured), np.finfo(float).tiny))
    residual_weight = max(residual_weight, np.finfo(float).tiny)
    selected = []
    removed = []
    for i, term in enumerate(lib.terms):
        if i == target_col:
            continue
        if i == gradient_col or is_gradient_baseline(term, target_name):
            if not include_first_order_terms:
                removed.append((str(term), "frozen first-order term"))
                continue
        selected.append(i)
    scaled = raw[:, selected] / weights[selected]
    norms = np.linalg.norm(scaled, axis=0)
    cutoff = tolerance * max(norms.max(initial=0), np.finfo(float).tiny)
    keep, units = [], []
    for i, norm, column in zip(selected, norms, scaled.T):
        if norm <= cutoff:
            removed.append((str(lib.terms[i]), "zero"))
            continue
        unit = column / norm
        if any(min(np.linalg.norm(unit - old), np.linalg.norm(unit + old))
               <= tolerance for old in units):
            removed.append((str(lib.terms[i]), "duplicate"))
            continue
        keep.append(i)
        units.append(unit)
    return ResidualProblem(
        Q=np.column_stack([residual, raw[:, keep]]),
        weights=np.concatenate([[residual_weight], weights[keep]]),
        terms=[lib.terms[target_col]] + [lib.terms[i] for i in keep],
        measured=measured, baseline=baseline, coefficient=coefficient,
        target_name=target_name, gradient_name=gradient_name,
        removed=removed, residual_is_zero=is_zero,
    )
