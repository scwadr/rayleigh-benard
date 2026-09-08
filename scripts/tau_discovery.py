"""Weak-form SGS discovery for COnvection-FILTER-Copy2 (run on PACE).

See papers/2D_Incompressible_SGS_Paper_2 (44).pdf, sections 2.2 and 3.1.
No simulation dependencies are imported until a discovery stage is requested.
"""

import json
from pathlib import Path

import numpy as np


def clean_library(lib, anchor=None, tolerance=1e-10):
    """Remove numerically zero/proportional predictors, preserving an RHS anchor.

    Compare dimensionless columns for zeros, then unit columns for duplicates.
    Keep all associated metadata aligned; report every removal for inspection.
    """
    q = np.asarray(lib.Q)
    weights = np.asarray(lib.col_weights, dtype=float)
    if not np.isfinite(q).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Nonfinite Q or invalid characteristic column scales")
    scaled = q / weights
    norms = np.linalg.norm(scaled, axis=0)
    cutoff = tolerance * max(norms.max(initial=0), np.finfo(float).tiny)
    order = list(range(q.shape[1]))
    if anchor is not None:
        if norms[anchor] <= cutoff:
            raise ValueError("Target is numerically zero; no closure can be identified")
        order.remove(anchor)
        order.insert(0, anchor)
    keep, removed, units = [], [], []
    for j in order:
        if norms[j] <= cutoff:
            removed.append((str(lib.terms[j]), "zero"))
            continue
        unit = scaled[:, j] / norms[j]
        # A predictor equal to the measured target is a valid exact closure,
        # not a redundant predictor to discard.
        duplicate = next((k for k, old in zip(keep, units) if k != anchor
                          if min(np.linalg.norm(unit - old),
                                 np.linalg.norm(unit + old)) <= tolerance), None)
        if duplicate is not None:
            removed.append((str(lib.terms[j]), "proportional to " + str(lib.terms[duplicate])))
        else:
            keep.append(j)
            units.append(unit)
    keep.sort()
    lib.Q = q[:, keep]
    lib.col_weights = weights[keep]
    lib.terms = [lib.terms[j] for j in keep]
    return keep, removed, (keep.index(anchor) if anchor is not None else None)


def spatial_derivative(field, spacing, axis):
    """Periodic centered x derivative; second-order one-sided y endpoints."""
    if axis == 0:
        return (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2 * spacing)
    return np.gradient(field, spacing, axis=axis, edge_order=2)


def gradient_stresses(u, dx, dy, delta):
    """Paper (3.1)-(3.4), full tensors, evaluated one time frame at a time.

    The two mixed Hessian contractions both contribute, hence the factor two.
    Ignore at least two y cells in subsequent fitting (boundary stencils).
    """
    tau2 = np.empty(u.shape[:-1] + (2, 2), dtype=float)
    tau4 = np.empty_like(tau2)
    for it in range(u.shape[2]):
        frame = u[:, :, it, :]
        ux = spatial_derivative(frame, dx, 0)
        uy = spatial_derivative(frame, dy, 1)
        uxx = spatial_derivative(ux, dx, 0)
        uxy = spatial_derivative(ux, dy, 1)
        uyy = spatial_derivative(uy, dy, 1)
        outer = lambda a: np.einsum("...i,...j->...ij", a, a)
        tau2[:, :, it] = delta**2 / 12 * (outer(ux) + outer(uy))
        tau4[:, :, it] = delta**4 / 288 * (outer(uxx) + 2 * outer(uxy) + outer(uyy))
    return tau2, tau4


def _is_target(term, name, time_order):
    return (len(term.primes) == 1
            and term.primes[0].derivand.string == name
            and term.primes[0].derivative.xorder == 0
            and term.primes[0].derivative.torder == time_order)


def discover_stage(fields, ranks, target, dxs, *, evolution=False,
                   complexity=6, max_observables=2, domain_size=(64, 64, 20),
                   ndomains=1000, validation_domains=250, train_fraction=0.75,
                   max_terms=8, workers=1, seed=1, tolerance=1e-10):
    """Fit scalar trace and STF relations with a structurally selected anchor.

    Validation uses later times and disjoint space-time integration domains.
    Sparsities are selected using training residual jumps; validation is reporting
    only. For evolution, the anchor is d_t target and all other time derivatives
    are excluded. Algebraic predictors contain no target or target derivatives.
    """
    from SPIDER.continuous.library import Observable
    from SPIDER.continuous.process_library_terms import SRDataset
    from SPIDER.commons.z3base import FullRank, SymmetricTraceFree
    from SPIDER.commons.sparse_reg_bf import (
        Initializer, ModelIterator, Residual, Scaler, Threshold, sparse_reg_bf,
    )

    shape = fields[target].shape[:3]
    if not 0 < train_fraction < 1 or min(ndomains, validation_domains, workers) < 1:
        raise ValueError("Use a split strictly between zero and one and positive domain/worker counts")
    if len(domain_size) != 3 or min(domain_size) < 5 or max_terms < 2:
        raise ValueError("Use three domain widths >= 5 and max_terms >= 2 (including the target)")
    if any(a.shape[:3] != shape for a in fields.values()):
        raise ValueError("Every discovery field must share x, y, t coordinates")
    split = int(shape[2] * train_fraction)
    # Two y cells protect the explicit Hessians used in residual targets.
    if min(split, shape[2] - split) < domain_size[2] or any(
            n < width for n, width in zip((shape[0], shape[1] - 4), domain_size[:2])):
        raise ValueError("Insufficient data for disjoint train/validation domains; reduce domain_size")
    observables = [Observable(string=name, rank=rank, can_commute_indices=(rank == 2))
                   for name, rank in ranks.items()]
    irreps = (FullRank(rank=0), SymmetricTraceFree(rank=2))
    datasets = []
    for start, stop, count in ((0, split, ndomains), (split, shape[2], validation_domains)):
        block = {name: a[:, 2:-2, start:stop, ...] for name, a in fields.items()}
        ds = SRDataset(world_size=block[target].shape[:3], data_dict=block,
                       observables=observables, dxs=dxs, irreps=irreps)
        ds.make_libraries(max_complexity=complexity, max_observables=max_observables,
                          max_dx=2, max_dt=int(evolution))
        for lib in ds.libs.values():
            def allowed(term):
                if _is_target(term, target, int(evolution)):
                    return True
                if any(p.derivative.torder for p in term.primes):
                    return False
                return evolution or all(p.derivand.string != target for p in term.primes)
            lib.terms = [term for term in lib.terms if allowed(term)]
        np.random.seed(seed)
        ds.make_domains(ndomains=count, domain_size=list(domain_size), pad=0)
        ds.make_weights(m=16, qmax=0)
        ds.set_LT_scale(L=dxs[0] * (domain_size[0] - 1),
                        T=dxs[2] * (domain_size[2] - 1))
        ds.make_library_matrices(parallel=workers > 1, num_processors=workers)
        datasets.append(ds)
    train, validation = datasets
    reports = {}
    for irrep in irreps:
        lib, heldout = train.libs[irrep], validation.libs[irrep]
        if lib.terms != heldout.terms:
            raise ValueError("Training and validation library order differs")
        if not np.isfinite(heldout.Q).all():
            raise ValueError("Nonfinite validation matrix")
        anchors = [j for j, term in enumerate(lib.terms)
                   if _is_target(term, target, int(evolution))]
        if len(anchors) != 1:
            raise ValueError(f"Expected one target in {irrep}, found {anchors}")
        keep, removed, anchor = clean_library(lib, anchors[0], tolerance)
        heldout.Q = heldout.Q[:, keep]
        k = min(max_terms, len(keep))
        if k < 2 or lib.Q.shape[0] <= len(keep):
            raise ValueError("Need more domains than columns and at least one predictor")
        result = sparse_reg_bf(
            lib.Q, scaler=Scaler(sub_inds=None, char_sizes=lib.col_weights,
                                 unit_rows=False, train_fraction=1),
            initializer=Initializer(method="power", start_k=k),
            model_iterator=ModelIterator(max_k=k, backward_forward=True, max_passes=3),
            residual=Residual(residual_type="matrix_relative"),
            threshold=Threshold(threshold_type="jump", gamma=1.5, delta=1e-8),
            inhomog=True, inhomog_col=anchor, seed=seed,
        )
        xi = np.asarray(result.xi)
        if not np.isfinite(xi).all() or xi[anchor] == 0:
            raise ValueError("Regression returned an invalid or unanchored model")
        xi = -xi / xi[anchor]  # physical relation: -target + sum(c_j F_j) = 0
        def error(q):
            norm = np.linalg.norm(q[:, anchor])
            return float(np.linalg.norm(q @ xi) / norm) if norm > 0 else None
        report = dict(terms=[str(term) for term in lib.terms], coefficients=xi.tolist(),
                      anchor=anchor, train_relative_error=error(lib.Q),
                      validation_relative_error=error(heldout.Q), removed=removed)
        reports[str(irrep)] = report
        print(irrep, "target:", lib.terms[anchor])
        for j in np.flatnonzero(xi):
            print(f"  {xi[j]:+.8g} {lib.terms[j]}")
        print("target-relative errors (train, validation):",
              report["train_relative_error"], report["validation_relative_error"])
    return reports


def save_report(path, reports, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(metadata=metadata, stages=reports), indent=2,
                               ensure_ascii=False, allow_nan=False), encoding="utf-8")
