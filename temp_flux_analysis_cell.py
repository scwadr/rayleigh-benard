# %% Replacement flux analysis: paste ALL of this file into ONE new notebook cell.
# Exported from COnvection-FILTER-Copy2.ipynb; that notebook is the canonical source.
# Interrupt the old flux cell first. Keep the kernel and its fitted models alive.
# Requires completed data preparation, discovery, and discovery archiving (through
# notebook cell 28, counting Markdown). No setup or regression is repeated here.
# This cell calculates, archives to the existing analysis_run_dir, and plots.
# Alternative from the SAME live kernel: %run -i temp_flux_analysis_cell.py

_flux_required = ('run_controls', 'u_bar', 'tau', 'phi', 'b_filt', 'p_filt',
                  'RUN_BASELINE_RESIDUALS', 'archive_directory', 'restored_archive',
                  'analysis_run_dir')
_flux_missing = [name for name in _flux_required if name not in globals()]
if _flux_missing:
    raise RuntimeError('This cell needs the existing discovery kernel; missing: ' + ', '.join(_flux_missing))

# Copy execution controls so the original discovery metadata stays unchanged.
run_controls = dict(run_controls)
run_controls.update(flux_tile_size=64, flux_batch_size=8, flux_cache_mib=256)
# Optional: run_controls['flux_workers'] = 8  # must fit the allocated CPU count

# Figure 2 benchmarks: parallel spatial tiles with reusable derivatives and terms.
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from threading import local
from time import perf_counter
import os
import inspect
import linecache

# Capture this executed cell, including notebook edits not yet saved to disk.
flux_executed_source = ''.join(linecache.getlines(inspect.currentframe().f_code.co_filename))
from itertools import product
from findiff import FinDiff
from PySPIDER.commons.z3base import LiteralIndex, highest_index

flux_batch_size = int(run_controls.get('flux_batch_size', 8))
flux_cache_mib = int(run_controls.get('flux_cache_mib', 256))
flux_include_general = bool(run_controls.get('flux_include_general', False))
flux_tile_size = int(run_controls.get('flux_tile_size', 64))
flux_available_workers = int(run_controls.get('workers') or os.environ.get('SLURM_CPUS_PER_TASK')
    or (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 1))
flux_workers = int(run_controls.get('flux_workers') or min(8, flux_available_workers))
flux_margin = 8  # common interior beyond sixth-order finite-difference stencils
flux_mean_ratio_rtol = 1e-10
if min(flux_batch_size, flux_cache_mib, flux_tile_size, flux_workers) < 1:
    raise ValueError('Flux batch size, tile size, worker count and cache budget must be positive')
if flux_workers > flux_available_workers:
    raise ValueError('flux_workers exceeds the configured/allocated CPU count')

# Eq. (3.8) integrates moments before taking the ratio, not per-frame accuracies.
flux_times = np.asarray(t, dtype=float)
if flux_times.ndim != 1 or len(flux_times) < 2 or not np.isfinite(flux_times).all() or np.any(np.diff(flux_times) <= 0):
    raise ValueError('Paper averaging requires at least two finite, increasing times')
flux_time_weights = np.empty(len(flux_times), dtype=float)
flux_time_weights[0] = (flux_times[1]-flux_times[0])/2
flux_time_weights[-1] = (flux_times[-1]-flux_times[-2])/2
flux_time_weights[1:-1] = (flux_times[2:]-flux_times[:-2])/2
flux_time_weights /= flux_time_weights.sum()

# Keep the established common crop so discovered coefficients retain their meaning.
flux_fit_slice = (slice(pad_x, -pad_x), slice(pad_y, -pad_y)) if RUN_BASELINE_RESIDUALS else (slice(None), slice(None))
fx, fy = np.asarray(x)[flux_fit_slice[0]], np.asarray(y)[flux_fit_slice[1]]
flux_resolved = {'u': u_bar[flux_fit_slice], 'b': b_filt[flux_fit_slice], 'p': p_filt[flux_fit_slice]}
flux_measured = {'tau': tau[flux_fit_slice], 'phi': phi[flux_fit_slice]}
if min(len(fx), len(fy)) <= 2*flux_margin:
    raise ValueError('Not enough common interior for flux derivatives')
for coordinates, spacing in ((fx, dx), (fy, dy)):
    if not np.allclose(np.diff(coordinates), spacing):
        raise ValueError('Pointwise derivatives require uniform spatial coordinates')
for name, field in {**flux_resolved, **flux_measured}.items():
    if field.shape[:3] != (len(fx), len(fy), len(flux_times)):
        raise ValueError(f'Flux field/coordinate mismatch: {name}')
cut = (slice(flux_margin, -flux_margin),)*2

def paper_accuracy(moments):
    aa, bb, ab = moments
    return float(np.clip(ab/max(aa, bb), -1., 1.)) if max(aa, bb) > 0 else None

def flux_inner(a, b, weights):
    """Component contraction, uniform space mean, global time weights."""
    if a.shape != b.shape:
        raise ValueError('Flux moments require aligned fields')
    # Nonfinite inputs propagate into moments and are rejected once per tile.
    # Avoid rescanning both arrays before every contraction and every tau pairing.
    shape = (*a.shape[:3], -1)
    return float(np.einsum('xyti,xyti,t->', a.reshape(shape), b.reshape(shape), weights,
                           optimize=False)/(a.shape[0]*a.shape[1]))

def flux_mean(a, weights):
    return float(np.dot(a.mean(axis=(0, 1)), weights))

def flux_project(value, rank):
    # A scalar piece stores tr(tau), whose full-tensor norm is trace^2 / 2.
    if rank == 2:
        symmetric = (value + value.swapaxes(-1, -2))/2
        return symmetric - np.trace(symmetric, axis1=-2, axis2=-1)[..., None, None]*np.eye(2)/2
    return value

def flux_compile_term(term):
    """Compile Einstein contractions once; primes use canonical component/order keys."""
    if any(prime.derivative.torder for prime in term.primes):
        raise ValueError('Flux closure terms must be spatial-only')
    count = highest_index(term.all_indices()) + 1
    plan = []
    for assignment in product(range(2), repeat=count):
        literal = term.map_all_indices(lambda idx: LiteralIndex(assignment[idx.value])) if count else term
        primes = []
        for prime in literal.primes:
            orders = prime.derivative.get_spatial_orders()
            primes.append((prime.derivand.string, tuple(idx.value for idx in prime.derivand.indices),
                           tuple(orders[LiteralIndex(i)] for i in range(2))))
        plan.append((assignment[:term.rank], primes))
    return plan

# Each thread owns its finite-difference operators; input fields are read-only.
flux_worker_state = local()

class FluxTileEvaluator:
    """Cache derivatives AND contracted terms on a small spatial tile plus halo."""
    def __init__(self, start, stop, x0, x1, y0, y1):
        region = (slice(x0-flux_margin, x1+flux_margin),
                  slice(y0-flux_margin, y1+flux_margin), slice(start, stop))
        self.fields = {name: np.asarray(value[region], dtype=float)
                       for name, value in flux_resolved.items()}
        if RUN_BASELINE_RESIDUALS:
            for target, (_, name) in first_order_sources.items():
                self.fields[name] = np.asarray(second_order_datasets[target].data_dict[name][region], dtype=float)
        self.shape = self.fields['b'].shape
        self.cache, self.cache_bytes = OrderedDict(), 0
        self.limit = flux_cache_mib*1024**2
        self.peak_bytes = self.derivative_evaluations = self.prime_hits = self.term_hits = self.evictions = 0
        if not hasattr(flux_worker_state, 'operators'):
            flux_worker_state.operators = {}
        self.operators = flux_worker_state.operators

    def remember(self, key, value):
        # Values stored here own their data; nbytes accounts for retained memory.
        while self.cache and self.cache_bytes + value.nbytes > self.limit:
            _, old = self.cache.popitem(last=False)
            self.cache_bytes -= old.nbytes
            self.evictions += 1
        if value.nbytes <= self.limit:
            self.cache[key] = value
            self.cache_bytes += value.nbytes
            self.peak_bytes = max(self.peak_bytes, self.cache_bytes)
        return value

    def prime(self, key):
        cache_key = ('prime', key)
        if cache_key in self.cache:
            self.prime_hits += 1
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]
        name, components, orders = key
        field = self.fields[name][(slice(None),)*3 + components]
        if not any(orders):
            return field
        if orders not in self.operators:
            self.operators[orders] = FinDiff(
                *((axis, spacing, order) for axis, (spacing, order) in enumerate(zip((dx, dy), orders)) if order), acc=6)
        self.derivative_evaluations += 1
        return self.remember(cache_key, self.operators[orders](field))

    def term(self, term):
        cache_key = ('term', term)
        if cache_key in self.cache:
            self.term_hits += 1
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]
        result = np.zeros(tuple(n-2*flux_margin for n in self.shape[:2]) + (self.shape[2],) + (2,)*term.rank)
        # Form products on the retained interior, not on the discarded halo.
        for components, primes in flux_term_plans[term]:
            value = np.ones(result.shape[:3])
            for key in primes:
                value *= self.prime(key)[cut]
            result[(slice(None),)*3 + components] += value
        return self.remember(cache_key, result)

# Preserve every returned residual candidate, with its original archive ID.
flux_model_groups = {}
if flux_include_general:
    for target, entries in (('tau', tau_regression_results), ('phi', phi_regression_results)):
        for (irrep, count), entry in entries.items():
            key = (target, irrep, 'general')
            flux_model_groups.setdefault(key, {})[count] = dict(rank=entry['equation'].rank,
                gradient_name=None, baseline_coefficient=0., predictor_terms=entry['predictor_terms'],
                predictor_coefficients=entry['predictor_coefficients'])
if RUN_BASELINE_RESIDUALS:
    for key, entry in second_order_results.items():
        problem = entry['problem']
        flux_model_groups[key] = {index: dict(rank=problem.terms[0].rank,
            gradient_name=problem.gradient_name, baseline_coefficient=problem.coefficient,
            predictor_terms=problem.terms[1:], predictor_coefficients=fit['coefficients'])
            for index, fit in enumerate(entry['fits'])}

flux_term_plans = {}
for entries in flux_model_groups.values():
    for entry in entries.values():
        if len(entry['predictor_terms']) != len(entry['predictor_coefficients']):
            raise ValueError('Misaligned flux terms and coefficients')
        for term, coefficient in zip(entry['predictor_terms'], entry['predictor_coefficients']):
            if not np.isfinite(coefficient):
                raise ValueError('Nonfinite model coefficient')
            if coefficient != 0 and term not in flux_term_plans:
                flux_term_plans[term] = flux_compile_term(term)

for plan in flux_term_plans.values():
    for _, primes in plan:
        for _, _, orders in primes:
            # Sixth-order centered stencil radius from findiff's coefficient rule.
            if any(order and ((order+1)//2 + 2) > flux_margin for order in orders):
                raise ValueError('Flux term requires a wider spatial halo')

# Compile sparse coefficient use once, not once per spatial tile. Libraries can
# have thousands of inactive columns, which should never enter the tile hot path.
flux_group_plans = []
for key, entries in flux_model_groups.items():
    active_terms, baselines = {}, []
    for index, entry in entries.items():
        if entry['gradient_name'] is not None:
            baselines.append((index, entry['baseline_coefficient']))
        for term, coefficient in zip(entry['predictor_terms'], entry['predictor_coefficients']):
            if coefficient != 0:
                active_terms.setdefault(term, []).append((index, coefficient))
    flux_group_plans.append(dict(key=key, rank=next(iter(entries.values()))['rank'],
        indices=tuple(entries), baselines=tuple(baselines),
        active_terms=tuple((term, tuple(uses)) for term, uses in active_terms.items())))

flux_specs = []
for mode in sorted({key[2] for key in flux_model_groups}):
    groups = {}
    for key, entries in flux_model_groups.items():
        if key[2] == mode:
            groups[key[0], next(iter(entries.values()))['rank']] = [(key, index) for index in entries]
    if not {('tau', 0), ('tau', 2), ('phi', 1)} <= groups.keys():
        raise ValueError(f'Incomplete tau/phi candidate groups for {mode}')
    flux_specs += [dict(target='tau', family=mode, pieces=list(pair))
                   for pair in product(groups.get(('tau', 0), []), groups.get(('tau', 2), []))]
    flux_specs += [dict(target='phi', family=mode, pieces=[piece]) for piece in groups.get(('phi', 1), [])]
for target in ('tau', 'phi'):
    for order in (1, 2):
        flux_specs.append(dict(target=target, family='prescribed', order=order, pieces=[]))

# Columns: closure reference norm, model norm, cross moment;
#          flux reference norm, model norm, cross moment; mean reference, model Pi.
def flux_compute_tile(task):
    start, stop, x0, x1, y0, y1 = task
    weights = flux_time_weights[start:stop]
    evaluator = FluxTileEvaluator(*task)
    tile_x = fx[x0-flux_margin:x1+flux_margin]
    tile_y = fy[y0-flux_margin:y1+flux_margin]
    moments = np.zeros((len(flux_specs), 8))
    # These low-order gradients exactly retain the original gradient-baseline convention.
    velocity, scalar = evaluator.fields['u'], evaluator.fields['b']
    D = np.stack(np.gradient(velocity, tile_x, tile_y, axis=(0, 1), edge_order=2), axis=-2)[cut]
    strain = (D + D.swapaxes(-1, -2))/2
    grad_b = np.stack(np.gradient(scalar, tile_x, tile_y, axis=(0, 1), edge_order=2), axis=-1)[cut]
    measured = {target: np.asarray(field[x0:x1, y0:y1, start:stop], dtype=float)
                for target, field in flux_measured.items()}
    references = {'tau': -np.einsum('...ij,...ij->...', strain, measured['tau']),
                  'phi': -np.einsum('...i,...i->...', grad_b, measured['phi'])}
    reference_stats = {target: (flux_inner(measured[target], measured[target], weights),
        flux_inner(values, values, weights), flux_mean(values, weights)) for target, values in references.items()}
    gradients = {'tau': np.einsum('...ki,...kj->...ij', D, D),
                 'phi': np.einsum('...ki,...k->...i', D, grad_b)}
    piece_stats, piece_fluxes = {}, {}

    # Reuse compiled sparse plans, derivatives and terms across every family.
    # Only scalar flux fields survive a group; full candidate tensors are released.
    for group in flux_group_plans:
        key, rank = group['key'], group['rank']
        target = key[0]
        values = {index: np.zeros(grad_b.shape[:3] + (2,)*rank) for index in group['indices']}
        for index, coefficient in group['baselines']:
            baseline = gradients[target]
            if rank == 0:
                baseline = np.trace(baseline, axis1=-2, axis2=-1)
            values[index] += coefficient*baseline
        for term, uses in group['active_terms']:
            field = evaluator.term(term)
            for index, coefficient in uses:
                values[index] += coefficient*field
        for index in group['indices']:
            value = flux_project(values.pop(index), rank)
            if rank == 0:
                closure_bb = flux_inner(value, value, weights)/2
                closure_ab = flux_inner(np.trace(measured[target], axis1=-2, axis2=-1), value, weights)/2
                predicted = -np.trace(strain, axis1=-2, axis2=-1)*value/2
            else:
                closure_bb = flux_inner(value, value, weights)
                closure_ab = flux_inner(measured[target], value, weights)
                predicted = (-np.einsum('...ij,...ij->...', strain, value) if target == 'tau'
                             else -np.einsum('...i,...i->...', grad_b, value))
            piece = (key, index)
            piece_stats[piece] = np.array([closure_bb, closure_ab,
                flux_inner(predicted, predicted, weights), flux_inner(references[target], predicted, weights),
                flux_mean(predicted, weights)])
            piece_fluxes[piece] = predicted

    for index, spec in enumerate(flux_specs):
        if not spec['pieces']:
            continue
        target, pieces = spec['target'], spec['pieces']
        cbb, cab, pbb, pab, pm = sum(piece_stats[piece] for piece in pieces)
        # Trace and STF are orthogonal for closure norms, but their fluxes are not.
        if len(pieces) == 2:
            pbb += 2*flux_inner(piece_fluxes[pieces[0]], piece_fluxes[pieces[1]], weights)
        caa, paa, pr = reference_stats[target]
        moments[index] += [caa, max(0., cbb), cab, paa, max(0., pbb), pab, pr, pm]
    del piece_fluxes, piece_stats

    # Prescribed NGM2/NGM4 stress and the analogous Gaussian scalar-flux expansion.
    # Mixed Hessian derivatives appear twice in the contraction over k,m.
    hessians = {target: np.zeros_like(gradients[target]) for target in ('tau', 'phi')}
    for orders, multiplicity in (((2, 0), 1), ((1, 1), 2), ((0, 2), 1)):
        hu = np.stack([evaluator.prime(('u', (i,), orders))[cut] for i in range(2)], axis=-1)
        hb = evaluator.prime(('b', (), orders))[cut]
        hessians['tau'] += multiplicity*np.einsum('...i,...j->...ij', hu, hu)
        hessians['phi'] += multiplicity*hu*hb[..., None]
    for index, spec in enumerate(flux_specs):
        if spec['family'] != 'prescribed':
            continue
        target = spec['target']
        model = (delta**2/12)*gradients[target]
        if spec['order'] == 2:
            model += (delta**4/288)*hessians[target]
        predicted = (-np.einsum('...ij,...ij->...', strain, model) if target == 'tau'
                     else -np.einsum('...i,...i->...', grad_b, model))
        caa, paa, pr = reference_stats[target]
        moments[index] += [caa, flux_inner(model, model, weights), flux_inner(measured[target], model, weights),
            paa, flux_inner(predicted, predicted, weights), flux_inner(references[target], predicted, weights),
            pr, flux_mean(predicted, weights)]
    if not np.isfinite(moments).all():
        raise ValueError(f'Nonfinite flux moments in tile {task}')
    # Local uniform means become one global uniform mean. Every retained point
    # belongs to exactly one tile, including narrow final tiles along each axis.
    spatial_weight = ((x1-x0)*(y1-y0))/((len(fx)-2*flux_margin)*(len(fy)-2*flux_margin))
    return (moments*spatial_weight, evaluator.peak_bytes, evaluator.derivative_evaluations,
            evaluator.prime_hits, evaluator.term_hits, evaluator.evictions)

def flux_tile_results(pool, tasks):
    """Bound queued work and completed results; cancel queued tiles on interruption."""
    if pool is None:
        for task in tasks:
            yield flux_compute_tile(task)
        return
    tasks = iter(tasks)
    pending = deque(pool.submit(flux_compute_tile, task) for task in islice(tasks, 2*flux_workers))
    try:
        while pending:
            # Consume in submission order for deterministic accumulation.
            future = pending.popleft()
            yield future.result()
            task = next(tasks, None)
            if task is not None:
                pending.append(pool.submit(flux_compute_tile, task))
    finally:
        for future in pending:
            future.cancel()

flux_moments = np.zeros((len(flux_specs), 8))
flux_cache_peak_bytes = 0
flux_profile = []
flux_spatial_tiles = [(x0, min(x0+flux_tile_size, len(fx)-flux_margin),
                       y0, min(y0+flux_tile_size, len(fy)-flux_margin))
    for x0 in range(flux_margin, len(fx)-flux_margin, flux_tile_size)
    for y0 in range(flux_margin, len(fy)-flux_margin, flux_tile_size)]
print(f'Flux evaluation: {len(flux_spatial_tiles)} spatial tiles per time batch, '
      f'{flux_workers} threads, tile width {flux_tile_size}; all points and times retained.', flush=True)
flux_started = perf_counter()
pool = ThreadPoolExecutor(max_workers=flux_workers) if flux_workers > 1 else None
try:
    for start in range(0, len(flux_times), flux_batch_size):
        stop = min(start+flux_batch_size, len(flux_times))
        batch_started = last_progress = perf_counter()
        counters = np.zeros(4, dtype=np.int64)
        batch_peak = 0
        tasks = ((start, stop, *tile) for tile in flux_spatial_tiles)
        for completed, result in enumerate(flux_tile_results(pool, tasks), 1):
            tile_moments, peak, evaluations, prime_hits, term_hits, evictions = result
            flux_moments += tile_moments
            batch_peak = max(batch_peak, peak)
            counters += (evaluations, prime_hits, term_hits, evictions)
            if perf_counter()-last_progress >= 30:
                print(f'Flux tiles: times {start+1}-{stop}, {completed}/{len(flux_spatial_tiles)} tiles, '
                      f'{perf_counter()-batch_started:.1f}s elapsed', flush=True)
                last_progress = perf_counter()
        seconds = perf_counter()-batch_started
        flux_profile.append(dict(time_start=start, time_stop=stop, tiles=len(flux_spatial_tiles),
            seconds=seconds, cache_peak_bytes_per_tile=batch_peak, derivative_evaluations=int(counters[0]),
            derivative_cache_hits=int(counters[1]), term_cache_hits=int(counters[2]), cache_evictions=int(counters[3])))
        flux_cache_peak_bytes = max(flux_cache_peak_bytes, batch_peak)
        print(f'Flux moments: {stop}/{len(flux_times)} times, batch {seconds:.1f}s; '
              f'cache hits {counters[1]}, term hits {counters[2]}, evictions {counters[3]}', flush=True)
finally:
    if pool is not None:
        pool.shutdown(wait=True, cancel_futures=True)
flux_elapsed_seconds = perf_counter()-flux_started

flux_rows = []
for index, spec in enumerate(flux_specs):
    caa, cbb, cab, paa, pbb, pab, mean_ref, mean_model = flux_moments[index]
    if not np.isfinite(flux_moments[index]).all():
        raise ValueError('Nonfinite integrated flux moments')
    pieces = spec['pieces']
    budgets = [candidate for key, candidate in pieces]
    if spec['family'] == 'prescribed':
        order = spec['order']
        label = 'Prescribed gradient' if order == 1 else 'Prescribed gradient + Hessian'
        model_ids = [f"{spec['target']}|prescribed|order={order}"]
    else:
        order = None if spec['family'] == 'general' else (1 if all(b == 0 for b in budgets) else 2)
        label = spec['family'].capitalize() + (' gradient' if order == 1 else ' residual-corrected' if order == 2 else ' general')
        label += ' [' + ','.join(map(str, budgets)) + ']'
        model_ids = [f'{key[0]}|{key[1]}|{key[2]}|candidate={candidate}' for key, candidate in pieces]
    ratio_defined = paa > 0 and abs(mean_ref) > flux_mean_ratio_rtol*np.sqrt(paa)
    flux_rows.append(dict(id=str(index), target=spec['target'], family=spec['family'],
        model_order=order, label=label, model_ids=model_ids,
        default_plot=bool(not budgets or len(set(budgets)) == 1),
        closure_accuracy_A=paper_accuracy((caa, cbb, cab)), flux_accuracy_A=paper_accuracy((paa, pbb, pab)),
        time_mean_flux_reference=float(mean_ref), time_mean_flux_model=float(mean_model),
        mean_flux_ratio=float(mean_model/mean_ref) if ratio_defined else None,
        mean_flux_ratio_defined=bool(ratio_defined),
        closure_relative_l2_error=float(np.sqrt(max(0., caa+cbb-2*cab)/caa)) if caa > 0 else None,
        flux_relative_l2_error=float(np.sqrt(max(0., paa+pbb-2*pab)/paa)) if paa > 0 else None))
print(f'Flux benchmarks complete: {len(flux_rows)} model combinations; all {len(flux_times)} times included.', flush=True)


import csv
flux_created = datetime.now(timezone.utc)
flux_dir = analysis_run_dir / ('flux_'+flux_created.strftime('%Y%m%dT%H%M%S.%fZ')+'_'+uuid4().hex[:8])
flux_dir.mkdir(parents=True, exist_ok=False)
flux_report = dict(schema_version=4, created_utc=flux_created.isoformat(),
    workflow='batched Figure 2 closure and flux benchmarks', validation_status='a priori diagnostics on discovery data',
    discovery_archive=str(archive_directory), discovery_bundle=restored_archive,
    prescribed_coefficients=dict(gradient=float(delta**2/12), hessian=float(delta**4/288)),
    definitions=dict(Pi_tau='-sum_ij S_ij tau_ij', S_ij='(d_i ubar_j+d_j ubar_i)/2', Pi_phi='-sum_i phi_i d_i bbar',
        A='integral <reference:modeled> dt / max(integral <reference:reference> dt, integral <modeled:modeled> dt)',
        A_source='local SGS paper equations (3.8)-(3.9), pages 8-9; Figure 2',
        mean_flux_ratio='time-integrated spatial mean modeled Pi / reference Pi',
        prescribed_tau_first='(delta^2/12) (d_k u_i)(d_k u_j)',
        prescribed_tau_second='tau_first + (delta^4/288) (d_k d_m u_i)(d_k d_m u_j)',
        prescribed_phi_first='(delta^2/12) (d_k u_i)(d_k b)',
        prescribed_phi_second='phi_first + (delta^4/288) (d_k d_m u_i)(d_k d_m b)',
        prescribed_source='stress: paper (3.1)-(3.4); phi: analogous Gaussian expansion, not a claim from this stress-only paper',
        discovered_second='frozen fitted/theoretical gradient plus discovered residual; not assumed to equal prescribed Hessian'),
    settings=dict(spatial_margin=flux_margin, batch_size=flux_batch_size, derivative_cache_mib=flux_cache_mib,
        derivative_cache_peak_bytes=flux_cache_peak_bytes, include_general=flux_include_general,
        time_count=len(flux_times), temporal_subsampling=False,
        mean_ratio_rtol=flux_mean_ratio_rtol,
        accuracy_averaging='uniform spatial means, component contraction, trapezoidal time integral',
        gradients='numpy.gradient edge_order=2, matching fitted baseline construction',
        hessians_and_library_derivatives='findiff acc=6; spatial-only on common crop, interior margin 8',
        crop='original baseline fit crop shared by all models, then derivative margin' if RUN_BASELINE_RESIDUALS else 'shared-library grid then derivative margin',
        fit_crop_points=[int(pad_x), int(pad_y)] if RUN_BASELINE_RESIDUALS else [0, 0],
        tau_assembly='all trace/STF pairs within each family via orthogonal closure moments and flux cross moments',
        default_curves='prescribed models, uncorrected baselines, and matched-budget residual candidates; no error-based selection',
        scalar_convention='b is simulated buoyancy/temperature-like scalar; no dimensional conversion',
        zero_conventions='A null for two zero fields; mean ratio null for near-zero reference mean'),
    execution=dict(method='parallel spatial tiles, exact full-grid moments',
        workers=flux_workers, tile_size=flux_tile_size, tile_halo=flux_margin,
        cache_scope='per active tile; shared derivative and contracted-term budget',
        elapsed_seconds=flux_elapsed_seconds, batches=flux_profile),
    flux_source=dict(git_commit=source_git('rev-parse', 'HEAD'), git_status=source_git('status', '--porcelain'),
        executed_cell_file='executed_flux_cell.py' if flux_executed_source else None,
        executed_cell_sha256=hashlib.sha256(flux_executed_source.encode('utf-8')).hexdigest() if flux_executed_source else None,
        note='Discovery source snapshots describe the earlier fits; executed_flux_cell.py describes this flux evaluation.'),
    results=flux_rows, executed_helpers={})
if flux_executed_source:
    (flux_dir / 'executed_flux_cell.py').write_text(flux_executed_source, encoding='utf-8')
for name in ('paper_accuracy', 'flux_inner', 'flux_mean', 'flux_project', 'flux_compile_term', 'FluxTileEvaluator', 'flux_compute_tile', 'flux_tile_results'):
    try:
        flux_report['executed_helpers'][name] = inspect.getsource(globals()[name])
    except (OSError, TypeError):
        flux_report['executed_helpers'][name] = None
for source_file in archive_directory.glob('source_*'):
    (flux_dir / source_file.name).write_bytes(source_file.read_bytes())
np.savez_compressed(flux_dir / 'flux_moments.npz', t=flux_times, time_weights=flux_time_weights,
    x=fx[flux_margin:-flux_margin], y=fy[flux_margin:-flux_margin], moments=flux_moments,
    candidate_ids=np.asarray([row['id'] for row in flux_rows]),
    columns=np.asarray(['closure_ref_norm2','closure_model_norm2','closure_cross',
                        'flux_ref_norm2','flux_model_norm2','flux_cross','mean_ref_flux','mean_model_flux']))
(flux_dir / 'flux.json').write_text(json.dumps(restored_json_value(flux_report),
    indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
with (flux_dir / 'flux.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(flux_rows[0]))
    writer.writeheader()
    writer.writerows(flux_rows)
(flux_dir / 'COMPLETE').write_text(flux_created.isoformat(), encoding='utf-8')
print('Flux archive:', flux_dir.resolve())


if 'flux_rows' in globals():
    # At one scale, show the same three metrics as Figure 2, without time-series plots.
    import matplotlib.pyplot as plt
    for target in ('tau', 'phi'):
        rows = [row for row in flux_rows if row['target'] == target and row['default_plot']]
        fig, axes = plt.subplots(3, 1, figsize=(max(8, .45*len(rows)), 10), sharex=True, constrained_layout=True)
        for axis, metric, label in zip(axes,
            ('closure_accuracy_A', 'flux_accuracy_A', 'mean_flux_ratio'),
            (f'A({target})', f'A(Pi_{target})', 'mean modeled Pi / mean reference Pi')):
            axis.plot(range(len(rows)), [np.nan if row[metric] is None else row[metric] for row in rows], 'o')
            axis.axhline(1, color='gray', ls='--', lw=.7)
            axis.set(ylabel=label)
        axes[-1].set_xticks(range(len(rows)), [row['label'] for row in rows], rotation=65, ha='right', fontsize=7)
        fig.suptitle(f'{target}: Delta/dx = {delta/dx:g}; time-averaged a priori benchmarks')
        fig.savefig(flux_dir / f'paper_accuracy_{target}.png', dpi=180)
        plt.show()
        plt.close(fig)
