"""Execute notebook SGS discovery, coefficient checks and flux analysis on PACE.

Scientific implementation and run archiving live in the notebook. This launcher
only selects cells, supplies configuration, and records execution/failure status.
Broad PDE recovery and legacy numerical-target experiments are notebook-only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import linecache
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from uuid import uuid4

SCRATCH_ROOT = Path('/storage/scratch1/2/ashelestov3')
DEFAULT_OUTPUT_ROOT = SCRATCH_ROOT / 'rayleigh-benard' / 'filter_sweeps'


def scratch_path(value):
    """Keep run data, archives and comparison output off the home filesystem."""
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(SCRATCH_ROOT.resolve()):
        raise ValueError(f'Use a path under {SCRATCH_ROOT}, not {path}; clear stale DATA_PATH/OUTPUT_ROOT overrides')
    return path


def scratch_runtime(root):
    """Redirect Python, plotting, JAX and temporary-file caches to scratch."""
    runtime = root / 'runtime'
    for variable, name in (('MPLCONFIGDIR', 'matplotlib'), ('XDG_CACHE_HOME', 'cache'),
                           ('JAX_COMPILATION_CACHE_DIR', 'jax'), ('TMPDIR', 'tmp'),
                           ('PYTHONPYCACHEPREFIX', 'pycache')):
        directory = runtime / name
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
    sys.pycache_prefix = os.environ['PYTHONPYCACHEPREFIX']
    tempfile.tempdir = os.environ['TMPDIR']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--delta-dx', type=float, help='Gaussian delta in raw x grid spacings')
    mode.add_argument('--compare-runs', help='Plot Figure 2 metrics versus width from completed batched flux archives under this root')
    mode.add_argument('--measure-li', action='store_true', help='Measure paper Eq. (3.3) on unfiltered velocity and archive a five-width plan; no filtering or regression')
    mode.add_argument('--scale-index', type=int, help='Run one of the five widths from --scale-plan (indices 0 through 4)')
    parser.add_argument('--scale-plan', help='Completed scale_plan.json from --measure-li; supplies widths, data/time identity and reference length')
    parser.add_argument('--li-wall-margin', type=int, default=50, help='Fixed number of raw y points excluded at each wall when measuring li (default: 50)')
    parser.add_argument('--max-delta-li', type=float, default=1., help='Upper normalized width for the five-point plan, capped by available filter-support interior')
    parser.add_argument('--reference-length', type=float, help='Plot Delta/L using this reference length; otherwise plot physical Delta')
    parser.add_argument('--include-corrections', action='store_true', help='Compatibility flag: matched correction budgets are now all plotted by default')
    parser.add_argument('--expected-scales', type=float, nargs='+', help='Comparison requires exactly these completed Delta/dx values')
    parser.add_argument('--correction-budgets', type=int, nargs='+', help='Comparison budgets to plot; default: every available matched positive budget')
    parser.add_argument('--prescribed-only', action='store_true', help='Evaluate/compare only fixed gradient and gradient-plus-Hessian models, preserving the baseline-fit crop; no library integration or regression')
    parser.add_argument('--discovery-benchmarks', action='store_true', help='Discover general tau/phi models and evaluate their fluxes alongside prescribed models; skip frozen-baseline residual searches')
    parser.add_argument('--reuse-runs', help='Scratch root of completed runs to reuse matching general discovery fits; never overwrite or refit a compatible saved model')
    parser.add_argument('--verify-only', action='store_true', help='With --compare-runs, validate completed archives and write an audit without plotting or recomputing fields')
    parser.add_argument('--data-path', default=os.environ.get('DATA_PATH', str(SCRATCH_ROOT / 'preprocessed_paper_data.npz')))
    parser.add_argument('--output-root',
                        default=os.environ.get('OUTPUT_ROOT', str(DEFAULT_OUTPUT_ROOT)),
                        help='Scratch run archive root; the Slurm launcher adds a unique sweep_<array-job> directory')
    parser.add_argument('--workers', type=int, default=int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else None)
    parser.add_argument('--domains', type=int, help='Explicit override; otherwise retain original domain-count formulas')
    parser.add_argument('--domain-width', type=int)
    parser.add_argument('--domain-time', type=int)
    parser.add_argument('--time-start', type=int, default=0)
    parser.add_argument('--time-stop', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--closure-max-terms', type=int)
    parser.add_argument('--maximum-correction-terms', type=int)
    parser.add_argument('--skip-inhomogeneous-experiments', action='store_true', help='Compatibility flag: legacy numeric-target experiments are always excluded')
    parser.add_argument('--skip-baseline-residuals', '--skip-second-order', dest='skip_baseline_residuals', action='store_true', help='Omit gradient calibration/residual searches; general discovery and prescribed first/second-order flux benchmarks remain')
    parser.add_argument('--skip-flux', action='store_true', help='Skip pointwise flux diagnostics')
    parser.add_argument('--flux-batch-size', type=int, default=8, help='Frames per vectorized flux batch; all frames contribute to the paper averages')
    parser.add_argument('--flux-cache-mib', type=int, default=256, help='Derivative and term cache budget per active spatial tile in MiB')
    parser.add_argument('--flux-tile-size', type=int, default=64, help='Spatial tile width; all grid points are included')
    parser.add_argument('--flux-workers', type=int, help='Concurrent flux tile threads (default: up to 8 allocated/configured CPUs)')
    parser.add_argument('--flux-include-general', action='store_true', help='Also evaluate general shared-library candidates; default focuses on first/second-order comparisons')
    parser.add_argument('--plots', action='store_true', help='Save single-scale flux plots; Slurm also submits the Delta-comparison job automatically')
    return parser.parse_args()


def apply_scale_plan(args):
    """Resolve an archived scientific plan; its computation lives in the notebook."""
    if args.scale_index is not None and not args.scale_plan:
        raise ValueError('--scale-index requires --scale-plan')
    if not args.scale_plan:
        return
    path = scratch_path(args.scale_plan)
    plan = json.loads(path.read_bytes())
    if not (path.parent / 'COMPLETE').is_file() or plan.get('schema_version') != 1:
        raise ValueError('Expected a completed schema-1 integral-scale plan')
    expected_hash = json.loads((path.parent / 'measurement.json').read_bytes())['scale_plan_sha256']
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError('Scale plan checksum mismatch')
    if args.compare_runs:
        if args.reference_length is not None or args.expected_scales is not None:
            raise ValueError('--scale-plan already supplies reference length and expected scales')
        args.reference_length = plan['li']
        args.expected_scales = plan['delta_dx']
    elif args.scale_index is not None:
        if args.scale_index not in range(5):
            raise ValueError('Scale index must be between 0 and 4')
        if (args.data_path != str(SCRATCH_ROOT / 'preprocessed_paper_data.npz')
                and str(scratch_path(args.data_path)) != plan['data']['path']):
            raise ValueError('Data override conflicts with scale plan')
        args.data_path = plan['data']['path']
        stat = Path(args.data_path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (plan['data']['size_bytes'], plan['data']['mtime_ns']):
            raise ValueError('Input file changed since integral-scale measurement')
        if args.time_start != 0 or args.time_stop is not None or args.alpha is not None:
            raise ValueError('The scale plan supplies the time window and alpha; omit overrides')
        args.time_start, args.time_stop = plan['time_start'], plan['time_stop']
        args.alpha = plan['alpha']
        args.delta_dx = plan['delta_dx'][args.scale_index]
    else:
        raise ValueError('--scale-plan requires --scale-index or --compare-runs')
    args.scale_plan = str(path)


def measure_li(args):
    """Execute only the source parser and the notebook integral-scale cell."""
    repo = Path(__file__).resolve().parent
    output = scratch_path(args.output_root)
    data = scratch_path(args.data_path)
    if not data.is_file():
        raise FileNotFoundError(data)
    if args.li_wall_margin < 3 or not 0 < args.max_delta_li <= 1:
        raise ValueError('Require li-wall-margin >= 3 and 0 < max-delta-li <= 1')
    root = output / ('li_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '_' + uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    scratch_runtime(root)
    os.chdir(repo)
    content = (repo / 'COnvection-FILTER-Copy2.ipynb').read_bytes()
    notebook = json.loads(content)
    (root / 'executed_notebook.ipynb').write_bytes(content)
    namespace = globals()
    namespace['LI_CONFIG'] = dict(data_path=str(data), run_dir=str(root), time_start=args.time_start,
        time_stop=args.time_stop, wall_margin=args.li_wall_margin, max_delta_li=args.max_delta_li,
        alpha=args.alpha if args.alpha is not None else 3., workers=args.workers or 8,
        notebook_sha256=hashlib.sha256(content).hexdigest())
    try:
        for cid in ('restored-archive-parser', 'measure-integral-scale'):
            cell = next(c for c in notebook['cells'] if c['id'] == cid)
            source = ''.join(cell['source'])
            filename = f'{repo / "COnvection-FILTER-Copy2.ipynb"}::cell::{cid}'
            linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
            exec(compile(source, filename, 'exec'), namespace)
        (root / 'COMPLETE').write_text(datetime.now(timezone.utc).isoformat(), encoding='utf-8')
        print('Integral-scale measurement and plan:', root, flush=True)
    except BaseException:
        (root / 'FAILED').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def compare_runs(args):
    """Execute only the notebook's archive-comparison plotting cell."""
    root = str(scratch_path(args.compare_runs))
    scratch_runtime(Path(root) / ('comparison_runtime_' + uuid4().hex[:8]))
    repo = Path(__file__).resolve().parent
    os.chdir(repo)
    os.environ['MPLBACKEND'] = 'Agg'
    notebook_path = repo / 'COnvection-FILTER-Copy2.ipynb'
    notebook = json.loads(notebook_path.read_bytes())
    cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code'
             and 'pace:sweep_plot' in cell.get('metadata', {}).get('tags', [])]
    if len(cells) != 1:
        raise RuntimeError('Expected one notebook sweep-comparison cell')
    namespace = globals()
    namespace['SWEEP_COMPARE_CONFIG'] = dict(root=root, reference_length=args.reference_length,
        expected_scales=args.expected_scales, correction_budgets=args.correction_budgets,
        prescribed_only=args.prescribed_only, discovery_benchmarks=args.discovery_benchmarks,
        verify_only=args.verify_only)
    if args.scale_plan:
        namespace['SWEEP_COMPARE_CONFIG']['scale_plan'] = args.scale_plan
    source = ''.join(cells[0]['source'])
    filename = f'{notebook_path}::cell::{cells[0]["id"]}'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, 'exec'), namespace)


def run_notebook(args):
    if (args.verify_only or args.reference_length is not None or args.include_corrections
            or args.expected_scales is not None or args.correction_budgets is not None):
        raise ValueError('Comparison options require --compare-runs')
    if args.prescribed_only and (args.skip_flux or args.flux_include_general):
        raise ValueError('--prescribed-only requires flux evaluation and excludes general models')
    if args.discovery_benchmarks and (args.prescribed_only or args.skip_flux):
        raise ValueError('--discovery-benchmarks requires discovery and flux evaluation')
    if args.reuse_runs and not args.discovery_benchmarks:
        raise ValueError('--reuse-runs requires --discovery-benchmarks')
    if sys.version_info < (3, 12):
        raise RuntimeError('Vendored SPIDER uses Python 3.12 syntax; activate its PACE environment')
    for name in ('delta_dx', 'workers', 'domains', 'domain_width', 'domain_time', 'alpha',
                 'closure_max_terms', 'maximum_correction_terms',
                 'flux_batch_size', 'flux_cache_mib', 'flux_tile_size', 'flux_workers'):
        if getattr(args, name) is not None and not getattr(args, name) > 0:
            raise ValueError(f'{name} must be positive')
    if args.time_start < 0 or (args.time_stop is not None and args.time_stop <= args.time_start):
        raise ValueError('Invalid time window')
    repo = Path(__file__).resolve().parent
    # Resolve user paths before switching to the repo for source provenance.
    output_root = scratch_path(args.output_root)
    data_path = str(scratch_path(args.data_path))
    if not Path(data_path).is_file():
        raise FileNotFoundError(f'Preprocessed input is missing: {data_path}')
    os.chdir(repo)
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    # PACE is Linux. Share loaded read-only fields with workers rather than
    # pickling the complete dataset per process (Python 3.14 changed defaults).
    multiprocessing.set_start_method('fork', force=True)
    created = datetime.now(timezone.utc)
    job_dir = output_root / (created.strftime('%Y%m%dT%H%M%S.%fZ')
                            + f'_delta_dx_{args.delta_dx:g}_' + uuid4().hex[:8])
    job_dir.mkdir(parents=True, exist_ok=False)
    scratch_runtime(job_dir)
    config = {name: getattr(args, name) for name in (
        'delta_dx', 'workers', 'domains', 'domain_width', 'domain_time', 'time_start', 'time_stop',
        'alpha', 'closure_max_terms', 'maximum_correction_terms',
        'flux_batch_size', 'flux_cache_mib', 'flux_include_general', 'flux_tile_size', 'flux_workers')
        if getattr(args, name) is not None}
    config['baseline_residuals'] = not (args.skip_baseline_residuals or args.prescribed_only or args.discovery_benchmarks)
    if args.prescribed_only:
        config['prescribed_only'] = True
    if args.discovery_benchmarks:
        config.update(discovery_benchmarks=True, flux_include_general=True)
        if args.reuse_runs:
            config['reuse_runs'] = str(scratch_path(args.reuse_runs))
    if args.scale_plan:
        config['scale_plan'] = args.scale_plan
        config['scale_plan_sha256'] = hashlib.sha256(Path(args.scale_plan).read_bytes()).hexdigest()
    config.update(homogeneous_recovery=False, numeric_target_experiments=False)
    config.update(output_root=str(output_root), run_dir=str(job_dir), single_scale_plots=args.plots)
    if data_path:
        config['data_path'] = data_path
    notebook_path = repo / 'COnvection-FILTER-Copy2.ipynb'
    content = notebook_path.read_bytes()
    notebook = json.loads(content)
    (job_dir / 'executed_notebook.ipynb').write_bytes(content)
    manifest = dict(created_utc=created.isoformat(), status='running', config=config,
        notebook_sha256=hashlib.sha256(content).hexdigest(),
        workflow=('Prescribed gradient/Hessian benchmarks; no discovery' if args.prescribed_only else
                  'SGS discovery, coefficient verification and tiled flux analysis; no broad PDE recovery or legacy experiments'),
        requested=dict(homogeneous_recovery=False, library_integration=not args.prescribed_only,
                       closures=not args.prescribed_only,
                       baseline_residuals=config['baseline_residuals'],
                       numeric_target_experiments=False,
                       flux=not args.skip_flux, plots=args.plots), cells=[])
    allowed = {'setup', 'library', 'closures', 'archive'}
    if args.prescribed_only:
        allowed = {'setup', 'prescribed_setup', 'archive'}
    if args.discovery_benchmarks:
        allowed.update({'prescribed_setup', 'reuse_discovery'})
    if config['baseline_residuals']:
        allowed.add('baseline_residuals')
    if not args.skip_flux:
        allowed.add('flux')
    flux_cells = [cell for cell in notebook['cells'] if 'pace:flux' in cell.get('metadata', {}).get('tags', [])]
    if len(flux_cells) != 1 or flux_cells[0]['id'] != 'dc5ac1e7':
        raise RuntimeError('Expected only the successful replacement cell dc5ac1e7 in the batch flux stage')
    manifest['excluded_cells'] = [cell['id'] for cell in notebook['cells']
        if any(tag.startswith('pace:') and tag[5:] not in allowed
               for tag in cell.get('metadata', {}).get('tags', []))]
    manifest['plot_policy'] = 'replacement-cell flux plots only; legacy diagnostic plot cells are not batch stages'
    print('Run directory:', job_dir, flush=True)
    # Real module globals keep notebook-defined functions picklable by SPIDER's
    # worker processes. The __main__ guard prevents relaunch on multiprocessing spawn.
    namespace = globals()
    namespace['RUN_CONFIG'] = config
    manifest_path = job_dir / 'execution.json'

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    save_manifest()
    try:
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            stages = [tag[5:] for tag in cell.get('metadata', {}).get('tags', []) if tag.startswith('pace:')]
            if not stages or stages[0] not in allowed:
                continue
            record = dict(id=cell['id'], stage=stages[0], status='running')
            manifest['cells'].append(record)
            save_manifest()
            source = ''.join(cell['source'])
            filename = f'{notebook_path}::cell::{cell["id"]}'
            # inspect.getsource can archive functions actually executed from cells.
            linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
            print(f'Executing {record["stage"]}: {cell["id"]}', flush=True)
            started = time.monotonic()
            exec(compile(source, filename, 'exec'), namespace)
            record.update(status='complete', seconds=time.monotonic()-started)
            save_manifest()
        executed = {r['stage'] for r in manifest['cells'] if r['status'] == 'complete'}
        if not allowed <= executed:
            raise RuntimeError(f'Missing requested notebook stages: {allowed-executed}')
        manifest.update(status='complete', validation_status='execution succeeded; scientific errors require review')
        if namespace.get('reuse_archive') is not None:
            manifest['reused_discovery'] = str(namespace['reuse_archive'])
            manifest['execution_methods'] = dict(library_integrals='loaded from archive',
                general_models='loaded exact archived coefficients; no refitting', flux='evaluated on filtered input fields')
        save_manifest()
        (job_dir / 'COMPLETE').write_text(datetime.now(timezone.utc).isoformat(), encoding='utf-8')
    except BaseException:
        manifest.update(status='failed', traceback=traceback.format_exc())
        if manifest['cells'] and manifest['cells'][-1]['status'] == 'running':
            manifest['cells'][-1]['status'] = 'failed'
        save_manifest()
        (job_dir / 'FAILED').write_text(manifest['traceback'], encoding='utf-8')
        raise
    print('Complete:', job_dir, flush=True)


if __name__ == '__main__':
    arguments = parse_args()
    if arguments.prescribed_only and arguments.discovery_benchmarks:
        raise ValueError('Choose --discovery-benchmarks or --prescribed-only, not both')
    apply_scale_plan(arguments)
    if arguments.measure_li:
        measure_li(arguments)
    elif arguments.compare_runs:
        compare_runs(arguments)
    else:
        run_notebook(arguments)
