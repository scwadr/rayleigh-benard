"""Execute tagged discovery notebook cells for one PACE filter-scale task.

Scientific implementation and run archiving live in the notebook. This launcher
only selects cells, supplies configuration, and records execution/failure status.
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
import time
import traceback
from uuid import uuid4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--delta-dx', type=float, help='Gaussian delta in raw x grid spacings')
    mode.add_argument('--compare-runs', help='Plot Figure 2 metrics versus width from completed batched flux archives under this root')
    parser.add_argument('--reference-length', type=float, help='Physical reference length for comparison x axis; otherwise use Delta/dx')
    parser.add_argument('--include-corrections', action='store_true', help='Also plot mixed trace/STF correction budgets (matched budgets are plotted by default)')
    parser.add_argument('--data-path')
    parser.add_argument('--output-root',
                        default=os.environ.get('OUTPUT_ROOT', '/storage/scratch1/2/ashelestov3/rayleigh-benard/filter_sweeps'),
                        help='Run archive root (default: OUTPUT_ROOT or the PACE scratch filter_sweeps directory)')
    parser.add_argument('--workers', type=int, default=int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else None)
    parser.add_argument('--domains', type=int, help='Explicit override; otherwise retain original domain-count formulas')
    parser.add_argument('--domain-width', type=int)
    parser.add_argument('--domain-time', type=int)
    parser.add_argument('--time-start', type=int, default=0)
    parser.add_argument('--time-stop', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--homogeneous-max-k', type=int, help='Explicit upper limit for original AIC sweep (default 39)')
    parser.add_argument('--closure-max-terms', type=int)
    parser.add_argument('--maximum-correction-terms', type=int)
    parser.add_argument('--skip-inhomogeneous-experiments', action='store_true', help='Skip the consolidated original numeric-target experiments')
    parser.add_argument('--skip-baseline-residuals', '--skip-second-order', dest='skip_baseline_residuals', action='store_true', help='Omit gradient calibration/residual searches; general discovery and prescribed first/second-order flux benchmarks remain')
    parser.add_argument('--skip-flux', action='store_true', help='Skip pointwise flux diagnostics')
    parser.add_argument('--flux-batch-size', type=int, default=8, help='Frames per vectorized flux batch; all frames contribute to the paper averages')
    parser.add_argument('--flux-cache-mib', type=int, default=256, help='Derivative and term cache budget per active spatial tile in MiB')
    parser.add_argument('--flux-tile-size', type=int, default=64, help='Spatial tile width; all grid points are included')
    parser.add_argument('--flux-workers', type=int, help='Concurrent flux tile threads (default: up to 8 allocated/configured CPUs)')
    parser.add_argument('--flux-include-general', action='store_true', help='Also evaluate general shared-library candidates; default focuses on first/second-order comparisons')
    parser.add_argument('--plots', action='store_true', help='Save the final plotting cells using Agg')
    return parser.parse_args()


def compare_runs(args):
    """Execute only the notebook's archive-comparison plotting cell."""
    root = str(Path(args.compare_runs).resolve())
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
                                            include_corrections=args.include_corrections)
    source = ''.join(cells[0]['source'])
    exec(compile(source, f'{notebook_path}::cell::{cells[0]["id"]}', 'exec'), namespace)


def run_notebook(args):
    if args.reference_length is not None or args.include_corrections:
        raise ValueError('--reference-length and --include-corrections require --compare-runs')
    if sys.version_info < (3, 12):
        raise RuntimeError('Vendored SPIDER uses Python 3.12 syntax; activate its PACE environment')
    for name in ('delta_dx', 'workers', 'domains', 'domain_width', 'domain_time', 'alpha',
                 'homogeneous_max_k', 'closure_max_terms', 'maximum_correction_terms',
                 'flux_batch_size', 'flux_cache_mib', 'flux_tile_size', 'flux_workers'):
        if getattr(args, name) is not None and not getattr(args, name) > 0:
            raise ValueError(f'{name} must be positive')
    if args.homogeneous_max_k is not None and args.homogeneous_max_k < 5:
        raise ValueError('The original homogeneous AIC sweep starts at max_k=5')
    if args.time_start < 0 or (args.time_stop is not None and args.time_stop <= args.time_start):
        raise ValueError('Invalid time window')
    repo = Path(__file__).resolve().parent
    # Resolve user paths before switching to the repo for source provenance.
    output_root = Path(args.output_root).resolve()
    data_path = str(Path(args.data_path).resolve()) if args.data_path else None
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
    config = {name: getattr(args, name) for name in (
        'delta_dx', 'workers', 'domains', 'domain_width', 'domain_time', 'time_start', 'time_stop',
        'alpha', 'homogeneous_max_k', 'closure_max_terms', 'maximum_correction_terms',
        'flux_batch_size', 'flux_cache_mib', 'flux_include_general', 'flux_tile_size', 'flux_workers')
        if getattr(args, name) is not None}
    config['baseline_residuals'] = not args.skip_baseline_residuals
    config.update(output_root=str(output_root), run_dir=str(job_dir))
    if data_path:
        config['data_path'] = data_path
    notebook_path = repo / 'COnvection-FILTER-Copy2.ipynb'
    content = notebook_path.read_bytes()
    notebook = json.loads(content)
    (job_dir / 'executed_notebook.ipynb').write_bytes(content)
    manifest = dict(created_utc=created.isoformat(), status='running', config=config,
        notebook_sha256=hashlib.sha256(content).hexdigest(),
        workflow='restored original notebook',
        requested=dict(baseline_residuals=not args.skip_baseline_residuals,
                       numeric_target_experiments=not args.skip_inhomogeneous_experiments,
                       flux=not args.skip_flux, plots=args.plots), cells=[])
    allowed = {'setup', 'library', 'homogeneous', 'closures', 'archive'}
    if not args.skip_baseline_residuals:
        allowed.add('baseline_residuals')
    if not args.skip_inhomogeneous_experiments:
        allowed.add('inhomogeneous')
    if not args.skip_flux:
        allowed.add('flux')
    if args.plots:
        allowed.add('plot')
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
            if stages[0] == 'plot':
                import matplotlib.pyplot as plt
                plot_dir = job_dir / 'plots'
                plot_dir.mkdir(exist_ok=True)
                original_show = plt.show
                figure_count = [0]

                def save_shown_figures(*unused_args, **unused_kwargs):
                    for number in plt.get_fignums():
                        figure_count[0] += 1
                        plt.figure(number).savefig(
                            plot_dir / f'{cell["id"]}_{figure_count[0]:02d}.png', dpi=150)
                    plt.close('all')

                plt.show = save_shown_figures
                try:
                    exec(compile(source, filename, 'exec'), namespace)
                finally:
                    plt.show = original_show
            else:
                exec(compile(source, filename, 'exec'), namespace)
            record.update(status='complete', seconds=time.monotonic()-started)
            save_manifest()
        executed = {r['stage'] for r in manifest['cells'] if r['status'] == 'complete'}
        if not allowed <= executed:
            raise RuntimeError(f'Missing requested notebook stages: {allowed-executed}')
        if namespace.get('legacy_regression_failures'):
            manifest.update(status='finished_with_errors',
                            regression_failures=namespace['legacy_regression_failures'])
            save_manifest()
            raise RuntimeError('Original homogeneous sweep caught errors; see archived failures')
        manifest.update(status='complete', validation_status='execution succeeded; scientific errors require review')
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
    if arguments.compare_runs:
        compare_runs(arguments)
    else:
        run_notebook(arguments)
