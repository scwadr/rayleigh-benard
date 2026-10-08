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
    mode.add_argument('--compare-runs', help='Plot paper accuracy versus width from completed flux archives under this root')
    parser.add_argument('--reference-length', type=float, help='Physical reference length for comparison x axis; otherwise use Delta/dx')
    parser.add_argument('--include-corrections', action='store_true', help='Include independently refitted correction candidates in comparison curves')
    parser.add_argument('--data-path')
    parser.add_argument('--output-root', default='thesis_runs')
    parser.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    parser.add_argument('--domains', type=int, default=1000)
    parser.add_argument('--domain-width', type=int, default=64)
    parser.add_argument('--domain-time', type=int, default=40)
    parser.add_argument('--time-start', type=int, default=0)
    parser.add_argument('--time-stop', type=int)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=3.)
    parser.add_argument('--recovery-max-terms', type=int, default=8)
    parser.add_argument('--closure-max-terms', type=int, default=6)
    parser.add_argument('--maximum-correction-terms', type=int, default=5)
    parser.add_argument('--skip-second-order', action='store_true', help='Run recovery and first-order/shared closures only; also skips flux')
    parser.add_argument('--skip-flux', action='store_true', help='Skip expensive pointwise second-order flux diagnostics')
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
                 'recovery_max_terms', 'closure_max_terms', 'maximum_correction_terms'):
        if not getattr(args, name) > 0:
            raise ValueError(f'{name} must be positive')
    if args.recovery_max_terms < 6:
        raise ValueError('Use at least 6 recovery terms to admit the complete momentum balance')
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
        'seed', 'alpha', 'recovery_max_terms', 'closure_max_terms', 'maximum_correction_terms')}
    config.update(output_root=str(output_root), run_dir=str(job_dir))
    if data_path:
        config['data_path'] = data_path
    notebook_path = repo / 'COnvection-FILTER-Copy2.ipynb'
    content = notebook_path.read_bytes()
    notebook = json.loads(content)
    (job_dir / 'executed_notebook.ipynb').write_bytes(content)
    manifest = dict(created_utc=created.isoformat(), status='running', config=config,
        notebook_sha256=hashlib.sha256(content).hexdigest(),
        requested=dict(second_order=not args.skip_second_order,
                       flux=not (args.skip_second_order or args.skip_flux), plots=args.plots), cells=[])
    allowed = {'setup', 'recovery', 'first_order', 'shared_closure'}
    if not args.skip_second_order:
        allowed.add('second_order')
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
            exec(compile(source, filename, 'exec'), namespace)
            record.update(status='complete', seconds=time.monotonic()-started)
            save_manifest()
        executed = {r['stage'] for r in manifest['cells'] if r['status'] == 'complete'}
        if not allowed <= executed:
            raise RuntimeError(f'Missing requested notebook stages: {allowed-executed}')
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
