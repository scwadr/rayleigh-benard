# Archive-only Figure 2 analysis: no filtering, fitting or flux recomputation.
if globals().get('SWEEP_COMPARE_CONFIG'):
    import csv
    import hashlib
    import inspect
    import json
    import linecache
    import re
    import shutil
    import traceback
    from datetime import datetime, timezone
    from pathlib import Path
    from uuid import uuid4

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter, MaxNLocator


    def sweep_audit(report_path):
        """Check execution, source snapshots, discovery arrays and retained coordinates.

        Moment/metric checks are performed by sweep_load below. This cannot prove
        DNS provenance or independently recompute moments from original fields.
        """
        parent = report_path.parent.parent
        execution_path = parent / 'execution.json'
        execution = json.loads(execution_path.read_bytes()) if execution_path.is_file() else None
        if execution is not None:
            if execution.get('status') != 'complete' or (parent / 'FAILED').exists():
                raise ValueError(f'Invalid completed execution: {parent}')
            if not execution.get('cells') or any(c['status'] != 'complete' for c in execution['cells']):
                raise ValueError(f'Unfinished notebook cells: {parent}')
            snapshot = parent / 'executed_notebook.ipynb'
            if hashlib.sha256(snapshot.read_bytes()).hexdigest() != execution['notebook_sha256']:
                raise ValueError(f'Executed notebook checksum mismatch: {snapshot}')
        elif compare_config.get('verify_only'):
            raise ValueError(f'Batch verification requires execution.json: {parent}')
        report = json.loads(report_path.read_bytes())
        bundle = report['discovery_bundle']
        # Use the adjacent archived directory, allowing a whole run to be copied.
        discovery = parent / Path(report['discovery_archive']).name
        if not (discovery / 'COMPLETE').is_file():
            raise ValueError(f'Missing completed discovery archive: {discovery}')
        if json.loads((discovery / 'run.json').read_bytes()) != bundle:
            raise ValueError(f'Flux/discovery manifest mismatch: {discovery}')
        for info in bundle['sources'].values():
            if info.get('archive_file'):
                for directory in (discovery, report_path.parent):
                    saved = directory / info['archive_file']
                    if hashlib.sha256(saved.read_bytes()).hexdigest() != info['sha256']:
                        raise ValueError(f'Source checksum mismatch: {saved}')
        flux_source = report['flux_source']
        if not flux_source.get('executed_cell_file'):
            raise ValueError('Missing executed flux source snapshot')
        saved = report_path.parent / flux_source['executed_cell_file']
        if hashlib.sha256(saved.read_bytes()).hexdigest() != flux_source['executed_cell_sha256']:
            raise ValueError(f'Flux source checksum mismatch: {saved}')
        if bundle.get('libraries'):
            with np.load(discovery / 'shared_libraries.npz', allow_pickle=False) as arrays:
                for index, lib in enumerate(bundle['libraries'].values()):
                    q = arrays[f'Q_{index}']
                    if q.ndim != 2 or q.shape[1] != len(lib['terms']) or not np.isfinite(q).all():
                        raise ValueError(f'Invalid integrated library {index}: {discovery}')
        physics = bundle['scientific_settings']
        if execution is not None and not np.isclose(execution['config']['delta_dx'], physics['delta_dx']):
            raise ValueError('Execution/archive filter-width mismatch')
        if not np.isclose(physics['delta'], physics['delta_dx'] * physics['dx'], rtol=1e-12, atol=0):
            raise ValueError('Inconsistent physical filter width')
        expected_coefficients = {'gradient': physics['delta']**2/12, 'hessian': physics['delta']**4/288}
        for name, value in expected_coefficients.items():
            if not np.isclose(report['prescribed_coefficients'][name], value, rtol=1e-12, atol=0):
                raise ValueError(f'Incorrect prescribed coefficient: {name}')
        with np.load(discovery / 'coordinates.npz', allow_pickle=False) as coordinates, np.load(
                report_path.parent / 'flux_moments.npz', allow_pickle=False) as moments:
            for axis, spacing, crop in zip(('x', 'y'), ('dx', 'dy'), report['settings']['fit_crop_points']):
                values = coordinates[axis]
                trim = int(crop) + int(report['settings']['spatial_margin'])
                retained = values[trim:-trim] if trim else values
                if (len(retained) < 2 or not np.isfinite(retained).all()
                        or not np.allclose(np.diff(retained), physics[spacing])
                        or not np.array_equal(retained, moments[axis])):
                    raise ValueError(f'Invalid retained {axis} coordinates')
            if not np.array_equal(coordinates['t'], moments['t']):
                raise ValueError('Discovery/flux time-coordinate mismatch')
            if (len(moments['t']) != report['settings']['time_count']
                    or len(moments['t']) != bundle['data']['raw_velocity_shape'][2]
                    or list(moments['t'][[0, -1]]) != bundle['data']['time_range']):
                raise ValueError('Flux time count/range mismatch')
        return dict(path=str(parent), delta_dx=physics['delta_dx'], status='archive integrity passed',
                    batch_execution_verified=execution is not None,
                    limitation='Stored moments are checked, not recomputed from DNS; input has path/stat identity only.')

    def sweep_load(fig2_input):
        if fig2_input.name == 'flux.json':
            fig2_input = fig2_input.parent
        if not (fig2_input / 'COMPLETE').is_file():
            raise RuntimeError(f'Flux archive is not marked COMPLETE: {fig2_input}')
        fig2_json_path = fig2_input / 'flux.json'
        fig2_npz_path = fig2_input / 'flux_moments.npz'
        fig2_json_bytes = fig2_json_path.read_bytes()
        fig2_report = json.loads(fig2_json_bytes)
        if fig2_report.get('schema_version') != 4:
            raise ValueError('This cell requires the completed schema-4 flux-moment archive.')

        fig2_expected_columns = [
            'closure_ref_norm2', 'closure_model_norm2', 'closure_cross',
            'flux_ref_norm2', 'flux_model_norm2', 'flux_cross', 'mean_ref_flux', 'mean_model_flux',
        ]
        with np.load(fig2_npz_path, allow_pickle=False) as fig2_npz:
            fig2_moments = np.asarray(fig2_npz['moments'], dtype=float)
            fig2_ids = [str(value) for value in fig2_npz['candidate_ids']]
            fig2_columns = [str(value) for value in fig2_npz['columns']]
            fig2_t = np.asarray(fig2_npz['t'], dtype=float)
            fig2_weights = np.asarray(fig2_npz['time_weights'], dtype=float)
        if fig2_columns != fig2_expected_columns:
            raise ValueError('Unexpected moment column definitions; do not interpret these columns by position.')
        if fig2_moments.shape != (len(fig2_ids), 8) or len(set(fig2_ids)) != len(fig2_ids):
            raise ValueError('Invalid moment table or duplicate candidate IDs.')
        if not np.isfinite(fig2_moments).all():
            raise ValueError('Nonfinite archived flux moments.')
        if (fig2_t.ndim != 1 or len(fig2_t) < 2 or not np.isfinite(fig2_t).all()
                or np.any(np.diff(fig2_t) <= 0) or fig2_weights.shape != fig2_t.shape):
            raise ValueError('Invalid archived times/weights.')
        fig2_expected_weights = np.empty_like(fig2_t)
        fig2_expected_weights[0] = (fig2_t[1]-fig2_t[0])/2
        fig2_expected_weights[-1] = (fig2_t[-1]-fig2_t[-2])/2
        fig2_expected_weights[1:-1] = (fig2_t[2:]-fig2_t[:-2])/2
        fig2_expected_weights /= fig2_expected_weights.sum()
        if not np.allclose(fig2_weights, fig2_expected_weights, rtol=1e-12, atol=1e-14):
            raise ValueError('The archive does not use the expected trapezoidal time average.')
        fig2_lookup = dict(zip(fig2_ids, fig2_moments))
        if {str(row['id']) for row in fig2_report['results']} != set(fig2_ids):
            raise ValueError('The JSON report and moment table refer to different candidate sets.')
        fig2_mean_rtol = float(fig2_report['settings']['mean_ratio_rtol'])
        fig2_physics = fig2_report['discovery_bundle']['scientific_settings']
        fig2_delta_dx = float(fig2_physics['delta_dx'])


        def fig2_metrics(moment):
            """Paper accuracy is a ratio of integrated moments, not mean frame accuracy."""
            caa, cbb, cab, paa, pbb, pab, mean_ref, mean_model = map(float, moment)
            for aa, bb, ab in ((caa, cbb, cab), (paa, pbb, pab)):
                if aa < 0 or bb < 0:
                    raise ValueError('Negative squared norm in archived moments.')
                tolerance = 1e-10*max(aa, bb, abs(ab), np.finfo(float).tiny)
                if abs(ab) > np.sqrt(aa)*np.sqrt(bb) + tolerance:
                    raise ValueError('Archived cross moment violates the Cauchy-Schwarz bound.')
            ratio_defined = paa > 0 and abs(mean_ref) > fig2_mean_rtol*np.sqrt(paa)
            return dict(
                closure_accuracy_A=float(np.clip(cab/max(caa, cbb), -1, 1)) if max(caa, cbb) > 0 else None,
                flux_accuracy_A=float(np.clip(pab/max(paa, pbb), -1, 1)) if max(paa, pbb) > 0 else None,
                mean_flux_ratio=mean_model/mean_ref if ratio_defined else None,
                closure_relative_l2_error=np.sqrt(max(0., caa+cbb-2*cab)/caa) if caa > 0 else None,
                flux_relative_l2_error=np.sqrt(max(0., paa+pbb-2*pab)/paa) if paa > 0 else None,
                time_mean_flux_reference=mean_ref, time_mean_flux_model=mean_model,
            )


        def fig2_same_number(a, b):
            if a is None or b is None:
                return a is None and b is None
            return bool(np.isclose(a, b, rtol=1e-9, atol=1e-12))


        def fig2_number(value):
            return 'undefined' if value is None else f'{value:.6g}'


        fig2_rows = []
        fig2_sequences = {target: {'fitted': {}, 'theoretical': {}, 'general': {}} for target in ('tau', 'phi')}
        fig2_prescribed = {target: {} for target in ('tau', 'phi')}
        for fig2_original_row in fig2_report['results']:
            fig2_row = dict(fig2_original_row)
            fig2_values = fig2_metrics(fig2_lookup[str(fig2_row['id'])])
            for fig2_metric, fig2_value in fig2_values.items():
                if not fig2_same_number(fig2_row.get(fig2_metric), fig2_value):
                    raise ValueError(f'JSON/moment mismatch: candidate {fig2_row["id"]}, {fig2_metric}.')
            fig2_row.update(fig2_values)
            fig2_row['correction_budget'] = None
            fig2_row['plotted'] = False
            fig2_target, fig2_family = fig2_row['target'], fig2_row['family']
            if fig2_target not in fig2_sequences:
                raise ValueError(f'Unknown target {fig2_target}.')
            if fig2_family in ('fitted', 'theoretical', 'general'):
                fig2_matches = [re.search(r'\|candidate=(\d+)$', model_id) for model_id in fig2_row['model_ids']]
                if not fig2_matches or any(match is None for match in fig2_matches):
                    raise ValueError('Cannot identify requested correction budgets from model IDs.')
                fig2_budgets = [int(match.group(1)) - (1 if fig2_family == 'general' else 0) for match in fig2_matches]
                if len(fig2_budgets) != (2 if fig2_target == 'tau' else 1):
                    raise ValueError('Unexpected trace/STF or vector candidate assembly.')
                if len(set(fig2_budgets)) == 1:
                    fig2_budget = fig2_budgets[0]
                    if fig2_budget in fig2_sequences[fig2_target][fig2_family]:
                        raise ValueError('Duplicate family/target/correction budget.')
                    fig2_row.update(correction_budget=fig2_budget if fig2_family != 'general' else None,
                                    predictor_budget=fig2_budget if fig2_family == 'general' else None, plotted=True)
                    fig2_sequences[fig2_target][fig2_family][fig2_budget] = fig2_row
            elif fig2_family == 'prescribed':
                fig2_order = int(fig2_row['model_order'])
                if fig2_order in fig2_prescribed[fig2_target]:
                    raise ValueError('Duplicate prescribed model.')
                fig2_row['plotted'] = True
                fig2_prescribed[fig2_target][fig2_order] = fig2_row
            fig2_rows.append(fig2_row)

        for fig2_target in ('tau', 'phi'):
            if set(fig2_prescribed[fig2_target]) != {1, 2}:
                raise ValueError(f'Missing prescribed first/second-order {fig2_target} model.')
        return fig2_report, fig2_rows, fig2_sequences, fig2_prescribed

    fig2_rc = {
        'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'text.usetex': False, 'font.size': 13,
        'axes.linewidth': .75, 'axes.labelsize': 18, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
        'lines.linewidth': 1.15, 'savefig.facecolor': 'white', 'figure.facecolor': 'white',
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    }

    # Reuse the next-cell analysis definitions, colors and layout; vary width here.
    compare_config = dict(SWEEP_COMPARE_CONFIG)
    compare_root = Path(compare_config['root']).expanduser().resolve()
    scratch_root = Path('/storage/scratch1/2/ashelestov3').resolve()
    if not compare_root.is_relative_to(scratch_root) or not compare_root.is_dir():
        raise ValueError('Comparison input/output must be an existing scratch directory.')
    reference_length = compare_config.get('reference_length')
    if reference_length is not None and (not np.isfinite(reference_length) or reference_length <= 0):
        raise ValueError('reference_length must be finite and positive.')
    executed_source = ''.join(linecache.getlines(inspect.currentframe().f_code.co_filename))
    runs, skipped, all_rows, audits = [], [], [], []
    signature = None
    plan = json.loads(Path(compare_config['scale_plan']).read_bytes()) if compare_config.get('scale_plan') else None
    for execution_path in sorted(compare_root.rglob('execution.json')):
        manifest = json.loads(execution_path.read_bytes())
        if manifest.get('status') != 'complete' or not (execution_path.parent / 'COMPLETE').is_file():
            skipped.append(dict(path=str(execution_path.parent), reason='incomplete or failed batch execution',
                                delta_dx=manifest.get('config', {}).get('delta_dx')))
    for report_path in sorted(compare_root.rglob('flux.json')):
        parent_run = report_path.parent.parent
        if (not (report_path.parent / 'COMPLETE').is_file()
                or ((parent_run / 'execution.json').is_file() and not (parent_run / 'COMPLETE').is_file())):
            skipped.append(dict(path=str(report_path), reason='incomplete flux or batch run'))
            continue
        if json.loads(report_path.read_bytes()).get('schema_version') != 4:
            skipped.append(dict(path=str(report_path), reason='different flux schema'))
            continue
        audits.append(sweep_audit(report_path))
        report, rows, sequences, prescribed = sweep_load(report_path.parent)
        discovery = report['discovery_bundle']
        physics = discovery['scientific_settings']
        delta, delta_dx = float(physics['delta']), float(physics['delta_dx'])
        if not np.isfinite([delta, delta_dx]).all() or min(delta, delta_dx) <= 0:
            raise ValueError(f'Invalid width: {report_path}')
        if any(np.isclose(delta_dx, run['delta_dx'], rtol=1e-12, atol=0) for run in runs):
            raise ValueError(f'Duplicate width {delta_dx}; compare one dedicated sweep directory.')
        controls = {k: v for k, v in discovery['run_controls'].items() if k not in {
            'run_dir', 'output_root', 'delta_dx', 'workers', 'flux_batch_size',
            'flux_cache_mib', 'flux_workers', 'flux_tile_size', 'single_scale_plots'}}
        data = dict(discovery['data'])
        data.pop('filtered_velocity_shape', None)
        settings = {k: v for k, v in report['settings'].items() if k not in {
            'batch_size', 'derivative_cache_mib', 'derivative_cache_peak_bytes', 'fit_crop_points'}}
        if compare_config.get('prescribed_only') or compare_config.get('discovery_benchmarks'):
            # Evaluation compatibility is checked separately from archived fit provenance.
            # Fixed-form-only plots may ignore discovery controls; discovered-model plots retain them.
            comparison_controls = {'time_start', 'time_stop', 'alpha'}
            if compare_config.get('discovery_benchmarks'):
                comparison_controls.update({'closure_max_terms', 'domains', 'domain_width', 'domain_time'})
                if not report['settings']['include_general']:
                    raise ValueError('Archive lacks discovered-model fluxes; reuse fits with --discovery-benchmarks to evaluate them.')
            controls = {k: v for k, v in controls.items() if k in comparison_controls}
            settings = {k: v for k, v in settings.items() if k not in ('include_general', 'tau_assembly', 'default_curves')}
            expected_crop = [int(np.ceil(physics['alpha'] * delta / physics[axis])) for axis in ('dx', 'dy')]
            if report['settings']['fit_crop_points'] != expected_crop:
                raise ValueError('Benchmark comparison requires the original baseline-fit crop at every width.')
        if plan is not None:
            identity = discovery['data']['npz']
            if identity != dict(plan['data'], exists=True):
                raise ValueError('Comparison data identity differs from integral-scale plan')
            if not np.isclose(physics['dx'], plan['dx'], rtol=1e-12, atol=0):
                raise ValueError('Comparison grid differs from integral-scale plan')
            if discovery['run_controls'].get('scale_plan_sha256') != hashlib.sha256(Path(compare_config['scale_plan']).read_bytes()).hexdigest():
                raise ValueError('Run does not reference this measured scale plan')
        with np.load(report_path.parent / 'flux_moments.npz', allow_pickle=False) as moments:
            time_digest = hashlib.sha256(np.asarray(moments['t'], dtype='<f8').tobytes()).hexdigest()
        current_signature = json.dumps(dict(data=data, time_sha256=time_digest, controls=controls, flux_settings=settings,
            physical_parameters={k: physics[k] for k in ('dx', 'dy', 'dt', 'Rayleigh', 'Prandtl', 'alpha', 'filter')}), sort_keys=True)
        if signature is not None and signature != current_signature:
            raise ValueError('Mixed data/settings in comparison root; use one directory per sweep.')
        signature = current_signature
        for row in rows:
            row.update(delta=delta, delta_dx=delta_dx, source_archive=str(report_path), plotted=False)
        all_rows.extend(rows)
        runs.append(dict(delta=delta, delta_dx=delta_dx, path=report_path,
                         sequences=sequences, prescribed=prescribed, definitions=report['definitions']))
    runs.sort(key=lambda run: run['delta'])
    if len(runs) < 2:
        raise ValueError(f'Need at least two completed widths; found {len(runs)}. Skipped: {skipped}')
    expected = compare_config.get('expected_scales')
    if expected is not None:
        actual = sorted(run['delta_dx'] for run in runs)
        if len(actual) != len(expected) or not np.allclose(actual, sorted(expected), rtol=1e-12, atol=0):
            raise ValueError(f'Expected Delta/dx={expected}, found {actual}.')
    available_budgets = sorted({k for run in runs for groups in run['sequences'].values()
                                for seq in groups.values() for k in seq if k > 0})
    if compare_config.get('discovery_benchmarks'):
        available_budgets = sorted({k for run in runs for seq in run['sequences'].values() for k in seq['general'] if k > 0})
        if not available_budgets:
            raise ValueError('No nonzero general-discovery predictor budgets to plot')
        for run in runs:
            for target in ('tau', 'phi'):
                if any(k not in run['sequences'][target]['general'] for k in available_budgets):
                    raise ValueError(f'Missing matched discovered-model budgets at {run["delta_dx"]}, {target}')
    budgets = compare_config.get('correction_budgets')
    if budgets is None:
        budgets = available_budgets or [0]  # Prescribed-only runs remain plottable.
    elif not budgets or any(k < 0 or k not in [0, *available_budgets] for k in budgets):
        raise ValueError(f'Unknown correction budgets; available: {available_budgets} (or 0 for baselines).')
    budgets = [] if compare_config.get('verify_only') else ([0] if compare_config.get('prescribed_only') else sorted(set(budgets)))
    x = [run['delta']/reference_length if reference_length else run['delta'] for run in runs]
    metrics = ('closure_accuracy_A', 'flux_accuracy_A', 'mean_flux_ratio')
    output = compare_root / ('figure2_delta_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '_' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    plot_records, notices, sources = [], [], []
    try:
        if executed_source:
            (output / 'executed_analysis_cell.py').write_text(executed_source, encoding='utf-8')
        if plan is not None:
            for name in ('scale_plan.json', 'measurement.json', 'li_moments.npz'):
                content = (Path(compare_config['scale_plan']).parent / name).read_bytes()
                (output / name).write_bytes(content)
        for index, run in enumerate(runs):
            source_dir = output / 'sources' / f'width_{index:03d}'
            source_dir.mkdir(parents=True)
            digests = {}
            for filename in ('flux.json', 'flux_moments.npz'):
                content = (run['path'].parent / filename).read_bytes()
                (source_dir / ('source_' + filename)).write_bytes(content)
                digests[filename] = hashlib.sha256(content).hexdigest()
            sources.append(dict(delta=run['delta'], delta_dx=run['delta_dx'],
                                archive=str(run['path']), sha256=digests))
        with plt.rc_context(fig2_rc):
            for budget in budgets:
                for panel, target in zip(('a', 'b'), ('tau', 'phi')):
                    def sequence(family, count):
                        return [run['sequences'][target][family].get(count) for run in runs]
                    curves = [
                        ('theoretical_correction', sequence('theoretical', budget), '#0072BD', '-', f'Theoretical baseline + {budget} terms'),
                        ('prescribed_second', [run['prescribed'][target][2] for run in runs], '#D95319', '-', 'Prescribed gradient + Hessian'),
                        ('fitted_baseline', sequence('fitted', 0), '#EDB120', '-.', 'Fitted gradient'),
                        ('fitted_correction', sequence('fitted', budget), '#7E2F8E', '--', f'Fitted baseline + {budget} terms'),
                        ('prescribed_first', [run['prescribed'][target][1] for run in runs], '#77AC30', '-', 'Prescribed gradient'),
                    ]
                    if compare_config.get('discovery_benchmarks'):
                        curves = [('general_discovery', sequence('general', budget), '#0072BD', '-',
                                   f'Discovered ({budget} predictors)')] + [curve for curve in curves
                                   if curve[0] in ('prescribed_first', 'prescribed_second')]
                    if compare_config.get('prescribed_only'):
                        curves = [curve for curve in curves if curve[0] in ('prescribed_first', 'prescribed_second')]
                    if budget == 0:
                        curves = [curve for curve in curves if curve[0] not in ('theoretical_correction', 'fitted_correction')]
                    curves = [curve for curve in curves if any(row is not None for row in curve[1])]
                    for key, rows, color, style, label in curves:
                        missing = [run['delta_dx'] for run, row in zip(runs, rows) if row is None]
                        if missing:
                            notices.append(f'{target}, N={budget}, {key}: missing at Delta/dx={missing}; gaps retained.')
                        for row in rows:
                            if row is not None:
                                row['plotted'] = True
                    fig, axes = plt.subplots(3, 1, figsize=(4.8, 9.5), sharex=True)
                    fig.subplots_adjust(left=.24, right=.97, bottom=.105, top=.81, hspace=.28)
                    symbol = r'\tau' if target == 'tau' else r'\phi'
                    labels = (rf'$\mathcal{{A}}({symbol})$', rf'$\mathcal{{A}}(\Pi_{{{symbol}}})$',
                              rf'$\langle\Pi_{{{symbol}}}^{{\mathrm{{LES}}}}\rangle/\langle\Pi_{{{symbol}}}^{{\mathrm{{DNS}}}}\rangle$')
                    for axis, metric, ylabel in zip(axes, metrics, labels):
                        values = []
                        for key, rows, color, style, label in curves:
                            y = [np.nan if row is None or row[metric] is None else row[metric] for row in rows]
                            axis.plot(x, y, color=color, linestyle=style, label=label, marker='o', markersize=3)
                            values.extend(value for value in y if np.isfinite(value))
                            plot_records.append(dict(target=target, correction_budget=None if compare_config.get('discovery_benchmarks') else budget,
                                predictor_budget=budget if compare_config.get('discovery_benchmarks') else None, metric=metric,
                                curve=key, label=label, color=color, linestyle=style, x=x,
                                y=[float(value) if np.isfinite(value) else None for value in y],
                                candidate_ids=[row['id'] if row is not None else None for row in rows]))
                        lo, hi = (0., 3.) if metric == 'mean_flux_ratio' else (0., 1.)
                        ticks = [0, 1, 2, 3] if metric == 'mean_flux_ratio' else [0, .25, .5, .75, 1]
                        if values and (min(values) < lo or max(values) > hi):
                            lo, hi = min(lo, min(values)), max(hi, max(values))
                            ticks = None
                            notices.append(f'Expanded {target}, N={budget}, {metric} limits to retain all values.')
                        padding = .025*(hi-lo)
                        axis.set(ylim=(lo-padding, hi+padding), ylabel=ylabel,
                                 xlabel=(r'$\Delta/\ell_i$' if plan is not None else r'$\Delta/L$') if reference_length else r'$\Delta$')
                        axis.set_xticks(x)
                        if ticks is not None:
                            axis.set_yticks(ticks)
                        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
                        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
                        axis.tick_params(direction='in', top=True, right=True, length=3.5, width=.75, labelbottom=True)
                        axis.grid(False)
                    handles, labels = axes[0].get_legend_handles_labels()
                    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.16, .956),
                               frameon=False, fontsize=10, handlelength=2.7, labelspacing=.35)
                    budget_symbol = 'pred' if compare_config.get('discovery_benchmarks') else 'corr'
                    fig.text(.6, .985, rf'${symbol},\quad N_{{\mathrm{{{budget_symbol}}}}}={budget}$', ha='center', va='top', fontsize=16)
                    fig.text(.6, .02, 'Convection stress' if target == 'tau' else 'Convection scalar flux', ha='center', fontsize=12)
                    for extension in ('png', 'pdf', 'svg'):
                        budget_prefix = 'P' if compare_config.get('discovery_benchmarks') else 'N'
                        fig.savefig(output / f'figure2_{target}_{budget_prefix}{budget}_vs_delta.{extension}', dpi=300)
                    plt.show()
                    plt.close(fig)
        columns = ['delta', 'delta_dx', 'target', 'id', 'family', 'label', 'model_order',
                   'correction_budget', 'predictor_budget', 'plotted', *metrics, 'closure_relative_l2_error',
                   'flux_relative_l2_error', 'time_mean_flux_reference', 'time_mean_flux_model',
                   'model_ids', 'source_archive']
        with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in all_rows:
                record = {key: row.get(key) for key in columns}
                record['model_ids'] = json.dumps(row['model_ids'])
                writer.writerow(record)
        notes = [
            'Figure 2-style measurements on convection data: closure accuracy A, flux accuracy A, mean modeled/reference flux.',
            'Not a reproduction of paper F1/F2: these are different flows, models and grid/filter conventions.',
            'Paper Fig. 2 compares NGMR, NGM4, similarity, dynamic mixed and dynamic Smagorinsky; only NGM4 matches a model here.',
            'The existing fine-grid edge-padded Gaussian convention is retained for comparability; no coarse-grid spectral projection.',
            'The paper axis is Delta/ell_i (Eq. 3.3); --scale-plan supplies a measured convection value, while a bare reference_length is user-provided.',
            'A is the ratio of integrated moments, not the average of per-frame accuracy.',
            'All archived moments, trapezoidal time weights and JSON metrics were checked for consistency.',
            'Each panel set holds its requested predictor/correction budget fixed and varies physical Delta (or Delta/L with explicit L).',
            'Only equal-budget trace/STF pairs are drawn; P denotes general predictors and N denotes frozen-baseline corrections.',
            'All candidates, including mixed-budget pairs, are preserved in metrics.csv and source archives.',
            'Candidates are refitted at each width; term support and the original fit crop may change.',
            'Discovered residuals are not assumed to equal the prescribed Hessian correction.',
            'General-discovery plots compare equal requested predictor counts (forced target excluded); tau uses the same budget for trace and STF.',
            'General discovered closures are fitted directly to measured tau/phi, without a frozen gradient baseline.',
            'These are discovery-data fits, not independent validation or thesis-selected models.', 
            'Prescribed phi is the analogous Gaussian scalar expansion; the reference paper studies stress.',
            'Colors/layout match the next-cell paper-style analysis; markers identify evaluated widths.',
            'Undefined ratios and missing candidates are gaps; no interpolation or best-model selection.',
        ]
        (output / 'analysis.txt').write_text('\n'.join(notes + notices) + '\n', encoding='utf-8')
        manifest = dict(config=compare_config, scale_plan=plan, sources=sources, skipped=skipped, audits=audits, notes=notes, notices=notices,
            definitions=runs[0]['definitions'], correction_budgets=budgets, plotted_curves=plot_records,
            plot_settings=dict(rc=fig2_rc, figure_inches=[4.8, 9.5], markers=True),
            executed_cell_sha256=hashlib.sha256(executed_source.encode('utf-8')).hexdigest())
        (output / 'analysis.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
        (output / 'COMPLETE').write_text(datetime.now(timezone.utc).isoformat(), encoding='utf-8')
    except BaseException:
        (output / 'FAILED').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    print('Archive integrity checks passed for Delta/dx:', [run['delta_dx'] for run in runs])
    print('Verification/benchmark output:', output)
