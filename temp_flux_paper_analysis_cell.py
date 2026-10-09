# %% Paste this entire file into ONE cell after the completed flux analysis.
# Or run in that SAME kernel: %run -i temp_flux_paper_analysis_cell.py
# Reads only the small completed flux archive. No fields, derivatives, fits, or
# fluxes are recomputed. Existing notebook state and archives are preserved.
#
# Optional overrides BEFORE running this cell:
# FIG2_ANALYSIS_CONFIG = {
#     'archive': str(flux_dir),  # defaults to the current kernel's flux_dir
#     'output_root': '/storage/scratch1/2/ashelestov3/rayleigh-benard/flux_analysis',
#     'mean_ratio_ylim': (0, 3), # paper limits; use (0, 1.1) for a closer view
#     'markers': False,         # paper-style lines; True marks evaluated models
# }
# The three measurements match paper Eqs. (3.8)-(3.9) and Figure 2(a).
# HERE the x-axis is requested correction count at ONE filter width, as requested;
# it is NOT the paper's delta = Delta / integral-length-scale axis.

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

fig2_executed_source = ''.join(linecache.getlines(inspect.currentframe().f_code.co_filename))
fig2_config = dict(globals().get('FIG2_ANALYSIS_CONFIG', {}))
fig2_input = fig2_config.get('archive', globals().get('flux_dir'))
if fig2_input is None:
    raise RuntimeError('Run the flux calculation/archive first, or set FIG2_ANALYSIS_CONFIG["archive"].')
fig2_input = Path(fig2_input).expanduser().resolve()
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
fig2_sequences = {target: {'fitted': {}, 'theoretical': {}} for target in ('tau', 'phi')}
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
    if fig2_family in ('fitted', 'theoretical'):
        fig2_matches = [re.search(r'\|candidate=(\d+)$', model_id) for model_id in fig2_row['model_ids']]
        if not fig2_matches or any(match is None for match in fig2_matches):
            raise ValueError('Cannot identify requested correction budgets from model IDs.')
        fig2_budgets = [int(match.group(1)) for match in fig2_matches]
        if len(fig2_budgets) != (2 if fig2_target == 'tau' else 1):
            raise ValueError('Unexpected trace/STF or vector candidate assembly.')
        if len(set(fig2_budgets)) == 1:
            fig2_budget = fig2_budgets[0]
            if fig2_budget in fig2_sequences[fig2_target][fig2_family]:
                raise ValueError('Duplicate family/target/correction budget.')
            fig2_row.update(correction_budget=fig2_budget, plotted=True)
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
    for fig2_family in ('fitted', 'theoretical'):
        if 0 not in fig2_sequences[fig2_target][fig2_family]:
            raise ValueError(f'Missing {fig2_family} {fig2_target} baseline; use a run with baseline/residual discovery.')
if not any(k > 0 for groups in fig2_sequences.values() for seq in groups.values() for k in seq):
    raise ValueError('No residual corrections to plot against correction count.')

# Paper/MATLAB color order and line styles. Labels identify OUR models, not the
# NGMR/Smagorinsky/similarity models used in the paper's different experiment.
fig2_styles = {
    'theoretical_sequence': dict(color='#0072BD', linestyle='-', label=r'Theoretical baseline + $N_{\rm corr}$ terms'),
    'prescribed_second': dict(color='#D95319', linestyle='-', label='Prescribed gradient + Hessian'),
    'fitted_baseline': dict(color='#EDB120', linestyle='-.', label='Fitted gradient (fixed baseline)'),
    'fitted_sequence': dict(color='#7E2F8E', linestyle='--', label=r'Fitted baseline + $N_{\rm corr}$ terms'),
    'prescribed_first': dict(color='#77AC30', linestyle='-', label='Prescribed gradient'),
}
fig2_metrics_to_plot = ('closure_accuracy_A', 'flux_accuracy_A', 'mean_flux_ratio')
fig2_text = [
    f'Figure 2 measurements at Delta/dx = {fig2_delta_dx:g}',
    f'Source archive: {fig2_input}',
    f'Time samples: {len(fig2_t)}; all archived trapezoidal weights verified.',
    'A = integral <reference:model> dt / max(integral <reference:reference> dt, integral <model:model> dt).',
    'Mean-flux ratio = time-averaged spatial mean modeled Pi / time-averaged spatial mean reference Pi.',
    'Pi_tau = -S_ij tau_ij; Pi_phi = -phi_i d_i bbar; b is the simulated temperature-like scalar.',
    'The x-axis is requested correction count, not filter width or asymptotic model order.',
    'For tau, N_corr is the requested budget for EACH trace/STF fit, paired at equal budgets.',
    'Horizontal baselines are fixed models, repeated only for visual comparison.',
    'All candidate equations/coefficients remain in source_flux.json; mixed-budget pairs remain in metrics.csv.',
    'These are a priori fits on discovery data, not independent validation or thesis model selection.',
    '',
]
for fig2_target in ('tau', 'phi'):
    fig2_text += [fig2_target.upper(), 'model                                  A(closure)    A(flux)       mean ratio    relative L2(closure)']
    for fig2_row in (row for row in fig2_rows if row['target'] == fig2_target and row['plotted']):
        fig2_text.append(f"{fig2_row['label']:<38} " + ' '.join(
            f'{fig2_number(fig2_row[key]):<13}' for key in (*fig2_metrics_to_plot, 'closure_relative_l2_error')))
    for fig2_family in ('fitted', 'theoretical'):
        fig2_seq = fig2_sequences[fig2_target][fig2_family]
        fig2_base = fig2_seq[0]
        fig2_worse = [str(k) for k, row in sorted(fig2_seq.items()) if k > 0
            and row['closure_accuracy_A'] is not None and fig2_base['closure_accuracy_A'] is not None
            and row['closure_accuracy_A'] < fig2_base['closure_accuracy_A'] - 1e-10]
        if fig2_worse:
            fig2_text.append(f'{fig2_family}: closure accuracy is BELOW its baseline at budgets ' + ', '.join(fig2_worse) + '.')
        fig2_text.append(f'{fig2_family}: no best budget is selected; every available matched budget is shown.')
    fig2_text.append('')
fig2_text += [
    'Interpretation: a good stress fit can still predict poor energy transfer.',
    'For ideal incompressible 2D flow, the first-gradient stress has zero contraction with S;',
    'near-zero first-order Pi_tau is therefore expected. This does not imply zero Pi_phi.',
    'Prescribed second order means gradient + Hessian. Discovered residuals can contain other terms.',
    'A fitted coefficient minimizes its weak-form regression objective, not these pointwise metrics.',
]

# Fresh scratch output, regardless of where the older discovery archive lives.
fig2_output_root = Path(fig2_config.get('output_root',
    '/storage/scratch1/2/ashelestov3/rayleigh-benard/flux_analysis')).expanduser().resolve()
fig2_output = fig2_output_root / ('figure2_count_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '_' + uuid4().hex[:8])
fig2_output.mkdir(parents=True, exist_ok=False)
(fig2_output / 'source_flux.json').write_bytes(fig2_json_bytes)
shutil.copyfile(fig2_npz_path, fig2_output / 'source_flux_moments.npz')
if fig2_executed_source:
    (fig2_output / 'executed_analysis_cell.py').write_text(fig2_executed_source, encoding='utf-8')

fig2_plot_records, fig2_notices = [], []
fig2_rc = {
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'text.usetex': False, 'font.size': 13,
    'axes.linewidth': .75, 'axes.labelsize': 18, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'lines.linewidth': 1.15, 'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
}
try:
    with plt.rc_context(fig2_rc):
        for fig2_panel, fig2_target in zip(('a', 'b'), ('tau', 'phi')):
            fig2_kmax = max(k for groups in fig2_sequences.values() for seq in groups.values() for k in seq)
            fig2_x = np.arange(fig2_kmax+1)
            fig2_curves = [
                ('theoretical_sequence', fig2_sequences[fig2_target]['theoretical'], False),
                ('prescribed_second', fig2_prescribed[fig2_target][2], True),
                ('fitted_baseline', fig2_sequences[fig2_target]['fitted'][0], True),
                ('fitted_sequence', fig2_sequences[fig2_target]['fitted'], False),
                ('prescribed_first', fig2_prescribed[fig2_target][1], True),
            ]
            fig2_figure, fig2_axes = plt.subplots(3, 1, figsize=(4.8, 9.5), sharex=True)
            fig2_figure.subplots_adjust(left=.24, right=.97, bottom=.105, top=.81, hspace=.28)
            fig2_symbol = r'\tau' if fig2_target == 'tau' else r'\phi'
            fig2_ylabels = (rf'$\mathcal{{A}}({fig2_symbol})$',
                           rf'$\mathcal{{A}}(\Pi_{{{fig2_symbol}}})$',
                           rf'$\langle\Pi_{{{fig2_symbol}}}^{{\mathrm{{LES}}}}\rangle/\langle\Pi_{{{fig2_symbol}}}^{{\mathrm{{DNS}}}}\rangle$')
            for fig2_axis, fig2_metric, fig2_ylabel in zip(fig2_axes, fig2_metrics_to_plot, fig2_ylabels):
                fig2_plotted_values = []
                for fig2_key, fig2_data, fig2_constant in fig2_curves:
                    fig2_curve_rows = [fig2_data if fig2_constant else fig2_data.get(int(k)) for k in fig2_x]
                    fig2_y = [np.nan if row is None or row[fig2_metric] is None else row[fig2_metric] for row in fig2_curve_rows]
                    fig2_axis.plot(fig2_x, fig2_y, **fig2_styles[fig2_key],
                        marker='o' if fig2_config.get('markers', False) and not fig2_constant else None, markersize=3)
                    fig2_plotted_values.extend(value for value in fig2_y if np.isfinite(value))
                    fig2_plot_records.append(dict(target=fig2_target, metric=fig2_metric,
                        curve=fig2_key, horizontal_reference=fig2_constant, x=fig2_x.tolist(),
                        y=[float(value) if np.isfinite(value) else None for value in fig2_y],
                        candidate_ids=[str(row['id']) if row is not None else None for row in fig2_curve_rows]))
                if fig2_metric == 'mean_flux_ratio':
                    fig2_limits = tuple(fig2_config.get('mean_ratio_ylim', (0, 3)))
                    if len(fig2_limits) != 2 or not np.isfinite(fig2_limits).all() or fig2_limits[0] >= fig2_limits[1]:
                        raise ValueError('mean_ratio_ylim must contain two finite increasing limits.')
                    fig2_lo, fig2_hi = map(float, fig2_limits)
                    fig2_ticks = [0, 1, 2, 3] if fig2_limits == (0, 3) else None
                else:
                    fig2_lo, fig2_hi = 0., 1.
                    fig2_ticks = [0, .25, .5, .75, 1]
                fig2_span = fig2_hi-fig2_lo
                if fig2_plotted_values and (min(fig2_plotted_values) < fig2_lo-.025*fig2_span
                                           or max(fig2_plotted_values) > fig2_hi+.025*fig2_span):
                    fig2_lo = min(fig2_lo, min(fig2_plotted_values))
                    fig2_hi = max(fig2_hi, max(fig2_plotted_values))
                    fig2_ticks = None
                    fig2_notices.append(f'Expanded {fig2_target} {fig2_metric} limits to retain all values.')
                fig2_padding = .025*(fig2_hi-fig2_lo)
                fig2_axis.set(ylim=(fig2_lo-fig2_padding, fig2_hi+fig2_padding),
                              xlim=(0, fig2_kmax), ylabel=fig2_ylabel)
                if fig2_ticks is not None:
                    fig2_axis.set_yticks(fig2_ticks)
                fig2_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
                fig2_axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
                fig2_axis.tick_params(direction='in', top=True, right=True, length=3.5, width=.75,
                                     labelbottom=True)
                fig2_axis.set_xlabel(r'$N_{\mathrm{corr}}$')
                fig2_axis.grid(False)
            fig2_handles, fig2_labels = fig2_axes[0].get_legend_handles_labels()
            fig2_figure.legend(fig2_handles, fig2_labels, loc='upper left', bbox_to_anchor=(.16, .956),
                              frameon=False, fontsize=10, handlelength=2.7, labelspacing=.35)
            fig2_figure.text(.6, .985, rf'${fig2_symbol},\quad \Delta/dx={fig2_delta_dx:g}$',
                             ha='center', va='top', fontsize=16)
            fig2_figure.text(.6, .02, f'({fig2_panel})', ha='center', fontsize=18)
            for fig2_extension in ('png', 'pdf', 'svg'):
                fig2_figure.savefig(fig2_output / f'figure2_{fig2_target}_vs_correction_count.{fig2_extension}', dpi=300)
            plt.show()
            plt.close(fig2_figure)

    fig2_csv_columns = ['id', 'target', 'family', 'label', 'model_order', 'correction_budget', 'plotted',
        *fig2_metrics_to_plot, 'closure_relative_l2_error', 'flux_relative_l2_error',
        'time_mean_flux_reference', 'time_mean_flux_model', 'model_ids']
    with (fig2_output / 'metrics.csv').open('w', newline='', encoding='utf-8') as fig2_stream:
        fig2_writer = csv.DictWriter(fig2_stream, fieldnames=fig2_csv_columns)
        fig2_writer.writeheader()
        for fig2_row in fig2_rows:
            fig2_record = {key: fig2_row.get(key) for key in fig2_csv_columns}
            fig2_record['model_ids'] = json.dumps(fig2_row['model_ids'], ensure_ascii=False)
            fig2_writer.writerow(fig2_record)
    fig2_text.extend(fig2_notices)
    fig2_analysis_text = '\n'.join(fig2_text) + '\n'
    (fig2_output / 'analysis.txt').write_text(fig2_analysis_text, encoding='utf-8')
    fig2_manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_archive=str(fig2_input), source_flux_sha256=hashlib.sha256(fig2_json_bytes).hexdigest(),
        source_moments_sha256=hashlib.sha256(fig2_npz_path.read_bytes()).hexdigest(),
        executed_cell_sha256=hashlib.sha256(fig2_executed_source.encode('utf-8')).hexdigest() if fig2_executed_source else None,
        measurements=fig2_report['definitions'], delta_dx=fig2_delta_dx,
        x_axis='requested residual correction budget at fixed filter width; matched trace/STF budgets',
        reference='local SGS paper Figure 2(a), equations (3.8)-(3.9); x-axis intentionally adapted',
        mean_ratio_ylim=list(fig2_config.get('mean_ratio_ylim', (0, 3))),
        plot_settings=dict(markers=bool(fig2_config.get('markers', False)),
            figure_inches=[4.8, 9.5], rc=fig2_rc),
        colors_and_lines=fig2_styles, plotted_curves=fig2_plot_records, notices=fig2_notices,
        validation='archived moment/report consistency checked; not independent scientific validation')
    (fig2_output / 'analysis.json').write_text(json.dumps(fig2_manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    (fig2_output / 'COMPLETE').write_text(datetime.now(timezone.utc).isoformat(), encoding='utf-8')
except BaseException:
    (fig2_output / 'FAILED').write_text(traceback.format_exc(), encoding='utf-8')
    raise

print(fig2_analysis_text)
print('Paper-style analysis and figures saved to:', fig2_output)
