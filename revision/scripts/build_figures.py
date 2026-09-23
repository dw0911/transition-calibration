#!/usr/bin/env python3
"""Draw the two main-paper figures from fixed, full-precision result records.

No model fitting, interval calibration, bootstrap, or result-record update occurs.
Four independent plots form Figure 1; Figure 2 contains mean differences only.
Data and source hashes are exported alongside the figures for reproduction.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'reproduction/results/phase2_E7/v1'
OUT = ROOT / 'paper/figures'
DATA = ROOT / 'figure_data'
SEEDS = (42, 123, 456)
GROUPS = (
    ('PEMS03_stae', 'P03/A', 'PeMS03 / STAEformer'),
    ('PEMS04_stae', 'P04/A', 'PeMS04 / STAEformer'),
    ('PEMS08_stae', 'P08/A', 'PeMS08 / STAEformer'),
    ('PEMS08_stid', 'P08/I', 'PeMS08 / STID'),
)
PROGRAMS = (
    ('uncond', 'Pointwise', 'o', 35),
    ('path_global', 'Path maximum', 's', 39),
    ('bonferroni_aH', 'Bonferroni', '^', 46),
    ('nh_path', 'NH', 'D', 37),
    ('g0', r'Learned $g_0$', 'o', 78),
    ('g1', r'Learned $g_1$', 'x', 45),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_values() -> tuple[list[dict], list[dict], list[dict]]:
    cost, delta, sources = [], [], []
    for group, short, label in GROUPS:
        rows = []
        for seed in SEEDS:
            path = RESULTS / f'{group}_s{seed}.json'
            row = json.loads(path.read_text(encoding='utf-8'))
            if row.get('status') != 'COMPLETED':
                raise ValueError(f'Incomplete configuration: {path.name}')
            rows.append(row)
            sources.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
        for key, disp, _, _ in PROGRAMS:
            coverages = [r['metrics'][key]['V1']['path_cov'] for r in rows]
            ratios = [r['metrics'][key]['V1']['mean_width'] /
                      r['metrics']['uncond']['V1']['mean_width'] for r in rows]
            value = {'group': group, 'short_label': short, 'label': label,
                     'program': key, 'program_label': disp,
                     'path_coverage': mean(coverages), 'width_ratio': mean(ratios),
                     'n_seeds': len(SEEDS)}
            if not (0 <= value['path_coverage'] <= 1 and value['width_ratio'] > 0):
                raise ValueError(f'Invalid operating point: {value}')
            cost.append(value)
        one = {'group': group, 'short_label': short, 'label': label,
               'n_seeds': len(SEEDS)}
        for event in ('Overall', 'TypeB', 'TypeC'):
            changes = []
            for r in rows:
                m0 = r['metrics']['g0']['V1']
                m1 = r['metrics']['g1']['V1']
                changes.append(100 * ((m1['path_cov'] - m0['path_cov']) if event == 'Overall'
                                     else (m1['group_path_cov'][event] - m0['group_path_cov'][event])))
            one[f'{event}_change_pp'] = mean(changes)
        one['width_ratio_g1_g0'] = mean([
            r['metrics']['g1']['V1']['mean_width'] / r['metrics']['g0']['V1']['mean_width']
            for r in rows])
        delta.append(one)
    return cost, delta, sources


def export_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_chart(fig, stem: str) -> None:
    # PDF/SVG contain vector marks and text; PNG is a convenience preview only.
    fig.savefig(OUT / f'{stem}.pdf', metadata={'Creator': 'build_figures.py', 'CreationDate': None, 'ModDate': None})
    fig.savefig(OUT / f'{stem}.svg', metadata={'Date': None})
    fig.savefig(OUT / f'{stem}.png', dpi=400)


def draw(cost: list[dict], delta: list[dict]) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    # Use the standard Matplotlib color cycle, paired with distinguishable markers.
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'font.size': 9, 'mathtext.fontset': 'dejavuserif',
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.linewidth': 0.65})
    handles = None
    for index, (group, short, label) in enumerate(GROUPS):
        fig = plt.figure(figsize=(3.35, 2.65))
        ax = fig.add_axes([0.23, 0.21, 0.74, 0.65])
        h = []
        for key, name, marker, size in PROGRAMS:
            p = next(x for x in cost if x['group'] == group and x['program'] == key)
            scatter = ax.scatter(p['width_ratio'], p['path_coverage'], s=size,
                                 marker=marker, linewidths=1.25, alpha=0.95,
                                 label=name, zorder=5 if key == 'g1' else 3)
            h.append(scatter)
        handles = h
        ax.axhline(0.90, linestyle='--', linewidth=0.8, alpha=0.65, zorder=1)
        ax.set_xlim(0.90, 3.22)
        ax.set_ylim(0.54, 1.00)
        ax.set_xticks([1, 1.5, 2, 2.5, 3])
        ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlabel('Width ratio to pointwise', labelpad=4)
        ax.set_ylabel('Achieved path coverage', labelpad=5)
        ax.set_title(f'({chr(97 + index)}) {label}', fontsize=9.5, pad=8)
        ax.grid(True, linewidth=0.45, alpha=0.20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(width=0.65, length=3)
        save_chart(fig, f'figure1_panel_{short.replace("/", "")}')
        plt.close(fig)

    # Shared legend is a separate vector asset, not an additional data panel.
    legend_fig = plt.figure(figsize=(6.80, 0.46))
    legend_fig.legend(handles, [p[1] for p in PROGRAMS], loc='center', ncol=3,
                      frameon=False, fontsize=9, columnspacing=2.2, handletextpad=0.4)
    save_chart(legend_fig, 'figure1_legend')
    plt.close(legend_fig)

    fig = plt.figure(figsize=(6.60, 2.85))
    ax = fig.add_axes([0.16, 0.21, 0.80, 0.62])
    offsets = (-0.22, 0.0, 0.22)
    for event, marker, offset in zip(('Overall', 'TypeB', 'TypeC'), ('o', 's', '^'), offsets):
        xs = [d[f'{event}_change_pp'] for d in delta]
        ys = [i + offset for i in range(len(delta))]
        ax.scatter(xs, ys, marker=marker, s=38, linewidths=0.9, label=event, zorder=3)
        for x, y in zip(xs, ys):
            ax.annotate(f'{x:+.2f}', (x, y), xytext=(7, 0), textcoords='offset points',
                        va='center', ha='left', fontsize=8.3)
    ax.axvline(0, linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xlim(-4.55, 6.05)
    ax.set_ylim(3.50, -0.50)
    ax.set_xticks([-4, -2, 0, 2, 4, 6])
    ax.set_yticks(range(4), [x['short_label'] for x in delta])
    ax.set_xlabel(r'Path coverage change, $g_1-g_0$ (percentage points)', labelpad=5)
    ax.grid(axis='x', linewidth=0.5, alpha=0.22)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=7)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=3,
              frameon=False, columnspacing=2.5, handletextpad=0.4)
    save_chart(fig, 'figure2_severity_reallocation')
    plt.close(fig)


def compose_figure1() -> None:
    tex = r'''\documentclass[border=0pt]{standalone}
\usepackage{graphicx}
\begin{document}
\begin{minipage}{170mm}
\centering
\includegraphics[width=\linewidth]{figure1_legend.pdf}\\[-1mm]
\includegraphics[width=.493\linewidth]{figure1_panel_P03A.pdf}\hfill
\includegraphics[width=.493\linewidth]{figure1_panel_P04A.pdf}\\[-1mm]
\includegraphics[width=.493\linewidth]{figure1_panel_P08A.pdf}\hfill
\includegraphics[width=.493\linewidth]{figure1_panel_P08I.pdf}
\end{minipage}
\end{document}
'''
    path = OUT / 'figure1_coverage_cost.tex'
    path.write_text(tex, encoding='utf-8', newline='\n')
    engine = shutil.which('pdflatex')
    if engine is None:
        raise RuntimeError('pdflatex is needed to assemble the four-panel Figure 1.')
    done = subprocess.run([engine, '-interaction=nonstopmode', '-halt-on-error', path.name],
                          cwd=OUT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if done.returncode:
        raise RuntimeError(done.stdout[-5000:])
    # Preserve text and vector paths in the assembled publication PDF.
    import fitz
    with fitz.open(OUT / 'figure1_coverage_cost.pdf') as doc:
        doc[0].get_pixmap(dpi=400).save(OUT / 'figure1_coverage_cost.png')
        (OUT / 'figure1_coverage_cost.svg').write_text(doc[0].get_svg_image(text_as_path=False), encoding='utf-8')
    for suffix in ('.aux', '.log'):
        (OUT / ('figure1_coverage_cost' + suffix)).unlink(missing_ok=True)


def main() -> int:
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        DATA.mkdir(parents=True, exist_ok=True)
        cost, delta, sources = load_values()
        export_csv(DATA / 'figure1_coverage_cost.csv', cost)
        export_csv(DATA / 'figure2_severity_increment.csv', delta)
        payload = {'scope': 'Presentation only; fixed result records; no new inference.',
                   'aggregation': 'Per-configuration ratios/differences followed by an unweighted mean over seeds 42,123,456.',
                   'population': 'V1: twelve positive targets at the node-window pair.',
                   'uncertainty': 'Markers are descriptive means; no error bars or equal-coverage frontier is drawn.',
                   'sources': sources, 'figure1': cost, 'figure2': delta}
        (DATA / 'plotted_values.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        draw(cost, delta)
        compose_figure1()
        print('Built Figure 1 (four panels) and Figure 2 from 12 unchanged result records.')
        return 0
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        print(f'Figure build failed: {exc}', file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
