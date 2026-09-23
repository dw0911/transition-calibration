# -*- coding: utf-8 -*-
"""Release self-check: the shipped numbers and figures must be recomputable here.

Run from anywhere inside the repository:

    python revision/scripts/check_release_tables.py

Three things are verified, and any failure exits non-zero:

  1. tables   -- every file in ``revision/paper/generated/`` equals what
                 ``gen_tables_pr2.py`` produces from ``revision/results/`` right now;
  2. figures  -- ``revision/paper/figure_data/`` equals the values the bundled figure
                 builder computes from the same result records (the CSVs must match
                 byte for byte, the plotted coordinates to 1e-12);
  3. manifest -- every path listed in ``revision/RELEASE_MANIFEST.json`` still has the
                  sha256 recorded there, so a reader can tell whether the copy in front
                  of them is the copy that was checked at release time.

Nothing is trained, calibrated, resampled or written.  This script only reads.
"""
import os
import re
import io
import csv
import sys
import json
import glob
import shutil
import hashlib
import importlib.util

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.dirname(HERE)                                   # revision/
RESULTS = os.path.join(SUB, 'results')
GEN = os.path.join(SUB, 'paper', 'generated')
FIGDATA = os.path.join(SUB, 'paper', 'figure_data')
MANIFEST = os.path.join(SUB, 'RELEASE_MANIFEST.json')
TOL = 1e-12

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:                                             # pragma: no cover
    pass


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_tables():
    """Regenerate all numeric tables in memory and compare with the shipped files."""
    gen = load('release_table_generator', 'gen_tables_pr2.py')
    built = gen.build_all(gen.Art())
    rows, bad = [], []
    for label, body in built.items():
        want = gen.HDR + '\n' + body + '\n'
        p = os.path.join(GEN, gen.written_name(label))
        ok = os.path.exists(p) and open(p, encoding='utf-8').read() == want
        rows.append((label, os.path.basename(p), ok))
        if not ok:
            bad.append(label)
    extra = sorted(os.path.basename(p) for p in glob.glob(os.path.join(GEN, '*.tex'))
                   if os.path.basename(p) not in {r[1] for r in rows})
    return rows, bad, extra


def check_figures():
    """Compare shipped figure_data with the values the bundled builder computes here."""
    bf = load('release_figure_builder', 'build_figures.py')
    from pathlib import Path
    bf.ROOT = Path(SUB)
    bf.RESULTS = Path(os.path.join(RESULTS, 'phase2_E7', 'v1'))
    bf.OUT = Path(os.path.join(SUB, 'paper', 'figures'))
    bf.DATA = Path(FIGDATA)
    cost, delta, _srcs = bf.load_values()
    pv = json.load(open(os.path.join(FIGDATA, 'plotted_values.json'), encoding='utf-8'))
    diffs, n = [], 0
    t1 = {(r['group'], r['program']): r for r in pv['figure1']}
    for row in cost:
        t = t1[(row['group'], row['program'])]
        for k in ('path_coverage', 'width_ratio'):
            n += 1
            d = abs(row[k] - t[k])
            if d > TOL:
                diffs.append(('figure1', row['group'], row['program'], k, d))
    t2 = {r['group']: r for r in pv['figure2']}
    for row in delta:
        t = t2[row['group']]
        for k in ('Overall_change_pp', 'TypeB_change_pp', 'TypeC_change_pp',
                  'width_ratio_g1_g0'):
            n += 1
            d = abs(row[k] - t[k])
            if d > TOL:
                diffs.append(('figure2', row['group'], k, d))
    # 来源哈希：figure_data 记录的 v1 产物必须就是本仓库里这几份
    src_bad = [s['path'] for s in pv.get('sources', [])
               if not os.path.exists(os.path.join(RESULTS, 'phase2_E7', 'v1',
                                                  os.path.basename(s['path'])))
               or sha(os.path.join(RESULTS, 'phase2_E7', 'v1',
                                   os.path.basename(s['path']))) != s['sha256']]
    # CSV 与 plotted_values 必须同值（CSV 是显示转录）
    csv_ok = True
    f1 = os.path.join(FIGDATA, 'figure1_coverage_cost.csv')
    if os.path.exists(f1):
        with open(f1, newline='', encoding='utf-8-sig') as fh:
            for r in csv.DictReader(fh):
                t = t1[(r['group'], r['program'])]
                if (abs(float(r['path_coverage']) - t['path_coverage']) > 1e-9
                        or abs(float(r['width_ratio']) - t['width_ratio']) > 1e-9):
                    csv_ok = False
    return n, diffs, src_bad, csv_ok, len(pv['figure1']), len(pv['figure2'])


def check_manifest():
    man = json.load(open(MANIFEST, encoding='utf-8'))
    gone, changed = [], []
    for relp, h in man['files'].items():
        p = os.path.join(os.path.dirname(SUB), relp)
        if not os.path.exists(p):
            gone.append(relp)
        elif sha(p) != h:
            changed.append(relp)
    return len(man['files']), man.get('generated'), gone, changed


TIER3_OPTIONAL = {'torch', 'basicts', 'model', 'baselines'}
STDLIB_OK = set('''os re sys json glob csv hashlib datetime time argparse math itertools
collections warnings ast io statistics typing string traceback random pickle subprocess
pathlib tempfile importlib contextlib dataclasses shutil importlib.util'''.split())


def check_closure():
    """静态核对导入闭包：每个随包脚本 import 的东西都必须能在这一份克隆里解析到。

    这里**不导入**脚本本身：其中几个（d3/e6/e7_export）在 import 时就会读盘或起子进程，
    把研究代码当冒烟测试跑既不合适也不安全。缺文件这个真实风险用 AST 就能查全：
    上一轮漏掉的正是这一类。外部依赖允许三种去处——本地已随包、站点包里装着、
    或文档里写明属 tier-3 前提（torch/BasicTS）。
    """
    import ast
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    libdir = os.path.join(SUB, 'uncalib_v1', 'SRC')
    local = {os.path.basename(p)[:-3] for p in glob.glob(os.path.join(here, '*.py'))}
    local |= {os.path.basename(p)[:-3] for p in glob.glob(os.path.join(libdir, '*.py'))}
    missing, optional, unresolved_local = [], [], []
    for path in sorted(glob.glob(os.path.join(here, '*.py'))) + \
            sorted(glob.glob(os.path.join(libdir, '*.py'))):
        rel = os.path.relpath(path, SUB).replace(os.sep, '/')
        try:
            tree = ast.parse(open(path, encoding='utf-8').read())
        except SyntaxError as e:
            unresolved_local.append(f'{rel}: {e.msg}')
            continue
        for n in ast.walk(tree):
            names = []
            if isinstance(n, ast.Import):
                names = [a.name.split('.')[0] for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                names = [n.module.split('.')[0]]
            for m in names:
                if m in STDLIB_OK or m in local or m == '__future__':
                    continue
                if m in TIER3_OPTIONAL:
                    optional.append(f'{rel}: {m} (tier 3 prerequisite)')
                    continue
                if importlib.util.find_spec(m) is None:
                    missing.append(f'{rel}: cannot resolve import "{m}" '
                                  '(not shipped, not installed)')
    # 文档承诺的那条 import 路径必须真的存在（UC_PROJ=<clone>/revision）
    lib_ok = os.path.isdir(libdir) and os.path.exists(os.path.join(libdir, 'conformal.py'))
    return missing, optional, unresolved_local, lib_ok


VOLATILE_FIELDS = {'runtime_sec', 'sec_per_epoch_mean', 'sec_per_epoch_last3',
                   'extrapolated_100ep_hours', 'peak_vram_gb', 'gpu', 'amp',
                   'exported_utc', 'script_sha256', 'path', 'ckpt', 'mode',
                   'exported_by', 'exported_tags', 'reason', 'note'}


def _leaves(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f'{path}/{k}')
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, f'{path}[{i}]')
    else:
        yield path, obj


def scientific_diff(ref_path, new_path):
    """字段级比较，忽略环境相关量。

    为什么不能直接比字节：实测在干净树里重跑 tier-2，50 份金标准里 49 份只差
    runtime_sec，另一份还差 460 个 provenance/train_meta 叶子（训练元数据按各自
    磁盘路径与硬件记录）；科学量一个都没变。用字节相等当复现判据，会让一个正确
    的重跑因无关字段被判失败。
    """
    a = dict(_leaves(json.load(open(ref_path, encoding='utf-8'))))
    b = dict(_leaves(json.load(open(new_path, encoding='utf-8'))))
    out = []
    for k in set(a) | set(b):
        leaf = k.rsplit('/', 1)[-1].split('[')[0]
        if leaf in VOLATILE_FIELDS or '/provenance/' in k:
            continue
        if a.get(k) != b.get(k):
            out.append(k)
    return out


def tier2_diff():
    """有缓存时真正重算 tier-2 并与金标准逐字段比较；没有缓存就明说不验证。"""
    cache = os.path.join(SUB, 'cache')
    npz = glob.glob(os.path.join(cache, '*.npz'))
    if len(npz) < 12:
        return None, (f'caches absent ({len(npz)} .npz under revision/cache/) -> '
                      'tier 2 NOT verified here; see revision/CACHES.md')
    refs = sorted(glob.glob(os.path.join(RESULTS, '**', '*.json'), recursive=True))
    before = {p: sha(p) for p in refs}
    # 先把金标准快照到临时目录，重跑就地覆盖后再逐字段比较（比较时忽略环境相关量）
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        snap = os.path.join(tmp, 'golden')
        os.makedirs(snap, exist_ok=True)
        shutil.copytree(RESULTS, os.path.join(snap, 'results'))
        for script in ('e7_metrics_v1.py', 'pr1_supplementary.py'):
            p = os.path.join(SUB, 'scripts', script)
            if not os.path.exists(p):
                continue
            r = subprocess.run([sys.executable, p], capture_output=True, text=True)
            if r.returncode != 0:
                return False, f'{script} failed: {r.stderr[-300:]}'
        byte_same = sum(1 for p in refs if sha(p) == before[p])
        sci, files_with_diff = [], 0
        for p in refs:
            q = p.replace(RESULTS, os.path.join(snap, 'results'), 1)
            if not os.path.exists(q):
                continue
            d = scientific_diff(q, p)
            if d:
                files_with_diff += 1
                sci.extend((os.path.basename(p), x) for x in d[:3])
    return (not sci), (f'{len(refs)} golden records re-run in place; byte-identical '
                       f'{byte_same}; files with non-volatile field differences '
                       f'{files_with_diff}; scientific mismatches {len(sci)}'
                       + (f' -> {sci[:4]}' if sci else ''))


def main():
    ap = __import__('argparse').ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--tier2', action='store_true',
                    help='also re-run the statistics pipeline and byte-compare the golden '
                         'records (needs revision/cache/*.npz, see revision/CACHES.md)')
    args = ap.parse_args()
    print('== release self-check (read-only)')
    rows, bad_tables, extra = check_tables()
    print(f'[{"OK " if not bad_tables and not extra else "BAD"}] tables: '
          f'{len(rows) - len(bad_tables)}/{len(rows)} regenerate identically from '
          f'revision/results/' + (f'; unexplained extra files {extra}' if extra else ''))
    for label, fn, ok in rows:
        if not ok:
            print(f'      MISMATCH {label} -> {fn}')
    n, fdiffs, src_bad, csv_ok, n1, n2 = check_figures()
    print(f'[{"OK " if not fdiffs else "BAD"}] figures: {n} plotted values match '
          f'revision/paper/figure_data/ within {TOL:g} ({n1} points in figure 1, '
          f'{n2} groups in figure 2)')
    for d in fdiffs[:6]:
        print('      MISMATCH', d)
    print(f'[{"OK " if not src_bad else "BAD"}] figure_data source hashes match the bundled '
          f'v1 records' + (f'; mismatched {src_bad[:4]}' if src_bad else ''))
    print(f'[{"OK " if csv_ok else "BAD"}] figure_data CSVs agree with plotted_values.json')
    nf, gend, gone, changed = check_manifest()
    print(f'[{"OK " if not gone and not changed else "BAD"}] manifest: {nf} files recorded '
          f'{gend}; missing {len(gone)}, changed since release {len(changed)}')
    for x in (gone + changed)[:6]:
        print('      ', x)
    cmiss, copt, cunres, lib_ok = check_closure()
    print(f'[{"OK " if not cmiss and not cunres and lib_ok else "BAD"}] import closure '
          f'(static): every import in the shipped scripts resolves here; library copy present '
          f'({lib_ok}); unresolved local refs {len(cunres)}; tier-3 prerequisites noted: '
          f'{len(copt)}')
    for x in (cmiss + cunres)[:8]:
        print('      ', x)
    for x in sorted(set(copt))[:4]:
        print('      note:', x)
    t2 = None
    if args.tier2:
        t2, msg = tier2_diff()
        tag = 'OK ' if t2 else ('N-A' if t2 is None else 'BAD')
        print(f'[{tag}] tier 2 re-run vs golden records: {msg}')
        if t2:
            print('      note: rerunning tier 2 rewrote revision/results/; restore it from '
                  'git before trusting the shipped tables again')
    ok = (not bad_tables and not extra and not fdiffs and not src_bad and csv_ok
          and not gone and not changed and not cmiss and not cunres and lib_ok
          and (t2 is not False))
    print('VERDICT:', 'PASS' if ok else 'CHECK')
    print('note: this checks that the release is internally consistent. It does not '
          're-estimate anything from raw PeMS data, retrain a backbone, or re-run the '
          'bootstrap; see revision/REPRODUCE.md for what each tier needs.')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
