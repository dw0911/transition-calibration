# -*- coding: utf-8 -*-
"""E7 新增 checkpoint 的预测导出（与 export_preds.py 同口径，含 PEMS08 test 闸门）。

与 S0 的 `export_preds.py` 的关系：
  - 推理协议逐行一致（IL/OL=12、purge=23、四段 int(T·r)、per-node train-only 归一化、
    STAEformer 路径 AMP 开 + bs=64 冻结，见 protocol.yaml :: point_forecast_inference）。
  - 新增两件事：
      1) 支持 STID backbone（e7_train_stid.py 的同口径推理：输入 ch0 per-node 归一化、
         ch1/ch2 原值、tod/dow 由 ch1/ch2 还原为整数索引、输出逆变换；AMP 关，与训练一致）
      2) **PEMS08 test 闸门**：除非 `plan/pems08_unseal_signoff.json` 存在且逐条引用
         `plan/pems08_unseal_checklist.yaml` 的条目并给出证据路径，否则拒绝导出 PEMS08 test，
         只导 sel/cal。闸门是硬编码的，不靠自觉。
  - 缓存命名：STAEformer 沿用 `{ds}_s{seed}_{tag}.npz`（与 S0 同构，新 config 不冲突）；
    STID 用 `{ds}_stid_s{seed}_{tag}.npz`，避免与 STAEformer 缓存混淆。
  - dtype 口径与 S0 一致：r_in_max float64，其余 float32（verify_cache 的容差政策依赖此口径）。

用法:
  python e7_export.py PEMS03 123 stae            # sel+cal+test（PEMS03 test 不受闸门限制）
  python e7_export.py PEMS08 42 stae             # 无签署时只导 sel+cal，test 被闸门拒绝并记录
  python e7_export.py PEMS08 42 stid --tags sel,cal
"""
import os
import sys
import json
import time
import argparse

_PKG = os.path.dirname(os.path.abspath(__file__))
PROJ = os.environ.get('UC_PROJ') or os.path.abspath(
    os.path.join(_PKG, os.pardir, os.pardir))   # parent of revision_v1; override: UC_PROJ
BASICTS = os.path.join(PROJ, 'BasicTS058')
UC = os.path.join(PROJ, 'uncalib_v1')
ROOT = os.path.join(PROJ, 'revision_v1')
CACHE = os.path.join(ROOT, 'cache')
CKPT = os.path.join(ROOT, 'checkpoints')
SIGNOFF = os.path.join(ROOT, 'plan', 'pems08_unseal_signoff.json')
CHECKLIST = os.path.join(ROOT, 'plan', 'pems08_unseal_checklist.yaml')

sys.path.insert(0, os.path.join(UC, 'SRC'))
sys.path.insert(0, BASICTS)
sys.path.insert(0, os.path.join(BASICTS, 'baselines', 'STAEformer'))
sys.path.insert(0, os.path.join(PROJ, 'baselines', 'STID'))
os.chdir(BASICTS)

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
import taxonomy as tx                                 # noqa: E402
import severity as sev                                # noqa: E402
from export_preds import (load_full, build_model, infer_range, window_features,
                          IL, OL, PURGE, ALPHA)       # noqa: E402

GATE_LOG = os.path.join(ROOT, 'logs', 'pems08_gate.jsonl')


def gate_decision(dataset, want_test):
    """返回 (allow_test, reason)。PEMS08 之外不受限。"""
    if dataset != 'PEMS08' or not want_test:
        return (True, 'not PEMS08 / test not requested') if dataset != 'PEMS08' \
            else (True, 'test not requested')
    if not os.path.exists(SIGNOFF):
        return False, f'missing signoff: {os.path.relpath(SIGNOFF, ROOT)}'
    try:
        sig = json.load(open(SIGNOFF, encoding='utf-8'))
    except Exception as e:                                    # noqa: BLE001
        return False, f'signoff unreadable: {e}'
    if not os.path.exists(CHECKLIST):
        return False, 'missing checklist yaml'
    import re
    txt = open(CHECKLIST, encoding='utf-8').read()
    item_ids = re.findall(r'^\s*-\s*id:\s*(U\d+)', txt, re.M)
    covered = set(sig.get('items', {}).keys())
    missing = [i for i in item_ids if i not in covered]
    if missing:
        return False, f'signoff does not cover checklist items: {missing}'
    unev = [i for i, v in sig.get('items', {}).items()
            if not (isinstance(v, dict) and v.get('evidence'))]
    if unev:
        return False, f'signoff items without evidence path: {unev}'
    if not sig.get('signed_by') or not sig.get('date'):
        return False, 'signoff lacks signed_by / date'
    return True, f'signoff ok ({sig.get("signed_by")}, {sig.get("date")})'


@torch.no_grad()
def infer_range_stid(model, full, g_start, g_end, mean_t, std_t, N, tod_all, dow_all, bs=64):
    """STID 同口径推理：输入 ch0 per-node 归一化、ch1/ch2 原值；AMP 关（与训练一致）。"""
    gs = np.arange(g_start, g_end)
    preds = []
    for s in range(0, len(gs), bs):
        gi = gs[s:min(s + bs, len(gs))]
        x = full[gi[:, None] + np.arange(IL)].copy()
        xt = torch.tensor(x, dtype=torch.float32, device='cuda')
        xt[..., 0] = (xt[..., 0] - mean_t) / std_t
        tod = torch.tensor(tod_all[gi], dtype=torch.long, device='cuda')
        dow = torch.tensor(dow_all[gi], dtype=torch.long, device='cuda')
        out = model(xt, tod, dow)                              # (B,12,N)
        preds.append((out.float() * std_t + mean_t).cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    yj = gs[:, None] + np.arange(IL, IL + OL)
    true = full[yj][..., 0]
    return pred.astype(np.float32), true.astype(np.float32)


def run(dataset, seed, backbone, tags):
    t0 = time.time()
    full, shape = load_full(dataset)
    T = shape[0]
    N = shape[1]
    train_end, select_end, calib_end = int(T * 0.6), int(T * 0.7), int(T * 0.8)
    ranges = {'sel': (train_end, select_end - PURGE),
              'cal': (select_end, calib_end - PURGE),
              'test': (calib_end, T - PURGE)}
    want_test = 'test' in tags
    allow_test, reason = gate_decision(dataset, want_test)
    if want_test and not allow_test:
        tags = [t for t in tags if t != 'test']
    os.makedirs(os.path.dirname(GATE_LOG), exist_ok=True)
    with open(GATE_LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                            'dataset': dataset, 'seed': seed, 'backbone': backbone,
                            'wanted_test': want_test, 'allow_test': allow_test,
                            'reason': reason}, ensure_ascii=False) + '\n')
    print(f'=== E7-export {dataset} s{seed} {backbone} T={T} N={N} tags={tags}', flush=True)
    print(f'    PEMS08-test gate: allow={allow_test} reason={reason}', flush=True)

    flow = full[:, :, 0]
    tau = tx.fit_per_node_q95(full[:train_end, :, 0])
    stats = sev.fit_severity_stats(full[:train_end, :, 0])
    s_series = sev.severity_series(flow, stats)
    mean = full[:train_end, :, 0].mean(axis=0)
    std = full[:train_end, :, 0].std(axis=0); std[std == 0] = 1.0
    mean_t = torch.tensor(mean, dtype=torch.float32, device='cuda')
    std_t = torch.tensor(std, dtype=torch.float32, device='cuda')
    tod_all = np.rint(full[:, 0, 1] * 288).astype(np.int64).clip(0, 287)
    dow_all = np.rint(full[:, 0, 2] * 7).astype(np.int64).clip(0, 6)

    if backbone == 'stae':
        ck = os.path.join(CKPT, f'{dataset}_e7_4split_seed{seed}', 'best.pt')
        model = build_model(N).cuda()
        infer = lambda a, b: infer_range(model, full, a, b, mean_t, std_t, N)   # noqa: E731
        amp = True
    else:
        from model import STID
        ck = os.path.join(CKPT, f'{dataset}_stid_e7_4split_seed{seed}', 'best.pt')
        model = STID(num_nodes=N, input_dim=3, his_steps=IL, pred_steps=OL).cuda()
        infer = lambda a, b: infer_range_stid(model, full, a, b, mean_t, std_t, N,   # noqa: E731
                                              tod_all, dow_all)
        amp = False
    sd = torch.load(ck, map_location='cpu', weights_only=False)
    model.load_state_dict(sd['model_state_dict'] if isinstance(sd, dict)
                          and 'model_state_dict' in sd else sd)
    model.eval()
    print(f'    ckpt={os.path.relpath(ck, ROOT)} best_mae={sd.get("best_mae")} '
          f'best_epoch={sd.get("best_epoch")} amp={amp}', flush=True)

    lab_test = tx.classify_from_flow(flow, np.arange(*ranges['test']), tau, IL, OL)
    suffix = '' if backbone == 'stae' else '_stid'
    meta = {'dataset': dataset, 'seed': seed, 'backbone': backbone, 'alpha': ALPHA,
            'IL': IL, 'OL': OL, 'purge': PURGE,
            'boundaries': [train_end, select_end, calib_end, T],
            'exported_tags': tags, 'test_gate': {'wanted': want_test, 'allowed': allow_test,
                                                 'reason': reason},
            'amp': amp, 'batch_size': 64, 'ckpt': os.path.relpath(ck, ROOT),
            'ckpt_best_mae': sd.get('best_mae'), 'ckpt_best_epoch': sd.get('best_epoch'),
            'exported_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    for tag in tags:
        a, b = ranges[tag]
        p, t = infer(a, b)
        g = np.arange(a, b)
        fw = window_features(s_series, g)
        flow_in = full[g[:, None] + np.arange(IL)][:, :, :, 0].mean(axis=1).astype(np.float32)
        pmag = np.abs(p).mean(axis=1).astype(np.float32)
        arrs = {'pred': p, 'true': t, 'flow_in': flow_in, 'pred_mag': pmag,
                'r_in_max': fw['r_in_max'].astype(np.float64),
                'argpos': fw['argpos'], 'lag': fw['lag'].astype(np.float32),
                'r_last3': fw['r_last3'].astype(np.float32),
                'r_recency': fw['r_recency'].astype(np.float32)}
        if tag == 'test':
            arrs['labels'] = lab_test
        fn = os.path.join(CACHE, f'{dataset}{suffix}_s{seed}_{tag}.npz')
        np.savez(fn, **arrs)
        print(f'  [{tag}] n={p.shape[0]} rawMAE={np.abs(p - t)[t > 0].mean():.4f} '
              f'-> {os.path.relpath(fn, ROOT)} ({time.time()-t0:.0f}s)', flush=True)
        del p, t, arrs
    json.dump(meta, open(os.path.join(CACHE, f'{dataset}{suffix}_s{seed}_export_meta.json'),
                         'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    torch.cuda.empty_cache()
    del model
    print(f'  done in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('seed', type=int)
    ap.add_argument('backbone', choices=['stae', 'stid'])
    ap.add_argument('--tags', default='sel,cal,test')
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    run(a.dataset, a.seed, a.backbone, [t.strip() for t in a.tags.split(',') if t.strip()])
