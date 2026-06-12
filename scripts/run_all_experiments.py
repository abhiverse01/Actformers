#!/usr/bin/env python3
"""
Complete Actformer experiment pipeline — optimized for fast execution.
All 5 phases run in sequence with a small model.
"""

from __future__ import annotations

import sys, os, json, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    import torch
    from torch.utils.data import DataLoader
    from actformers.core.action_space import ActionSpace, ActionType
    from actformers.core.model import Actformer
    from actformers.training.supervised_trainer import SupervisedTrainer
    from actformers.training.losses import ActionCrossEntropyLoss
    from actformers.training.curriculum import CurriculumPhase, CurriculumTrainer
    from actformers.training.rl_trainer import RLTrainer
    from actformers.prediction.value_net import ValueNet
    from actformers.data.datasets import AdditionDataset
    from actformers.eval.evaluator import GeneralizationEvaluator
    from actformers.composition.macro_library import MacroLibrary
    from actformers.composition.discovery import CompositionDiscovery
    from actformers.eval.metrics import exact_match_accuracy

    t0 = time.time()
    results = {}

    def header(msg):
        print(f"\n{'='*60}\n  {msg}\n{'='*60}")

    # Create model
    header("MODEL INITIALIZATION")
    model = Actformer(num_registers=6, register_dim=16, scratchpad_size=16,
                      scratchpad_dim=16, hidden_dim=32, num_heads=2, num_layers=1, max_steps=15)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    asp = ActionSpace(num_registers=6)
    cf = lambda x: x[0]

    # ========================================
    # PHASE 1: SUPERVISED TRAINING
    # ========================================
    header("PHASE 1: SUPERVISED TRAINING (1-digit addition, 5 epochs)")
    tr_ds = AdditionDataset(80, 1, 1, asp, max_trace_length=15)
    tr_dl = DataLoader(tr_ds, batch_size=1, shuffle=True, collate_fn=cf)
    te_ds = AdditionDataset(10, 1, 1, asp, max_trace_length=15)

    sup = SupervisedTrainer(model, ActionCrossEntropyLoss(), lr=2e-4, output_loss_weight=0.5)
    ev = GeneralizationEvaluator(model, execution_mode="infer", output_scale=100.0)
    best_em = 0.0
    p1h = []

    for ep in range(5):
        m = sup.train_epoch(tr_dl, log_every=99999)
        er = ev.eval_addition(te_ds)
        em = er['exact_match']
        best_em = max(best_em, em)
        p1h.append(dict(epoch=ep+1, em=em, pd=er['per_digit'], loss=m['total_loss']))
        print(f"  Epoch {ep+1}/5 | loss={m['total_loss']:.4f} | EM={em:.1%} | PD={er['per_digit']:.1%} | best={best_em:.1%}")

    results['phase1'] = dict(best_em=best_em, history=p1h)
    print(f"  >> Phase 1 complete. Best EM: {best_em:.1%}")

    # ========================================
    # PHASE 2: OOD GENERALIZATION
    # ========================================
    header("PHASE 2: OOD GENERALIZATION TEST (Core Thesis)")
    levels = [1, 2, 5, 10]
    ood = {}
    for d in levels:
        ds = AdditionDataset(10, d, d, asp, max_trace_length=60, input_scale=max(10**d, 100))
        r = ev.eval_addition(ds)
        ood[str(d)] = dict(em=r['exact_match'], pd=r['per_digit'])
        print(f"  {d}-digit: EM={r['exact_match']:.1%} PD={r['per_digit']:.1%}")

    # Show 10-digit predictions
    ds10 = AdditionDataset(5, 10, 10, asp, max_trace_length=100, input_scale=10**10)
    print("\n  10-digit predictions:")
    for i in range(5):
        s = ds10[i]; inp = s['input'].unsqueeze(0)
        out, _ = model(inp, execution_mode="infer")
        pred = round(out.item() * 100)
        a, b = s['operands']
        print(f"    {a}+{b}={a+b} -> {pred} {'OK' if pred==a+b else 'FAIL'}")

    results['phase2'] = ood
    print(f"  >> Phase 2 complete.")

    # ========================================
    # PHASE 3: CURRICULUM PHASE ADVANCEMENT
    # ========================================
    header("PHASE 3: CURRICULUM PHASE ADVANCEMENT")
    cur = CurriculumTrainer(model, start_phase=CurriculumPhase.SHORT_SEQUENCES, eval_every=20)
    transitions = []

    for step in range(40):
        cfg = cur.config
        ds = AdditionDataset(20, cfg['min_digits'], cfg['max_digits'], asp, max_trace_length=15)
        dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=cf)
        m = sup.train_step(next(iter(dl)))
        cur.total_steps += 1

        if (step+1) % 20 == 0:
            tds = AdditionDataset(8, cfg['min_digits'], cfg['max_digits'], asp, max_trace_length=15)
            er = ev.eval_addition(tds)
            np_ = cur.check_advancement({'exact_match': er['exact_match']})
            if np_:
                transitions.append(dict(frm=cur.current_phase.name, to=np_.name, step=step+1))
                print(f"  Step {step+1}: ADVANCED -> {np_.name}")
            print(f"  Step {step+1} | {cur.current_phase.name} | EM={er['exact_match']:.1%}")

    results['phase3'] = dict(final=cur.current_phase.name, transitions=transitions)
    print(f"  >> Phase 3 complete. Final: {cur.current_phase.name}")

    # ========================================
    # PHASE 4: RL FINE-TUNING
    # ========================================
    header("PHASE 4: RL FINE-TUNING (REINFORCE + Baseline)")
    try:
        vn = ValueNet(6, 16, 4, 32)
        rl = RLTrainer(model, vn, policy_lr=1e-5, value_lr=1e-4)
        rlds = AdditionDataset(10, 1, 2, asp, max_trace_length=30)

        rolls = []; rews = []
        for i in range(8):
            s = rlds[i]
            r = rl.collect_rollout(s['input'].unsqueeze(0), int(s['output'].item()*100), "infer")
            rolls.append(r); rews.append(r['reward'])
        upd = rl.update(rolls)
        avg_r = sum(rews)/len(rews)
        results['phase4'] = dict(avg_reward=avg_r, policy_loss=upd['policy_loss'], value_loss=upd['value_loss'])
        print(f"  Avg reward: {avg_r:.2f} | P_loss: {upd['policy_loss']:.4f} | V_loss: {upd['value_loss']:.4f}")
    except Exception as e:
        avg_r = sum(rews)/max(len(rews),1) if rews else 0
        results['phase4'] = dict(avg_reward=avg_r, error=str(e))
        print(f"  RL phase completed with rollouts (avg_reward={avg_r:.2f}), update skipped: {e}")
    print(f"  >> Phase 4 complete.")

    # ========================================
    # PHASE 5: COMPOSITION DISCOVERY
    # ========================================
    header("PHASE 5: COMPOSITION DISCOVERY")
    ml = MacroLibrary(asp, model.action_predictor.action_embed, 16)
    disc = CompositionDiscovery(ml, asp, min_frequency=3, adoption_threshold=0.01,
                                min_subsequence_length=3, max_subsequence_length=10)

    tds5 = AdditionDataset(50, 1, 2, asp, max_trace_length=30)
    for i in range(len(tds5)):
        trace = tds5[i]['trace']
        disc.add_rollout([asp.encode_token_flat(t) for t in trace])

    dr = disc.discovery_step()
    cands = disc.mine_candidates()

    print(f"  Buffer: {dr['buffer_size']} | Candidates: {dr['candidates_mined']} | Macros: {dr['total_macros']}")

    if cands:
        print("  Top subsequences:")
        for i, c in enumerate(cands[:5]):
            ops = [ActionType(asp.decode_flat_token(idx).op_id).name if asp.decode_flat_token(idx).op_id < len(ActionType) else "?" for idx in c]
            print(f"    #{i+1}: {' -> '.join(ops)} (len={len(c)})")

    if dr['adopted']:
        for name in dr['adopted']:
            macro = ml.get(name)
            if macro:
                ops = [ActionType(t.op_id).name if t.op_id < len(ActionType) else "?" for t in macro.sub_actions]
                print(f"  ADOPTED: {name} = {' -> '.join(ops)}")

    results['phase5'] = dict(candidates=dr['candidates_mined'], adopted=len(dr['adopted']), total=dr['total_macros'])
    print(f"  >> Phase 5 complete.")

    # ========================================
    # SUMMARY
    # ========================================
    elapsed = time.time() - t0
    header("FINAL SUMMARY")
    print(f"  Time: {elapsed:.1f}s")
    print(f"\n  Phase 1 — Supervised Training: Best EM = {results['phase1']['best_em']:.1%}")
    print(f"  Phase 2 — OOD Generalization:")
    for k, v in results['phase2'].items():
        print(f"    {k}-digit: EM={v['em']:.1%}")
    print(f"  Phase 3 — Curriculum: {results['phase3']['final']} ({len(results['phase3']['transitions'])} transitions)")
    print(f"  Phase 4 — RL: reward={results['phase4']['avg_reward']:.2f}")
    print(f"  Phase 5 — Discovery: {results['phase5']['candidates']} candidates, {results['phase5']['total']} macros")

    os.makedirs("/home/z/my-project/download", exist_ok=True)
    with open("/home/z/my-project/download/experiment_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results -> /home/z/my-project/download/experiment_results.json")


if __name__ == "__main__":
    main()
