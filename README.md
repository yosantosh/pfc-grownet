# PFC-GrowNet — a growable neuroplastic net

A from-scratch neural net that **is born with 3 neurons and grows itself** —
new neurons (widen), new layers (deepen), new synapses (sprout) — committing
only the direction that most reduces held-out loss per added parameter, during
training *and* inference. PFC/basal-ganglia/dopamine-inspired: growable
dendritic posterior (L1, seed 3) + growable deep layer (L2, born empty),
gated working-memory stripes, surprise-driven ephemeral growth with a hybrid
commit gate. Baselines carry the same residual skip, so comparisons isolate
*growth*, not scaffolding.

## Layout
- `pfc_grownet.py` — the growable architecture (dendritic layer + synapse mask,
  BG gates, all growth ops, structural snapshot/restore)
- `growth_trial.py` — loss-driven direction selection: trial widen vs deepen vs
  sprout on probe data, commit winner by drop-per-parameter
- `baselines.py` — fixed-size MLP baselines (same I/O, same skip, same protocol)
- `dataset.py` — tiny compositional language (unseen-combo generalization test)
- `train_compare.py` — train GrowNet vs Fixed-Small vs Fixed-Large on tiny lang
- `infer_viz.py` — inference demo: generations, surprise growth, gate traces
- `shake_data.py` / `train_shake.py` / `infer_shake.py` — Tiny Shakespeare scale-up
- `tinyshakespeare.txt` — public-domain-ish tiny Shakespeare corpus (Karpathy's char-rnn)
- `plots*/` — loss curves, growth trajectories, generalization gaps, robustness

## Run (conda env `tf`)
```
conda run -n tf python dataset.py
conda run -n tf python -u train_compare.py
conda run -n tf python -u infer_viz.py
conda run -n tf python -u train_shake.py
conda run -n tf python -u infer_shake.py
```

## Idea in one line
Every structural edit is **function-preserving at birth** (zero-output init,
zero-init sprouted synapses, maturity gate) and **loss-gated for life**: each
growth check trials widen / deepen / sprout on probe data and commits only the
direction with the best held-out loss drop per added parameter
(`score = drop / (dparams + 1)`); synapses cost 0 params so rewiring is tried
first, new neurons/layers only when they earn it; dormant units and weak edges
are pruned, the rest consolidated with EWC + sleep replay.

## Results (v2, fair-skip baselines)
- Tiny compositional lang: GrowNet (3 neurons, 0 deep) val 0.64 / unseen 0.75 —
  ties Fixed-Large-48 (0.63/0.75), beats Fixed-Small-3 (0.73/0.85).
- Shakespeare char-LM (100k chars): GrowNet-3 val 2.09 / future 2.10 (ppl 8.2),
  vs Large-96 1.98/1.96 (ppl 7.1), Small-3 2.14/2.15 (ppl 8.6). Ranking holds
  under 0→30% input corruption.
