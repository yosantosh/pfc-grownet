# PFC-GrowNet — a growable neuroplastic net

A from-scratch neural net that **starts tiny (3 neurons) and grows itself** —
new neurons, new layers, new synapses — in whatever direction most reduces
loss, during training *and* inference. PFC/basal-ganglia/dopamine-inspired:
growable dendritic posterior, gated working-memory stripes, surprise-driven
ephemeral growth with a hybrid commit gate.

## Layout
- `pfc_grownet.py` — the growable architecture (dendritic layer, BG gates, growth ops)
- `baselines.py` — fixed-size MLP baselines (same I/O, same training protocol)
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
maturity gate) and **loss-gated for life**: widen / deepen / sprout synapses
wherever the predicted loss reduction per added parameter is largest; prune
what stays dormant; consolidate the rest with EWC + sleep replay.
