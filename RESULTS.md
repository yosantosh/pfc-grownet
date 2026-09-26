# Results v2 — 3-neuron seed, loss-driven growth in all dimensions

Seed: every GrowNet is born with exactly **3 posterior (L1) neurons** and an
**empty L2 deep layer**. Growth checks trial widen / deepen / sprout on probe
data and commit only the best held-out loss drop per added parameter.
Fixed baselines carry the same residual skip, so gaps isolate *growth*.

## Experiment A — tiny compositional language (next-token, V=18, ctx=4)

| Model | Layers (active neurons) | Params total (alloc) | Params active | Val loss / acc | Test-unseen loss / acc (ppl) | Gen gap (test−train) |
|---|---|---|---|---|---|---|
| GrowNet (ours) | emb→**L1: 3** (3 dendrites each)→L2: 0/16→mem 16→out 18 + skip; **427/576 L1 synapses** live | 17,090 | **2,583** (emb 288, L1 585, gates 192, readout 360, skip 1,152) | 0.642 / 0.70 | **0.753 / 0.616** (2.12) | −0.168 |
| Fixed-Small | emb→h: 3→out 18 + skip | 1,713 | 1,713 | 0.725 / 0.70 | 0.851 / 0.607 (2.34) | −0.100 |
| Fixed-Large | emb→h: 48→out 18 + skip | 5,538 | 5,538 | **0.630** / 0.68 | **0.745** / 0.616 (2.11) | −0.142 |

Read: 3 growable neurons tie a 48-neuron fixed net (val gap 0.012, test gap
0.008) and clearly beat the fixed 3-neuron net. Growth trials repeatedly
chose `sprout` (free: 0 new params) early and `none` at convergence — edge
rewiring sufficed, so no 4th neuron or L2 was ever committed.

## Experiment B — Tiny Shakespeare char-LM (100k train chars, V=61, ctx=8)

| Model | Layers (active neurons) | Params total (alloc) | Params active | Val loss / acc | Test-future loss / acc (ppl) | Noisy-10% loss / acc | Gap future−val |
|---|---|---|---|---|---|---|---|
| GrowNet (ours) | emb→**L1: 3**→L2: 0/32→mem 24→out 61 + skip; **1,439 live L1 synapses** | 94,645 | **16,915** (emb 1,464, L1 1,737, gates 288, readout 1,708, skip 11,712) | 2.094 / 0.417 | 2.099 / 0.404 (8.16) | 2.349 / 0.359 | +0.005 |
| Fixed-Small | emb→h: 3→out 61 + skip | 14,005 | 14,005 | 2.145 / 0.402 | 2.150 / 0.391 (8.59) | 2.385 / 0.344 | +0.006 |
| Fixed-Large | emb→h: 96→out 61 + skip | 37,813 | 37,813 | **1.984** / 0.442 | **1.961** / 0.441 (7.11) | **2.266** / 0.382 | −0.022 |

Read: Fixed-Large wins narrowly on real text; GrowNet-3 beats Fixed-Small-3
everywhere with ~21% more active params (dendrites + memory + skip) and holds
rank under 0→30% input corruption. The trial gate stayed conservative on
Shakespeare (2 quick-tune steps undervalue newborns whose maturity starts at
0) — next knob is a longer trial horizon / baby-boost during trials.

## Growth-trial log (typical, tiny-lang)
`[epoch 2] widen +0.0866/522p, deepen +0.0871/330p, sprout +0.0871/0p → sprout`
`[epoch 10] widen −0.0014, deepen −0.0021, sprout −0.0021 → none`
All births are function-preserving (verified maxdiff ≤ 2.4e-07).
