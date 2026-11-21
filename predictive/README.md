# ResidueHead — causal associative prediction from temporal residues

**Runnable offline tool, not a claim of a new AI algorithm.** It turns the 21 November 2026 Sihti/Sigh/head discussion into the smallest engineering machine that (a) remembers a conditional *change* rather than a raw snapshot, (b) can be queried by the current state, (c) predicts the next observation, and (d) receives an error signal only when the future arrives.

## Open it

Double-click `index.html` in Edge or Chrome. No installation, Python, model download, network, GPU, or server is needed for the built-in worlds or for selecting a local video file. Webcam access depends on browser permissions/security rules; if blocked on `file://`, select a video or use a browser with a trusted localhost origin.

Use **Bouncing pulse**, press **Start**, and watch at least two cycles (96 frames/cycle). The green curve should eventually drop below the yellow hold-last curve. **Step once** is available for controlled inspection. **Reverse motion** causes a surprise. Change **Shape + motion** to **Shape only** to test the need for history when the same position is revisited while moving in opposite directions. Toggle **Allow memory to learn** off to freeze the bank while testing. Switch to your own video or webcam to test whether the method actually helps on real data; it may not. CSV exports only error statistics, not raw video frames.

## Exact mechanism

Frames are 48×36 luminance observations, and addresses contain a 12×9 block-averaged gist and optionally the block-averaged temporal difference from the previous frame. Every completed transition writes a *bridge* `(address_t, observation_{t+1}-observation_t)` into a bounded bank of 640 entries. The head retrieves up to six nearest addresses (distance combines gist and recent motion), discounts old memories by age, and adds the similarity-weighted stored delta to the present observation. Its confidence drops when no address is close.

When the actual next observation arrives, model MSE and hold-last MSE are logged **before** this newly revealed transition is stored. A small local update uses the actual prediction residue (observed minus predicted next frame) to correct sufficiently close past matches; old links gradually lose influence. There are no trainable neural parameters, no labels and no look-ahead.

### Honest limitations

- It's nearest-neighbour transition memory, not a trained transformer, neuron model or general world model.
- The coordinate-addressed pixel deltas assume recurrent scenes are spatially aligned. Camera motion, unknown objects, illumination and changes of scale may defeat it.
- Uniform video sampling has its own timing assumptions; results depend on FPS and temporal aliasing.
- Highly repetitive synthetic worlds are intentionally easy for a transition bank. Results from them do *not* establish useful general video prediction.
- The first moment is not predictable from the past; abrupt interventions cause residual error. The best proof of usefulness is improvement over a causal hold-last baseline on a genuinely unknown real video.

## Frozen-memory benchmark (no training on the test cycle)

The deterministic benchmark trains on exactly **one period** of the built-in sequence, then **freezes all memory writes** and evaluates the following full period against a hold-last baseline. That makes it a valid causal test on a repeated trajectory, though not a test of generalization to novel video.

| World | Address | Relative error reduction vs hold-last, frozen second cycle |
|---|---|---:|
| Bouncing pulse | Coarse shape + recent motion | **92.1%** |
| Bouncing pulse | Coarse shape only | **2.7%** |
| Orbit | Coarse shape + recent motion | **97.7%** |
| Orbit | Coarse shape only | **97.0%** |

The bounce task has an intentional directional ambiguity: the same position is revisited in both directions. The orbit is almost unambiguous from position alone, so this control need not hurt it. These observations support **temporal context when it contains predictive information**, not an advantage on arbitrary signals.

Run `node benchmark.js` to regenerate `results/benchmark.json`.

## Run deterministic checks

With Node.js installed (only for tests, not the GUI):

```bash
node test_head.js
```

Tests cover no-future leakage, differentiating opposing motion with a temporal address, performance on a repeated held-out motion cycle, and failure to anticipate an unseen change. Results are not tuned against external videos.

## Lineage

- [SighImageSuper](https://github.com/anttiluode/SighImageSuper): recurring signal, and the image's loss of contrast.
- [Sihti](https://github.com/anttiluode/Sihti): retain every residue instead of discarding it.
- [SaturdayAfterRHproof](https://github.com/anttiluode/SaturdayAfterRHproof): predictive closure; observable futures can depend on discarded history.
- [HEAD](https://github.com/anttiluode/HEAD): an addressable reading can learn the wrong route; controls matter.
- [WaveFormer](https://github.com/anttiluode/WaveFormer): recurrent memory carried task information across a gap.

This program uses an established idea (associative memory for transition prediction) as an engineering instrument. The point is to observe what it can learn *online*, then let real signals decide whether it is worth developing.