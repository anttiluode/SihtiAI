# SihtiAI — can the sieve's exhaust teach its next graph?

**Experiment, not a successful segmentation method.** This companion to [Sihti](https://github.com/anttiluode/Sihti) takes the 2026-11-21 discussion with Neon and Astra literally: first sieve an image using the colour-blind lattice, then use the *unsigned lifetime profile* of what disappeared to build a new affinity graph. Compare the new graph with signed residues, core-only blurring, the original self-graph, and spatially shuffled residue features.

The source of the idea is **not** a new neural architecture. A Sihti purifier is an operator whose iterations yield a telescoping set of residues; graph affinities are related to kernel smoothing/attention. This is a test of **feedback from residues to graph construction** using established tools (diffusion, spectral clustering, k-means).

## How to run (Windows, Python 3.13)

Clone the two repositories as *siblings*:

~~~text
D:\anttiluode\Sihti\
D:\anttiluode\SihtiAI\
~~~

Install the original Sihti runtime dependencies, from either directory:

~~~bat
py -3.13 -m pip install numpy scipy pillow matplotlib
cd /d D:\anttiluode\SihtiAI
py -3.13 persistence_head.py --beta 1 --image results\default_stripes.png --scene stripes --output results\default_stripes.json
py -3.13 persistence_head.py --beta 8 --image results\beta8_stripes.png --scene stripes --output results\beta8_stripes.json
py -3.13 persistence_head.py --beta 8 --scene all --output results\beta8_all.json
py -3.13 -m unittest discover -s tests -v
~~~

Original Sihti source inspected against **[b3af693](https://github.com/anttiluode/Sihti/commit/b3af6936c35c1eae63a40117e803309fd33f6244)**. The script uses the original package's functions, not a forked implementation. The parent directory needs to contain a clone of Sihti because it is not a pip-installable package.

## The proposed feedback loop

1. Convert RGB to CIELAB (same scale as Sihti).
2. Apply the original **blank-lattice** lazy graph diffusion at depths 0, 1, 2, 4, 8, 16 (radius 2, laziness 0.5). This ignores image colour when choosing its graph.
3. Save signed residues \(R_j=x_{d_j}-x_{d_{j+1}}\), along with the core \(x_{16}\). Reconstruction is exactly \(x_0=x_{16}+\sum_j R_j\).
4. Take the norm of each **signed** residual before spatial smoothing: \(E_j(p)=\|R_j(p)\|_2\). Six diffusion passes on each \(E_j\) remove black/white polarity cancellation inside stripes. Signed control smooths \(R_j\) directly.
5. Create a second graph on \([x_{16},\,\beta E_0,\ldots,\beta E_J]\). Segmentation uses Sihti's normalized spectral modes and NJW k-means. **No ground-truth ownership labels enter the graph or sieve.**

**Controls:** original raw-Lab self-affinity, core-only, core plus smoothed signed residue, core plus smoothed unsigned residue, unsigned energy only, spatially permuted unsigned energy with its marginal distribution preserved, and raw Lab+XY k-means.

The user's key distinction is kept intact: a lifetime signature is a candidate **relation** between pixels, not by itself an object or a learned attention head.

## First result: original proposed parameter \(\beta=1\) failed

The fixed first-run parameters were: 72×96 scenes, radius 2, sigma 8, J=4 (depth 16), six energy-smoothing passes, beta 1, original scene noise, one scene seed. In an independent standalone reproduction of the upstream mathematical primitives:

| Scene | Original self graph, ARI | Core only, ARI | Signed residue, ARI | Unsigned residue, ARI | Shuffled energy, ARI |
|---|---:|---:|---:|---:|---:|
| Twins | 1.000 | 0.059 | 0.060 | 0.059 | 0.059 |
| Shade | 0.215 | 0.000 | 0.000 | 0.000 | 0.000 |
| Overlap | 1.000 | 0.076 | 0.077 | 0.076 | 0.077 |
| Stripes | 0.710 | 0.000 | 0.000 | 0.000 | 0.000 |
| Leaves | 0.671 | 0.383 | 0.390 | 0.386 | 0.384 |

**The proposed plain feature concatenation did not solve stripes, and damaged most of the other scenes.** A low core boundary contrast and a relatively small unsigned-energy contribution gave the spectral graph no useful cut. The failure is not due to missing signal: the unsigned persistence norm averages about **20.5 inside stripes versus 3.4 outside**.

A reconstruction test found maximum error below **1.5e-14** on all five scenes. This confirms the sieve bookkeeping, not the success of the segmentation.

## Post-hoc scale sweep: there really is a texture signal

After seeing beta=1 fail, we **exploratorily** increased beta. At beta=8 the striped disk became a single spectral object (ARI **0.971**); core-only, signed and shuffled controls remained near zero. On **ten different scene-noise seeds** at beta=8:

| Method | Mean stripes ARI across 10 seeds |
|---|---:|
| Raw-Lab self graph | 0.753 |
| Core only | 0.000 |
| Smoothed signed residue | 0.000 |
| **Unsigned lifetime signature** | **0.970** |
| Spatially scrambled unsigned signature | -0.002 |

These are real results of a small synthetic task, **but beta=8 was chosen with the stripe result in view**. They are not blind confirmation.

And the scene-generalization problem remains. At beta=8, ARI on the original other scene seeds was: twins **0.025**, shade **0.000**, overlap **0.182**, leaves **0.390**. The raw self graph scores 1.000, 0.215, 1.000 and 0.671 respectively. At beta=16, twins improves (0.818), stripes stays high (0.959), but the other cases remain substantially below the self graph.

## What this says, and does not say

**Supported:** Signed black/white contrast can become a spatially coherent *unsigned death-profile* in a lattice sieve; it can teach a second affinity graph to group textures in a case where signed/scrambled controls fail.

**Not supported:** That a simple channel concatenation is a generally superior graph, that this is a neural-network learning rule, that it solves semantic objects, or that a higher beta reliably improves unfamiliar images.

**Next worthwhile conceptual question:** how can the system tell a broad **2D interior of high persistence energy** from the **thin halo of energy around an ordinary object edge**? Both produce unsigned residue, but only one should create new cross-colour connections. A separate multirelational graph or a boundary-sensitive rule must beat the self graph **on held-out scenes**, not merely rescue stripes. No further gate-tuning is claimed here.

The companion experiment keeps the negative first-run result front and center rather than replacing it with the favourable sweep.

## Provenance

- Original science and code modules: [anttiluode/Sihti](https://github.com/anttiluode/Sihti) (MIT).
- The discussion's proposed four-arm test: Neon / Gemini, November 21, 2026.
- Implementation, checks, and negative controls: November 21, 2026.
- No dataset download, GPU, model weights or server needed.
