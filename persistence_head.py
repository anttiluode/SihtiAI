"""SihtiAI: two-stage persistence-head ablation (no learned weights).

Requires the sibling checkout ../Sihti from anttiluode/Sihti.
The sieve is lossless; the added feedback graph need not be useful.
Hypothesis and controls: Gemini discussion 2026-11-21.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

UPSTREAM = Path(__file__).resolve().parent.parent / "Sihti"
if UPSTREAM.is_dir():
    sys.path.insert(0, str(UPSTREAM))

try:
    from sihti.color import srgb_to_lab
    from sihti.graph import grid_graph, lattice_affinity, self_affinity
    from sihti.purifiers import GraphDiffusion
    from sihti.sieve import sift, octave_depths
    from sihti.modes import spectral_modes, segments
    from sihti.baselines import kmeans_lab_xy
    from sihti.metrics import covering, ari, vi
    from sihti.scenes import SCENES
except ImportError as exc:
    raise SystemExit("Clone https://github.com/anttiluode/Sihti next to SihtiAI; "
                     "install numpy scipy pillow. Original import error: " + str(exc))

METHODS = ("self", "core_only", "signed_res", "unsigned_res",
           "unsigned_scrambled", "energy_only", "kmeans_xy")


def sieve_features(lab, radius=2., J=4, smooth_passes=6, laziness=.5):
    """The same lattice step and octave decomposition for all test arms."""
    h, w = lab.shape[:2]
    step = GraphDiffusion(lattice_affinity(h, w, radius=radius,
                                           sigma_x=max(1., radius)),
                          h, w, laziness=laziness)
    record = sift(lab, step, octave_depths(J))
    core = record.core
    bands = record.residues
    signed = np.concatenate(bands, axis=-1)
    energy = np.stack([np.linalg.norm(b, axis=-1) for b in bands], axis=-1)
    for _ in range(smooth_passes):
        signed = step(signed)
        energy = step(energy)
    reconstruction_error = float(
        np.abs(core + np.sum(record.residues, axis=0) - lab).max())
    return core, signed, energy, reconstruction_error


def choose_features(mode, lab, core, signed, energy, beta, seed=131):
    if mode == "self":
        return lab
    if mode == "core_only":
        return core
    if mode == "signed_res":
        return np.concatenate((core, beta * signed), axis=-1)
    if mode == "unsigned_res":
        return np.concatenate((core, beta * energy), axis=-1)
    if mode == "energy_only":
        return beta * energy
    if mode == "unsigned_scrambled":
        rng = np.random.default_rng(seed)
        flat = energy.reshape(-1, energy.shape[-1])
        scrambled = flat[rng.permutation(flat.shape[0])].reshape(energy.shape)
        return np.concatenate((core, beta * scrambled), axis=-1)
    raise ValueError(mode)


def run_scene(name, h=72, w=96, noise=.02, seed=0, radius=2.,
              sigma=8., J=4, smooth_passes=6, beta=1., laziness=.5,
              methods=METHODS):
    make = SCENES[name]
    scene_seed = list(SCENES).index(name) + seed
    rgb, gt = make(H=h, W=w, noise=noise, seed=scene_seed)
    lab = srgb_to_lab(rgb)
    k = int(np.unique(gt).size)
    core, signed, energy, error = sieve_features(
        lab, radius=radius, J=J, smooth_passes=smooth_passes, laziness=laziness)
    g = grid_graph(h, w, radius=radius)
    rows = []
    segmentations = {}
    for method in methods:
        started = time.perf_counter()
        if method == "kmeans_xy":
            seg = kmeans_lab_xy(lab, k, seed=0)
        else:
            feat = choose_features(method, lab, core, signed, energy, beta)
            Wm = g.matrix(g.weights(feat, sigma=sigma,
                                    sigma_x=max(1., radius)))
            md = spectral_modes(Wm, h, w, k=max(16, k + 4), seed=0)
            seg = segments(md, k, seed=0)
        segmentations[method] = seg
        rows.append(dict(scene=name, method=method, seed=seed,
                         beta=beta, covering=covering(seg, gt),
                         ari=ari(seg, gt), vi=vi(seg, gt),
                         seconds=round(time.perf_counter() - started, 3)))
    return dict(records=rows, reconstruction_error=error,
                rgb=rgb, gt=gt, segmentations=segmentations,
                core=core, energy=energy)


def save_png(run, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    titles = ["input", "ground truth"] + list(run["segmentations"])
    fig, axs = plt.subplots(1, len(titles), figsize=(2.2 * len(titles), 2.4))
    axs[0].imshow(run["rgb"])
    axs[1].imshow(run["gt"], cmap="tab20", interpolation="nearest")
    for i, (mode, seg) in enumerate(run["segmentations"].items(), 2):
        axs[i].imshow(seg, cmap="tab20", interpolation="nearest")
    for ax, title in zip(axs, titles):
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=["all", *SCENES], default="all")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--beta", type=float, default=1.)
    p.add_argument("--height", type=int, default=72)
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--sigma", type=float, default=8.)
    p.add_argument("--radius", type=float, default=2.)
    p.add_argument("--J", type=int, default=4)
    p.add_argument("--smooth-passes", type=int, default=6)
    p.add_argument("--noise", type=float, default=.02)
    p.add_argument("--output", type=Path, default=Path("results/latest.json"))
    p.add_argument("--image", type=Path, help="Optional: PNG for the first scene/seed")
    args = p.parse_args()
    if args.J < 0 or args.smooth_passes < 0 or args.beta < 0:
        p.error("J, smooth-passes and beta must be nonnegative")
    scenes = list(SCENES) if args.scene == "all" else [args.scene]
    rows = []
    recon_errors = {}
    for scene in scenes:
        for seed in args.seeds:
            run = run_scene(scene, h=args.height, w=args.width,
                            sigma=args.sigma, radius=args.radius, J=args.J,
                            smooth_passes=args.smooth_passes,
                            noise=args.noise, seed=seed, beta=args.beta)
            rows += run["records"]
            recon_errors[f"{scene}_{seed}"] = run["reconstruction_error"]
            for row in run["records"]:
                print(f"{row['scene']:8s} {row['method']:20s} "
                      f"ARI {row['ari']:.3f} cover {row['covering']:.3f} "
                      f"VI {row['vi']:.3f}", flush=True)
            if args.image and scene == scenes[0] and seed == args.seeds[0]:
                args.image.parent.mkdir(parents=True, exist_ok=True)
                save_png(run, args.image)
    output = {"parameters": vars(args) | {"output": str(args.output),
                                          "image": str(args.image)},
              "reconstruction_errors": recon_errors, "runs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
