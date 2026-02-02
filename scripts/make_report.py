"""Tables and figures from the stored results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from aopa import cases  # noqa: E402
from aopa.farfield import array_factor  # noqa: E402

OURS, PUB, RAND, GA, MUTED, INK, INK2, GRID = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#6250d6",
    "#a3a29c",
    "#0b0b0b",
    "#52514e",
    "#e6e5e0",
)
plt.rcParams.update(
    {
        "font.size": 9.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "lines.linewidth": 2.0,
    }
)
LABEL = {
    "zang2025-32": "Zang 2025, 32 ch, ±60°",
    "zang2025-64": "Zang 2025, 64 ch, ±60°",
    "ma2024-128": "Ma 2024, 128 ch, ±47°",
    "zhuang2018-128": "Zhuang 2018, 128 ch, ±82°",
    "komljenovic2017-192": "Komljenovic 2017, 192 ch, ±90°",
}


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def designs(root: Path) -> dict:
    return {n: load(root / "designs" / f"{n}.json") for n in cases.CASES if (root / "designs" / f"{n}.json").exists()}


def fig_benchmarks(d: dict, out: Path) -> None:
    names = list(d)[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 0.62 * len(names) + 1.1))
    for i, n in enumerate(names):
        rep, ours, floor = d[n]["reference"]["reported_db"], d[n]["worst_db"], d[n]["mean_level_db"]
        ax.plot([rep, ours], [i, i], color=MUTED, lw=1.5, zorder=1)
        ax.scatter([rep], [i], s=60, color=PUB, zorder=3, label="Published" if i == 0 else None)
        ax.scatter([ours], [i], s=60, color=OURS, zorder=3, label="This work" if i == 0 else None)
        ax.scatter(
            [floor],
            [i],
            s=40,
            marker="|",
            color=INK2,
            zorder=2,
            label="Mean-level floor of this layout" if i == 0 else None,
        )
        ax.text(rep + 0.25, i + 0.22, f"{rep:.1f}", color=INK2, fontsize=8, ha="left")
        ax.text(ours - 0.25, i + 0.22, f"{ours:.1f}  ({rep - ours:.1f} dB lower)", color=INK, fontsize=8, ha="right")
    ax.set_yticks(range(len(names)), [LABEL.get(n, n) for n in names])
    ax.set_xlabel("Worst-case peak sidelobe level over the steering range (dB, lower is better)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(names) - 0.3)
    fig.savefig(out / "benchmarks.png")
    plt.close(fig)


def fig_scan(d: dict, out: Path) -> None:
    names = list(d)
    fig, axes = plt.subplots(1, len(names), figsize=(2.3 * len(names), 2.6), sharey=True)
    for ax, n in zip(np.atleast_1d(axes), names):
        r = d[n]
        ax.plot(r["scan_angles_deg"], r["scan_db"], color=OURS, label="This work")
        ax.axhline(r["reference"]["reported_db"], color=PUB, ls="--", lw=1.5, label="Published worst case")
        ax.set_title(LABEL.get(n, n).replace(", ", "\n", 1), fontsize=8.5, color=INK)
        ax.set_xlabel("Steering angle (°)")
    np.atleast_1d(axes)[0].set_ylabel("Peak sidelobe level (dB)")
    np.atleast_1d(axes)[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "steering_scan.png")
    plt.close(fig)


def fig_pattern(d: dict, root: Path, out: Path, name: str = "zang2025-64") -> None:
    if name not in d:
        return
    spec = cases.get(name)
    x = np.array(d[name]["positions_um"])
    base = load(root / "baselines" / f"{name}.json")
    pso = None
    if base:
        runs = [r for r in base["runs"] if r["method"] == "pso"]
        pso = np.array(min(runs, key=lambda r: r["worst_db"])["positions_um"]) if runs else None
    theta0 = -spec.steer_deg
    u0 = np.sin(np.radians(theta0))
    th = np.linspace(-90, 90, 6001)
    u = np.sin(np.radians(th))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.8, 4.2), height_ratios=[3, 1.1])

    def pattern_db(pos):
        inten = spec.pattern(u) ** 2 * np.abs(array_factor(pos, u - u0, spec.wavelength)) ** 2
        return 10 * np.log10(inten / (spec.pattern(np.array([u0]))[0] ** 2 * spec.n**2))

    if pso is not None:
        ax.plot(th, pattern_db(pso), color=MUTED, lw=1.2, label="Particle swarm, same metric")
    ax.plot(th, pattern_db(x), color=OURS, lw=1.4, label="This work")
    ax.axhline(d[name]["reference"]["reported_db"], color=PUB, ls="--", lw=1.3, label="Published worst case")
    ax.axhline(d[name]["worst_db"], color=OURS, ls=":", lw=1.3)
    ax.set_ylim(-35, 2)
    ax.set_xlim(-90, 90)
    ax.set_ylabel("Intensity (dB)")
    ax.set_title(f"{LABEL[name]}: beam steered to {theta0:.0f}°, the worst case", fontsize=9, color=INK)
    ax.legend(fontsize=7.5, loc="upper right", ncols=1)
    ax2.eventplot([x], colors=[OURS], lineoffsets=[1], linelengths=[0.8])
    if pso is not None:
        ax2.eventplot([pso], colors=[MUTED], lineoffsets=[0], linelengths=[0.8])
    ax2.set_yticks([0, 1] if pso is not None else [1], ["PSO", "Ours"] if pso is not None else ["Ours"])
    ax2.set_xlabel("Emitter position (µm)")
    ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "pattern.png")
    plt.close(fig)


def fig_channels(d: dict, root: Path, out: Path) -> list[str]:
    rows = [load(p) for p in sorted((root / "channels").glob("*.json"))]
    if not rows:
        return []
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    lines = []
    for base, color in (("komljenovic2017-192", OURS), ("zhuang2018-128", RAND)):
        pts = sorted([(r["n"], r["worst_db"]) for r in rows if r["case"] == base])
        if base in d:
            pts = sorted(set(pts + [(cases.get(base).n, d[base]["worst_db"])]))
        if not pts:
            continue
        n, v = zip(*pts)
        ax.plot(n, v, color=color, marker="o", ms=5, label=f"This work, {LABEL[base].split(',')[0]} constraints")
        rep_n, rep = cases.get(base).n, cases.get(base).reference["reported_db"]
        ax.axhline(rep, color=color, ls="--", lw=1.0, alpha=0.7)
        ax.scatter(
            [rep_n],
            [rep],
            marker="D",
            s=50,
            color=color,
            edgecolor=INK,
            zorder=4,
            label=f"{LABEL[base].split(',')[0]} (published)",
        )
        match = [k for k, val in pts if val <= rep]
        if match:
            lines.append(
                f"{LABEL[base].split(',')[0]}: {min(match)} emitters reach the published "
                f"{rep} dB of the {rep_n}-emitter design"
            )
    ax.set_xlabel("Number of emitters (phase shifters)")
    ax.set_ylabel("Worst-case PSL (dB)")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "channels.png")
    plt.close(fig)
    return lines


def fig_baselines(d: dict, root: Path, out: Path) -> list[str]:
    names = [n for n in d if (root / "baselines" / f"{n}.json").exists()]
    if not names:
        return []
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 2.9))
    lines = []
    for ax, n in zip(np.atleast_1d(axes), names):
        b = load(root / "baselines" / f"{n}.json")
        for meth, color, lab in (("pso", PUB, "Particle swarm"), ("ga", GA, "Genetic algorithm")):
            runs = [r for r in b["runs"] if r["method"] == meth]
            for k, r in enumerate(runs):
                t, v = zip(*r["trace"])
                ax.step(
                    np.maximum(t, 0.5), v, where="post", color=color, lw=1.2, alpha=0.9, label=lab if k == 0 else None
                )
            if runs:
                lines.append(
                    f"| {n} | {lab} | {len(runs)} | {min(r['worst_db'] for r in runs):.2f} | "
                    f"{np.median([r['worst_db'] for r in runs]):.2f} |"
                )
        h = np.array(d[n]["history"])
        ax.step(np.maximum(h[:, 0], 0.5), h[:, 1], where="post", color=OURS, label="This work")
        ax.axhline(
            b["random"]["best_db"], color=RAND, ls=":", lw=1.3, label=f"Best of {b['random']['evaluations']:,} random"
        )
        ax.axhline(d[n]["reference"]["reported_db"], color=INK2, ls="--", lw=1.2, label="Published")
        lines.append(
            f"| {n} | Random search | {b['random']['evaluations']:,} layouts | {b['random']['best_db']:.2f} | "
            f"{b['random']['median_db']:.2f} (all) |"
        )
        lines.append(f"| {n} | **This work** | 1 | **{d[n]['worst_db']:.2f}** | |")
        ax.set_xscale("log")
        ax.set_xlabel("Wall-clock time (s, one core)")
        ax.set_title(LABEL.get(n, n), fontsize=9, color=INK)
    np.atleast_1d(axes)[0].set_ylabel("Best worst-case PSL so far (dB)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncols=5, fontsize=7.5)
    fig.savefig(out / "baselines.png")
    plt.close(fig)
    return lines


def fig_robustness(root: Path, out: Path) -> tuple[list[str], list[str]]:
    r = load(root / "robustness.json")
    if not r:
        return [], []
    base = load(root / "robustness_baselines.json") or {}
    lines, band_lines = [], []
    for n in [n for n in r if n in LABEL]:
        e = r[n]
        p50, p100 = e["position"]["0.05"], e["position"]["0.1"]
        row = (
            f"| {n} | this work | {e['nominal_db']:.2f} | {p50['mean']:.2f} | {p100['mean']:.2f} (95 %: "
            f"{p100['p95']:.2f}) | {e['phase']['10.0']['mean']:.2f} | {max(e['wavelength_db']):.2f} |"
        )
        lines.append(row)
        for meth, lab in (("pso", "particle swarm"), ("ga", "genetic algorithm")):
            if n in base:
                b = base[n][meth]
                q50, q100 = b["position"]["0.05"], b["position"]["0.1"]
                lines.append(
                    f"| {n} | {lab} | {b['nominal_db']:.2f} | {q50['mean']:.2f} | {q100['mean']:.2f} (95 %: "
                    f"{q100['p95']:.2f}) | | |"
                )
    show = [n for n in ("zang2025-32", "zang2025-64") if n in r]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    a1, a2 = axes
    for n, ls in zip(show, ("-", "--")):
        e = r[n]
        amps = np.array([0.0] + [float(k) for k in e["position"]]) * 1000
        short = f"{cases.get(n).n} ch"
        a1.plot(
            amps,
            [e["nominal_db"]] + [v["mean"] for v in e["position"].values()],
            color=OURS,
            ls=ls,
            marker="o",
            ms=4,
            label=f"This work, {short}",
        )
        if n in base:
            for meth, color, lab in (("pso", PUB, "PSO"), ("ga", GA, "GA")):
                b = base[n][meth]
                a1.plot(
                    amps[: len(b["position"]) + 1],
                    [b["nominal_db"]] + [v["mean"] for v in b["position"].values()],
                    color=color,
                    ls=ls,
                    marker="o",
                    ms=4,
                    label=f"{lab}, {short}",
                )
        a2.plot(
            np.array(e["wavelength_um"]) * 1000,
            e["wavelength_db"],
            color=OURS,
            ls=ls,
            label=f"Designed at 1550 nm, {short}",
        )
        bnd = load(root / "band" / f"{n}-band.json")
        if bnd:
            from aopa import robustness

            spec = cases.get(n)
            wl = np.array(e["wavelength_um"])
            sweep = robustness.wavelength_sweep(spec, np.array(bnd["positions_um"]), wl)
            a2.plot(wl * 1000, sweep, color=RAND, ls=ls, label=f"Designed for 1500-1600 nm, {short}")
            band_lines.append(
                f"| {n} | {e['nominal_db']:.2f} | {max(e['wavelength_db']):.2f} | {sweep[len(sweep) // 2]:.2f} | "
                f"{sweep.max():.2f} |"
            )
    a1.set_xlabel("Emitter displacement, uniform ± (nm)")
    a1.set_ylabel("Worst-case PSL, mean of 200 trials (dB)")
    a2.set_xlabel("Wavelength (nm)")
    a2.set_ylabel("Worst-case PSL (dB)")
    a1.legend(fontsize=6.5, ncols=2)
    a2.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "robustness.png")
    plt.close(fig)
    return lines, band_lines


def fig_planar(root: Path, out: Path) -> dict | None:
    rec = load(root / "planar" / "gao2024-625.json")
    if not rec:
        return None
    from run_planar import SPEC

    from aopa.planar import PlanarEvaluator

    p = np.array(rec["positions_um"])
    ev = PlanarEvaluator(SPEC, oversample=6)
    f = ev.profile(p)
    full = np.concatenate([f[::-1, :0:-1], f], axis=1)
    vy = np.concatenate([-ev.vy[:0:-1], ev.vy])
    db = 10 * np.log10(np.maximum(full, 1e-6))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 4.0), width_ratios=[1, 1.25])
    a1.scatter(p[:, 0], p[:, 1], s=4, color=OURS)
    a1.set_aspect("equal")
    a1.set_xlabel("x (µm)")
    a1.set_ylabel("y (µm)")
    a1.set_title(f"{SPEC.n} emitters, d_min = {SPEC.d_min} µm", fontsize=9, color=INK)
    a1.grid(False)
    im = a2.imshow(
        db.T,
        origin="lower",
        extent=[ev.vx[0], ev.vx[-1], vy[0], vy[-1]],
        cmap="Blues_r",
        vmin=-40,
        vmax=rec["worst_db"],
    )
    circ = plt.Circle((0, 0), SPEC.v_max, fill=False, color=INK2, lw=0.8, ls="--")
    a2.add_patch(circ)
    a2.set_xlabel("v_x")
    a2.set_ylabel("v_y")
    a2.set_title("Steering-weighted sidelobe map (dB), all beam directions in the 40° cone", fontsize=8.5, color=INK)
    a2.grid(False)
    fig.colorbar(im, ax=a2, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out / "planar.png")
    plt.close(fig)
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results")
    args = parser.parse_args()
    root = Path(args.root)
    figs = root / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    d = designs(root)
    md = ["# Results", ""]
    if d:
        fig_benchmarks(d, figs)
        fig_scan(d, figs)
        fig_pattern(d, root, figs)
        md += [
            "## Published designs, same constraints",
            "",
            "| Case | Emitters | Steering | Published (dB) | This work (dB) | Improvement (dB) | "
            "Mean-level floor (dB) | Independent scan (dB) | Beam FWHM | Search |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for n, r in d.items():
            s = cases.get(n)
            md.append(
                f"| {n} | {s.n} | ±{s.steer_deg:g}° | {r['reference']['reported_db']:.2f} | "
                f"**{r['worst_db']:.2f}** | {r['reference']['reported_db'] - r['worst_db']:.2f} | "
                f"{r['mean_level_db']:.2f} | {r['scan_worst_db']:.2f} | {r['fwhm_deg']:.3f}° | "
                f"{r['budget_s'] / 60:.0f} min, {r['local_runs']} local runs |"
            )
        md.append("")
    lines = fig_baselines(d, root, figs)
    if lines:
        md += (
            [
                "## Same metric, different optimisers",
                "",
                "| Case | Method | Runs | Best (dB) | Median (dB) |",
                "|---|---|---|---|---|",
            ]
            + lines
            + [""]
        )
    lines = fig_channels(d, root, figs)
    if lines:
        md += ["## Fewer channels for the same sidelobe level", ""] + [f"* {s}" for s in lines] + [""]
    lines, band_lines = fig_robustness(root, figs)
    if lines:
        md += (
            [
                "## Tolerance",
                "",
                "Worst-case PSL over the steering range (dB). Displacement: every emitter moved by a uniform random "
                "amount up to the stated value, mean of 200 trials. Phase: Gaussian residual error per emitter.",
                "",
                "| Design | Layout | Nominal | ±50 nm | ±100 nm | 10° phase error | Worst over 1500-1600 nm |",
                "|---|---|---|---|---|---|---|",
            ]
            + lines
            + [""]
        )
    if band_lines:
        md += (
            [
                "## Designing over a wavelength band",
                "",
                "| Case | 1550 nm design, at 1550 nm | 1550 nm design, worst over band | "
                "Band design, at 1550 nm | Band design, worst over band |",
                "|---|---|---|---|---|",
            ]
            + band_lines
            + [""]
        )
    rec = fig_planar(root, figs)
    if rec:
        md += [
            "## Planar array",
            "",
            "| Layout | Worst-case PSL over the 40° cone (dB) |",
            "|---|---|",
            f"| Published, uniform amplitudes | {rec['reference']['uniform_db']} |",
            f"| Published, amplitudes and phases optimised per beam | {rec['reference']['tapered_db']} |",
            f"| Random Poisson-disc layouts (best of {len(rec['random_layouts_db'])}) | "
            f"{min(rec['random_layouts_db']):.2f} |",
            f"| **This work, uniform amplitudes** | **{rec['worst_db']:.2f}** |",
            "",
            f"Direct steering check over {len(rec['check_db'])} beam directions: worst {max(rec['check_db']):.2f} dB. "
            f"Mean-level floor {rec['mean_level_db']:.2f} dB.",
            "",
        ]
    (root / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
