import os
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt


def load_robot(path: str):
    return np.load(path, allow_pickle=True).item()


def normalize_masses(masses: np.ndarray, target_scale: float = 1.0):
    masses = masses.astype(np.float32).copy()

    center = masses.mean(axis=0, keepdims=True)
    masses = masses - center

    span = masses.max(axis=0) - masses.min(axis=0)
    max_span = max(float(span[0]), float(span[1]), 1e-6)
    masses = masses / max_span * target_scale

    return masses


def plot_robot(ax, robot, title=""):
    masses = normalize_masses(robot["masses"])
    springs = robot["springs"]

    for spring in springs:
        i, j = int(spring[0]), int(spring[1])

        ax.plot(
            [masses[i, 0], masses[j, 0]],
            [masses[i, 1], masses[j, 1]],
            color="blue",
            alpha=0.7,
            linewidth=1.8,
        )

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)

    ax.set_xlim(-0.65, 0.65)
    ax.set_ylim(-0.65, 0.65)

    ax.set_xticks([])
    ax.set_yticks([])


def add_arrow_between_axes(fig, ax_from, ax_to):
    p1 = ax_from.get_position()
    p2 = ax_to.get_position()

    x1 = p1.x0 + p1.width * 0.5
    y1 = p1.y0 - 0.02
    x2 = p2.x0 + p2.width * 0.5
    y2 = p2.y1 + 0.02

    ax_from.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        xycoords=fig.transFigure,
        textcoords=fig.transFigure,
        arrowprops=dict(arrowstyle="->", lw=1.8),
    )


def make_seed_plot(base_dir: str, seed: int, mode: str, output_dir: str):

    seed_dir = os.path.join(base_dir, f"seed_{seed}", mode)

    before_paths = [
        os.path.join(seed_dir, f"before_robot_{i}.npy") for i in range(3)
    ]

    after_paths = [
        os.path.join(seed_dir, f"after_robot_{i}.npy") for i in range(3)
    ]

    missing = [p for p in before_paths + after_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    before_robots = [load_robot(p) for p in before_paths]
    after_robots = [load_robot(p) for p in after_paths]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for i in range(3):

        before_fit = before_robots[i].get("fitness", None)
        after_fit = after_robots[i].get("fitness", None)
        slot_idx = after_robots[i].get("slot_idx", None)

        before_title = f"Before #{i+1}"
        if slot_idx is not None:
            before_title += f"\nslot={slot_idx}"
        if before_fit is not None:
            before_title += f"\nfit={before_fit:.3f}"

        after_title = f"After #{i+1}"
        if slot_idx is not None:
            after_title += f"\nslot={slot_idx}"
        if after_fit is not None:
            after_title += f"\nfit={after_fit:.3f}"

        plot_robot(axes[0, i], before_robots[i], before_title)
        plot_robot(axes[1, i], after_robots[i], after_title)

        add_arrow_between_axes(fig, axes[0, i], axes[1, i])

    axes[0, 0].set_ylabel("Ancestor", fontsize=12)
    axes[1, 0].set_ylabel("Evolved", fontsize=12)

    fig.suptitle(
        f"Seed {seed}: {mode.capitalize()} lineage-matched top 3",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, f"seed_{seed}_{mode}_lineage_top3.png")

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--base_dir", type=str, default="compare_10gen")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)

    parser.add_argument(
        "--mode",
        type=str,
        choices=["random", "surrogate"],
        default="random",
    )

    parser.add_argument("--output_dir", type=str, default="robot_plots")

    args = parser.parse_args()

    for seed in args.seeds:
        make_seed_plot(
            base_dir=args.base_dir,
            seed=seed,
            mode=args.mode,
            output_dir=args.output_dir,
        )