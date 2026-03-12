import os
import copy
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import load_config
from robot import load_robots
from surrogate_datacollect import ParallelHillClimber


def save_top3(save_dir, phc, robots, params, fitness):
    os.makedirs(save_dir, exist_ok=True)

    ranking = np.argsort(fitness)[::-1][:3]
    for k, idx in enumerate(ranking):
        robot = dict(robots[idx])
        robot["control_params"] = params[idx]
        robot["max_n_masses"] = phc.max_num_masses
        robot["max_n_springs"] = phc.max_num_springs
        robot["fitness"] = float(fitness[idx])
        robot["slot_idx"] = int(idx)
        np.save(
            os.path.join(save_dir, f"robot_{k}.npy"),
            robot,
            allow_pickle=True,
        )


def save_top3_with_lineage_matches(
    save_dir,
    phc,
    final_robots,
    final_params,
    final_fitness,
    initial_robots,
    initial_params,
    initial_fitness,
):
    os.makedirs(save_dir, exist_ok=True)

    ranking = np.argsort(final_fitness)[::-1][:3]
    for k, idx in enumerate(ranking):
        # after-evolution robot
        robot_after = dict(final_robots[idx])
        robot_after["control_params"] = final_params[idx]
        robot_after["max_n_masses"] = phc.max_num_masses
        robot_after["max_n_springs"] = phc.max_num_springs
        robot_after["fitness"] = float(final_fitness[idx])
        robot_after["slot_idx"] = int(idx)
        np.save(
            os.path.join(save_dir, f"after_robot_{k}.npy"),
            robot_after,
            allow_pickle=True,
        )

        # lineage-matched initial robot from same slot
        robot_before = dict(initial_robots[idx])
        robot_before["control_params"] = initial_params[idx]
        robot_before["max_n_masses"] = phc.max_num_masses
        robot_before["max_n_springs"] = phc.max_num_springs
        robot_before["fitness"] = float(initial_fitness[idx])
        robot_before["slot_idx"] = int(idx)
        np.save(
            os.path.join(save_dir, f"before_robot_{k}.npy"),
            robot_before,
            allow_pickle=True,
        )


def plot_compare(random_traces, surrogate_traces, output_prefix):
    plots_dir = os.path.join(output_prefix, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    random_stacked = np.stack(random_traces, axis=0)
    surrogate_stacked = np.stack(surrogate_traces, axis=0)

    random_mean = random_stacked.mean(axis=(0, 2))
    surrogate_mean = surrogate_stacked.mean(axis=(0, 2))

    random_best = random_stacked.max(axis=2).mean(axis=0)
    surrogate_best = surrogate_stacked.max(axis=2).mean(axis=0)

    plt.figure()
    plt.plot(random_mean, label="Random mean")
    plt.plot(surrogate_mean, label="Surrogate mean")
    plt.plot(random_best, label="Random best", linewidth=2)
    plt.plot(surrogate_best, label="Surrogate best", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Random vs Surrogate PHC")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(plots_dir, "combined.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure()
    plt.plot(random_mean, label="Random mean")
    plt.plot(surrogate_mean, label="Surrogate mean")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Mean Fitness Comparison")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(plots_dir, "mean.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure()
    plt.plot(random_best, label="Random best", linewidth=2)
    plt.plot(surrogate_best, label="Surrogate best", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Best Fitness Comparison")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(plots_dir, "best.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--num_generations", type=int, default=10)
    parser.add_argument("--mutation_type_prob", type=float, default=0.65)
    parser.add_argument("--num_children_per_parent", type=int, default=4)
    parser.add_argument("--num_candidates_per_parent", type=int, default=32)
    parser.add_argument("--num_random_explore", type=int, default=1)
    parser.add_argument("--surrogate_checkpoint", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20, 21, 22, 23])
    parser.add_argument("--output_prefix", type=str, default="compare_10gen")
    parser.add_argument("--save_top", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_prefix, exist_ok=True)

    base_config = load_config(args.config)

    random_traces = []
    surrogate_traces = []

    for seed in args.seeds:
        print(f"\n=== Comparing seed {seed} ===", flush=True)

        seed_dir = os.path.join(args.output_prefix, f"seed_{seed}")
        before_dir = os.path.join(seed_dir, "before")
        random_dir = os.path.join(seed_dir, "random")
        surrogate_dir = os.path.join(seed_dir, "surrogate")

        os.makedirs(seed_dir, exist_ok=True)

        np.random.seed(seed)
        torch.manual_seed(seed)

        config_random = copy.deepcopy(base_config)
        config_random["seed"] = seed

        config_surrogate = copy.deepcopy(base_config)
        config_surrogate["seed"] = seed

        n_sims = config_random["simulator"]["n_sims"]
        initial_parents = load_robots(num_robots=n_sims)

        # -------------------------
        # RANDOM
        # -------------------------
        np.random.seed(seed)
        torch.manual_seed(seed)

        random_phc = ParallelHillClimber(
            mutation_type_prob=args.mutation_type_prob,
            num_generations=args.num_generations,
            config=config_random,
            seed_id=seed,
            num_children_per_parent=args.num_children_per_parent,
            surrogate_checkpoint=None,
            num_candidates_per_parent=args.num_children_per_parent,
            num_random_explore=0,
        )
        random_phc.parent_robots = copy.deepcopy(initial_parents)

        print(
            f"[random][seed {seed}] len(parent_robots)={len(random_phc.parent_robots)} n_sims={n_sims}",
            flush=True,
        )

        fs0, params0, _ = random_phc.run_sim_robots(random_phc.parent_robots)

        # optional: save global top-3 initial robots by fitness
        if args.save_top:
            save_top3(
                before_dir,
                random_phc,
                robots=copy.deepcopy(initial_parents),
                params=copy.deepcopy(params0),
                fitness=fs0.copy(),
            )

        random_phc.run()
        random_traces.append(random_phc.fitness_trace)

        if args.save_top:
            save_top3_with_lineage_matches(
                random_dir,
                random_phc,
                final_robots=random_phc.parent_robots,
                final_params=random_phc.best_control_params,
                final_fitness=random_phc.fitness_scores,
                initial_robots=initial_parents,
                initial_params=params0,
                initial_fitness=fs0,
            )

        # -------------------------
        # SURROGATE
        # -------------------------
        np.random.seed(seed)
        torch.manual_seed(seed)

        surrogate_phc = ParallelHillClimber(
            mutation_type_prob=args.mutation_type_prob,
            num_generations=args.num_generations,
            config=config_surrogate,
            seed_id=seed,
            num_children_per_parent=args.num_children_per_parent,
            surrogate_checkpoint=args.surrogate_checkpoint,
            num_candidates_per_parent=args.num_candidates_per_parent,
            num_random_explore=args.num_random_explore,
        )
        surrogate_phc.parent_robots = copy.deepcopy(initial_parents)

        print(
            f"[surrogate][seed {seed}] len(parent_robots)={len(surrogate_phc.parent_robots)} n_sims={n_sims}",
            flush=True,
        )

        surrogate_phc.run()
        surrogate_traces.append(surrogate_phc.fitness_trace)

        if args.save_top:
            save_top3_with_lineage_matches(
                surrogate_dir,
                surrogate_phc,
                final_robots=surrogate_phc.parent_robots,
                final_params=surrogate_phc.best_control_params,
                final_fitness=surrogate_phc.fitness_scores,
                initial_robots=initial_parents,
                initial_params=params0,
                initial_fitness=fs0,
            )

    plot_compare(random_traces, surrogate_traces, args.output_prefix)
    print(f"\nSaved results under: {args.output_prefix}")