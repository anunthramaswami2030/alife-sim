import os
from pathlib import Path
from simulator import Simulator
from robot import load_robots, mutate_mask, mask_to_robot, SCALE, MASK_DIM
import numpy as np
import copy
import torch
import torch.nn as nn
from utils import load_config
from argparse import ArgumentParser
from new_model import CNNEncoder
import matplotlib.pyplot as plt


def global_max_bounds(mask_dim: int):
    max_masses = (mask_dim + 1) ** 2
    max_springs = 4 * mask_dim**2 + 2 * mask_dim
    return max_masses, max_springs


def plot_fitness_trace(
    fitness_trace: np.ndarray,
    save_path: str | None = None,
    title: str = "Parallel Hill Climber Fitness Progress",
):
    mean_trace = fitness_trace.mean(axis=1)
    best_trace = fitness_trace.max(axis=1)

    plt.figure()
    plt.plot(mean_trace, label="Mean fitness")
    plt.plot(best_trace, label="Best fitness", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


class AcceptClassifier(nn.Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 64):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=in_channels, latent_dim=latent_dim)

        self.head = nn.Sequential(
            nn.Linear(3 * latent_dim + 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        parent_mask: torch.Tensor,
        child_mask: torch.Tensor,
        parent_fitness: torch.Tensor,
        edit_distance: torch.Tensor,
    ) -> torch.Tensor:
        z_parent = self.encoder(parent_mask)
        z_child = self.encoder(child_mask)
        z_diff = z_child - z_parent

        scalar_feats = torch.stack([parent_fitness, edit_distance], dim=1)
        feat = torch.cat([z_parent, z_child, z_diff, scalar_feats], dim=1)

        logits = self.head(feat).squeeze(1)
        return logits


class ParallelHillClimber:
    def __init__(
        self,
        mutation_type_prob,
        num_generations,
        config,
        seed_id: int,
        num_children_per_parent: int = 4,
        surrogate_checkpoint: str | None = None,
        num_candidates_per_parent: int = 32,
        num_random_explore: int = 1,
    ):
        self.mutation_type_prob = mutation_type_prob
        self.num_generations = num_generations
        self.config = config
        self.seed_id = seed_id
        self.num_children_per_parent = num_children_per_parent
        self.num_candidates_per_parent = num_candidates_per_parent
        self.num_random_explore = num_random_explore

        if self.num_children_per_parent < 1:
            raise ValueError("num_children_per_parent must be >= 1")
        if self.num_candidates_per_parent < self.num_children_per_parent:
            raise ValueError("num_candidates_per_parent must be >= num_children_per_parent")
        if self.num_random_explore < 0 or self.num_random_explore > self.num_children_per_parent:
            raise ValueError("num_random_explore must be between 0 and num_children_per_parent")

        self.parent_robots = load_robots(num_robots=self.config["simulator"]["n_sims"])
        self.initial_parent_robots = copy.deepcopy(self.parent_robots)

        self.fitness_scores = None
        self.initial_fitness_scores = None
        self.initial_control_params = None
        self.best_control_params = None
        self.fitness_trace = None

        max_masses, max_springs = global_max_bounds(MASK_DIM)
        self.max_num_masses = max_masses
        self.max_num_springs = max_springs
        self.config["simulator"]["n_masses"] = max_masses
        self.config["simulator"]["n_springs"] = max_springs

        self.simulator = Simulator(
            sim_config=self.config["simulator"],
            taichi_config=self.config["taichi"],
            seed=self.config["seed"],
            needs_grad=True
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.surrogate = None
        if surrogate_checkpoint is not None:
            ckpt = torch.load(surrogate_checkpoint, map_location=self.device)
            latent_dim = ckpt["latent_dim"]
            in_channels = ckpt["in_channels"]

            self.surrogate = AcceptClassifier(
                in_channels=in_channels,
                latent_dim=latent_dim,
            ).to(self.device)
            self.surrogate.load_state_dict(ckpt["model_state_dict"])
            self.surrogate.eval()
            print(f"[Seed {self.seed_id}] Loaded surrogate from {surrogate_checkpoint}", flush=True)

        self.dataset = {
            "parent_masks": [],
            "child_masks": [],
            "parent_fitness": [],
            "child_fitness": [],
            "fitness_delta": [],
            "accepted": [],
            "generation": [],
            "sim_idx": [],
            "seed": [],
            "parent_id": [],
            "candidate_idx": [],
            "num_candidates_for_parent": [],
            "sibling_rank": [],
            "is_best_sibling": [],
            "edit_distance": [],
            "child_com": [],
        }

    def subsample_com(self, com_traj: np.ndarray, n_points: int = 16):
        idx = np.linspace(0, com_traj.shape[0] - 1, n_points, dtype=int)
        return com_traj[idx].astype(np.float32)

    def build_robot_from_mask(self, mutated_mask):
        masses, springs = mask_to_robot(mutated_mask)
        masses = masses * SCALE
        return {
            "n_masses": int(masses.shape[0]),
            "n_springs": int(springs.shape[0]),
            "masses": masses,
            "springs": springs,
            "mask": mutated_mask,
        }

    def run_sim_robots(self, robots):
        n_sims = self.config["simulator"]["n_sims"]
        if len(robots) < 1:
            raise ValueError("run_sim_robots requires at least one robot")

        padded_robots = robots.copy()
        while len(padded_robots) < n_sims:
            padded_robots.append(copy.deepcopy(robots[-1]))

        masses = [robot["masses"] for robot in padded_robots]
        springs = [robot["springs"] for robot in padded_robots]
        self.simulator.initialize(masses, springs)

        fitness_history = self.simulator.train()
        fitness = fitness_history[:, -1].astype(np.float32)

        params = self.simulator.get_control_params(list(range(n_sims)))
        com = self.simulator.get_com_trajectories().astype(np.float32)

        return (
            fitness[:len(robots)],
            params[:len(robots)],
            com[:len(robots)],
        )

    def generate_children_for_parent(self, parent_robot, num_children):
        parent_mask = parent_robot["mask"]
        children = []
        metadata = []

        for candidate_idx in range(num_children):
            mutated_mask = mutate_mask(parent_mask, add_prob=self.mutation_type_prob).astype(np.float32)
            child = self.build_robot_from_mask(mutated_mask)
            edit_distance = int(np.sum(parent_mask != mutated_mask))

            children.append(child)
            metadata.append(
                {
                    "candidate_idx": candidate_idx,
                    "edit_distance": edit_distance,
                }
            )

        return children, metadata

    def score_children_with_surrogate(self, parent_robot, parent_fitness, children, metadata):
        if self.surrogate is None:
            raise ValueError("Surrogate model is not loaded.")

        parent_mask = np.asarray(parent_robot["mask"], dtype=np.float32)[None, :, :]
        child_masks = np.stack(
            [np.asarray(child["mask"], dtype=np.float32)[None, :, :] for child in children],
            axis=0,
        )
        parent_masks = np.repeat(parent_mask[None, ...], len(children), axis=0)
        parent_fitness_arr = np.full((len(children),), float(parent_fitness), dtype=np.float32)
        edit_distance_arr = np.asarray([m["edit_distance"] for m in metadata], dtype=np.float32)

        parent_masks_t = torch.from_numpy(parent_masks).float().to(self.device)
        child_masks_t = torch.from_numpy(child_masks).float().to(self.device)
        parent_fitness_t = torch.from_numpy(parent_fitness_arr).float().to(self.device)
        edit_distance_t = torch.from_numpy(edit_distance_arr).float().to(self.device)

        with torch.no_grad():
            logits = self.surrogate(
                parent_masks_t,
                child_masks_t,
                parent_fitness_t,
                edit_distance_t,
            )
            probs = torch.sigmoid(logits).cpu().numpy()

        return probs

    def select_children_for_simulation(self, parent_robot, parent_fitness, children, metadata):
        n_keep = self.num_children_per_parent

        if len(children) <= n_keep:
            return children, metadata

        if self.surrogate is None:
            idx = np.random.choice(len(children), size=n_keep, replace=False).tolist()
            return [children[i] for i in idx], [metadata[i] for i in idx]

        probs = self.score_children_with_surrogate(
            parent_robot=parent_robot,
            parent_fitness=parent_fitness,
            children=children,
            metadata=metadata,
        )

        n_top = n_keep - self.num_random_explore
        ranked_idx = np.argsort(probs)[::-1]

        chosen = []
        if n_top > 0:
            chosen.extend(ranked_idx[:n_top].tolist())

        remaining = [i for i in range(len(children)) if i not in chosen]

        if self.num_random_explore > 0 and len(remaining) > 0:
            rand_idx = np.random.choice(
                remaining,
                size=min(self.num_random_explore, len(remaining)),
                replace=False,
            ).tolist()
            chosen.extend(rand_idx)

        selected_children = [children[i] for i in chosen]
        selected_metadata = [metadata[i] for i in chosen]
        return selected_children, selected_metadata

    def log_parent_children(
        self,
        parent_robot,
        parent_fitness,
        children,
        child_fitness,
        child_com,
        gen,
        sim_idx,
        metadata,
    ):
        parent_mask = np.asarray(parent_robot["mask"], dtype=np.float32)
        deltas = child_fitness - float(parent_fitness)

        order = np.argsort(deltas)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(deltas))

        parent_id = (
            self.seed_id * 10_000_000
            + gen * 10_000
            + sim_idx
        )

        best_idx = int(np.argmax(deltas))

        for i, child in enumerate(children):
            child_mask = np.asarray(child["mask"], dtype=np.float32)
            cf = float(child_fitness[i])
            delta = float(deltas[i])
            accepted = 1.0 if cf > float(parent_fitness) else 0.0

            self.dataset["parent_masks"].append(parent_mask[None, :, :])
            self.dataset["child_masks"].append(child_mask[None, :, :])
            self.dataset["parent_fitness"].append(float(parent_fitness))
            self.dataset["child_fitness"].append(cf)
            self.dataset["fitness_delta"].append(delta)
            self.dataset["accepted"].append(accepted)
            self.dataset["generation"].append(gen)
            self.dataset["sim_idx"].append(sim_idx)
            self.dataset["seed"].append(self.seed_id)
            self.dataset["parent_id"].append(parent_id)
            self.dataset["candidate_idx"].append(metadata[i]["candidate_idx"])
            self.dataset["num_candidates_for_parent"].append(len(children))
            self.dataset["sibling_rank"].append(int(ranks[i]))
            self.dataset["is_best_sibling"].append(1 if i == best_idx else 0)
            self.dataset["edit_distance"].append(metadata[i]["edit_distance"])
            self.dataset["child_com"].append(self.subsample_com(child_com[i], n_points=16))

    def finalize_dataset(self):
        out = {
            "parent_masks": np.stack(self.dataset["parent_masks"]).astype(np.float32),
            "child_masks": np.stack(self.dataset["child_masks"]).astype(np.float32),
            "parent_fitness": np.asarray(self.dataset["parent_fitness"], dtype=np.float32),
            "child_fitness": np.asarray(self.dataset["child_fitness"], dtype=np.float32),
            "fitness_delta": np.asarray(self.dataset["fitness_delta"], dtype=np.float32),
            "accepted": np.asarray(self.dataset["accepted"], dtype=np.float32),
            "generation": np.asarray(self.dataset["generation"], dtype=np.int64),
            "sim_idx": np.asarray(self.dataset["sim_idx"], dtype=np.int64),
            "seed": np.asarray(self.dataset["seed"], dtype=np.int64),
            "parent_id": np.asarray(self.dataset["parent_id"], dtype=np.int64),
            "candidate_idx": np.asarray(self.dataset["candidate_idx"], dtype=np.int64),
            "num_candidates_for_parent": np.asarray(self.dataset["num_candidates_for_parent"], dtype=np.int64),
            "sibling_rank": np.asarray(self.dataset["sibling_rank"], dtype=np.int64),
            "is_best_sibling": np.asarray(self.dataset["is_best_sibling"], dtype=np.int64),
            "edit_distance": np.asarray(self.dataset["edit_distance"], dtype=np.int64),
            "child_com": np.stack(self.dataset["child_com"]).astype(np.float32),
        }
        return out

    def save_dataset(self, path="morphology_datasetv2.npz"):
        data = self.finalize_dataset()
        np.savez_compressed(path, **data)

    def run(self):
        n_sims = self.config["simulator"]["n_sims"]
        self.fitness_trace = np.zeros((self.num_generations + 1, n_sims), dtype=np.float32)

        print(f"[Seed {self.seed_id}] Initial evaluation (generation 0/{self.num_generations})", flush=True)
        fs, params, _ = self.run_sim_robots(self.parent_robots)
        self.fitness_scores = fs
        self.initial_fitness_scores = fs.copy()
        self.initial_control_params = copy.deepcopy(params)
        self.best_control_params = params
        self.fitness_trace[0] = self.fitness_scores

        for gen in range(1, self.num_generations + 1):
            print(f"[Seed {self.seed_id}] Generation {gen}/{self.num_generations}", flush=True)

            next_parents = copy.deepcopy(self.parent_robots)
            next_fitness = self.fitness_scores.copy()
            next_params = copy.deepcopy(self.best_control_params)

            all_children = []
            all_metadata = []
            parent_ranges = []

            cursor = 0
            for j, parent_robot in enumerate(self.parent_robots):
                candidate_children, candidate_metadata = self.generate_children_for_parent(
                    parent_robot=parent_robot,
                    num_children=self.num_candidates_per_parent,
                )

                selected_children, selected_metadata = self.select_children_for_simulation(
                    parent_robot=parent_robot,
                    parent_fitness=float(self.fitness_scores[j]),
                    children=candidate_children,
                    metadata=candidate_metadata,
                )

                start_idx = cursor
                all_children.extend(selected_children)
                all_metadata.extend(selected_metadata)
                cursor += len(selected_children)
                end_idx = cursor

                parent_ranges.append((start_idx, end_idx))

            total_children = len(all_children)
            all_fs = []
            all_params = []
            all_com = []

            for start in range(0, total_children, n_sims):
                batch_children = all_children[start:start + n_sims]
                fs_batch, params_batch, com_batch = self.run_sim_robots(batch_children)

                all_fs.append(fs_batch)
                all_params.extend(params_batch)
                all_com.append(com_batch)

            all_fs = np.concatenate(all_fs, axis=0)
            all_com = np.concatenate(all_com, axis=0)

            for j, parent_robot in enumerate(self.parent_robots):
                start_idx, end_idx = parent_ranges[j]

                children_j = all_children[start_idx:end_idx]
                metadata_j = all_metadata[start_idx:end_idx]
                fs_child_j = all_fs[start_idx:end_idx]
                params_child_j = all_params[start_idx:end_idx]
                com_child_j = all_com[start_idx:end_idx]

                self.log_parent_children(
                    parent_robot=parent_robot,
                    parent_fitness=float(self.fitness_scores[j]),
                    children=children_j,
                    child_fitness=fs_child_j,
                    child_com=com_child_j,
                    gen=gen,
                    sim_idx=j,
                    metadata=metadata_j,
                )

                best_idx = int(np.argmax(fs_child_j))
                if fs_child_j[best_idx] > self.fitness_scores[j]:
                    next_parents[j] = children_j[best_idx]
                    next_fitness[j] = fs_child_j[best_idx]
                    next_params[j] = params_child_j[best_idx]

            self.parent_robots = next_parents
            self.fitness_scores = next_fitness
            self.best_control_params = next_params
            self.fitness_trace[gen] = self.fitness_scores

    def save_top_with_corresponding_before(self, seed_dir: str):
        seed_dir = Path(seed_dir)
        robots_dir = seed_dir / "robots"
        robots_dir.mkdir(parents=True, exist_ok=True)

        ranking = np.argsort(self.fitness_scores)[::-1]
        top_3_idxs = ranking[:3].tolist()

        for rank_k, idx in enumerate(top_3_idxs):
            # final robot
            final_robot = dict(self.parent_robots[idx])
            final_robot["control_params"] = self.best_control_params[idx]
            final_robot["max_n_masses"] = self.max_num_masses
            final_robot["max_n_springs"] = self.max_num_springs
            final_robot["fitness"] = float(self.fitness_scores[idx])
            final_robot["slot_idx"] = int(idx)
            np.save(
                robots_dir / f"top{rank_k+1}_after_seed{self.seed_id}_slot{idx}.npy",
                final_robot,
                allow_pickle=True,
            )

            # corresponding initial robot from same slot/lineage
            before_robot = dict(self.initial_parent_robots[idx])
            before_robot["control_params"] = self.initial_control_params[idx]
            before_robot["max_n_masses"] = self.max_num_masses
            before_robot["max_n_springs"] = self.max_num_springs
            before_robot["fitness"] = float(self.initial_fitness_scores[idx])
            before_robot["slot_idx"] = int(idx)
            np.save(
                robots_dir / f"top{rank_k+1}_before_seed{self.seed_id}_slot{idx}.npy",
                before_robot,
                allow_pickle=True,
            )


def concat_datasets(dataset_list):
    keys = dataset_list[0].keys()
    out = {}
    for key in keys:
        out[key] = np.concatenate([d[key] for d in dataset_list], axis=0)
    return out


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset_path", type=str, default="morphology_datasetv2.npz")
    parser.add_argument("--num_generations", type=int, default=200)
    parser.add_argument("--mutation_type_prob", type=float, default=0.65)
    parser.add_argument("--num_children_per_parent", type=int, default=4)
    parser.add_argument("--num_candidates_per_parent", type=int, default=32)
    parser.add_argument("--num_random_explore", type=int, default=1)
    parser.add_argument("--surrogate_checkpoint", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--save_top", action="store_true")
    parser.add_argument("--output_dir", type=str, default="phc_outputs")
    args = parser.parse_args()

    base_config = load_config(args.config)

    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    seed_outputs_dir = output_dir / "per_seed"
    plots_dir.mkdir(parents=True, exist_ok=True)
    seed_outputs_dir.mkdir(parents=True, exist_ok=True)

    all_datasets = []
    all_traces = []

    for seed in args.seeds:
        print(f"\n=== Running seed {seed} ===")

        seed_dir = seed_outputs_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        config = copy.deepcopy(base_config)
        config["seed"] = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        phc = ParallelHillClimber(
            mutation_type_prob=args.mutation_type_prob,
            num_generations=args.num_generations,
            config=config,
            seed_id=seed,
            num_children_per_parent=args.num_children_per_parent,
            surrogate_checkpoint=args.surrogate_checkpoint,
            num_candidates_per_parent=args.num_candidates_per_parent,
            num_random_explore=args.num_random_explore,
        )
        phc.run()

        all_datasets.append(phc.finalize_dataset())
        all_traces.append(phc.fitness_trace)

        # per-seed plot
        plot_fitness_trace(
            phc.fitness_trace,
            save_path=str(seed_dir / "fitness_trace.png"),
            title=f"Seed {seed} Fitness Progress",
        )

        if args.save_top:
            phc.save_top_with_corresponding_before(seed_dir=str(seed_dir))

    merged = concat_datasets(all_datasets)
    dataset_out_path = output_dir / args.dataset_path
    np.savez_compressed(dataset_out_path, **merged)

    print(f"\nSaved merged dataset to: {dataset_out_path}")
    print(f"Total samples: {merged['accepted'].shape[0]}")

    if len(all_traces) > 0:
        stacked = np.stack(all_traces, axis=0)

        mean_trace = stacked.mean(axis=(0, 2))
        best_per_seed = stacked.max(axis=2)
        best_trace = best_per_seed.mean(axis=0)

        plt.figure()
        plt.plot(mean_trace, label="Mean fitness")
        plt.plot(best_trace, label="Best fitness", linewidth=2)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("PHC Fitness Progress Across Seeds")
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / "fitness_trace_multi_seed.png", dpi=200, bbox_inches="tight")
        plt.close()