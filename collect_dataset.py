from simulator import Simulator
from robot import load_robots, mutate_mask, mask_to_robot, SCALE, MASK_DIM
import numpy as np
import copy
from utils import load_config
from argparse import ArgumentParser


def global_max_bounds(mask_dim: int):
    max_masses = (mask_dim + 1) ** 2
    max_springs = 4 * mask_dim**2 + 2 * mask_dim
    return max_masses, max_springs


def plot_fitness(fitness_trace: np.ndarray, save_path: str | None = None):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fitness_trace)
    plt.xlabel("Generation")
    plt.ylabel("Fitness (best-so-far)")
    plt.title("Parallel Hill Climber Fitness per Climber")
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


class ParallelHillClimber:
    def __init__(
        self,
        mutation_type_prob,
        num_generations,
        config,
        seed_id: int,
        num_children_per_parent: int = 4,
    ):
        self.mutation_type_prob = mutation_type_prob
        self.num_generations = num_generations
        self.config = config
        self.seed_id = seed_id
        self.num_children_per_parent = num_children_per_parent

        if self.num_children_per_parent < 1:
            raise ValueError("num_children_per_parent must be >= 1")

        self.parent_robots = load_robots(num_robots=self.config["simulator"]["n_sims"])

        self.fitness_scores = None
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
        """
        Simulate an arbitrary list of robots.
        If fewer than n_sims are provided, repeat the last robot to fill the batch.
        Only the first len(robots) outputs are returned.
        """
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

        # rank 0 is best sibling
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
        self.best_control_params = params
        self.fitness_trace[0] = self.fitness_scores

        for gen in range(1, self.num_generations + 1):
            print(f"[Seed {self.seed_id}] Generation {gen}/{self.num_generations}", flush=True)

            next_parents = copy.deepcopy(self.parent_robots)
            next_fitness = self.fitness_scores.copy()
            next_params = copy.deepcopy(self.best_control_params)

            # ------------------------------------------------------------
            # 1) Generate all children for all parents
            # ------------------------------------------------------------
            all_children = []
            all_metadata = []
            parent_ranges = []  # stores (start_idx, end_idx) for each parent in all_children

            cursor = 0
            for j, parent_robot in enumerate(self.parent_robots):
                children, metadata = self.generate_children_for_parent(
                    parent_robot=parent_robot,
                    num_children=self.num_children_per_parent,
                )

                start_idx = cursor
                all_children.extend(children)
                all_metadata.extend(metadata)
                cursor += len(children)
                end_idx = cursor

                parent_ranges.append((start_idx, end_idx))

            # ------------------------------------------------------------
            # 2) Simulate all children in full batches of n_sims
            # ------------------------------------------------------------
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

            # ------------------------------------------------------------
            # 3) Regroup results parent-by-parent, log data, and select best
            # ------------------------------------------------------------
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

    def save(self, prefix=""):
        ranking = np.argsort(self.fitness_scores)[::-1]
        top_3_idxs = ranking[:3].tolist()
        for k, idx in enumerate(top_3_idxs):
            robot = dict(self.parent_robots[idx])
            robot["control_params"] = self.best_control_params[idx]
            robot["max_n_masses"] = self.max_num_masses
            robot["max_n_springs"] = self.max_num_springs
            robot["fitness"] = float(self.fitness_scores[idx])
            np.save(f"{prefix}robot_{k}.npy", robot, allow_pickle=True)


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
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--save_top", action="store_true")
    args = parser.parse_args()

    base_config = load_config(args.config)

    all_datasets = []
    all_traces = []

    for seed in args.seeds:
        print(f"\n=== Running seed {seed} ===")

        config = copy.deepcopy(base_config)
        config["seed"] = seed
        np.random.seed(seed)

        phc = ParallelHillClimber(
            mutation_type_prob=args.mutation_type_prob,
            num_generations=args.num_generations,
            config=config,
            seed_id=seed,
            num_children_per_parent=args.num_children_per_parent,
        )
        phc.run()

        all_datasets.append(phc.finalize_dataset())
        all_traces.append(phc.fitness_trace)

        if args.save_top:
            phc.save(prefix=f"seed_{seed}_")

    merged = concat_datasets(all_datasets)
    np.savez_compressed(args.dataset_path, **merged)

    print(f"\nSaved merged dataset to: {args.dataset_path}")
    print(f"Total samples: {merged['accepted'].shape[0]}")

    if len(all_traces) > 0:
        stacked = np.stack(all_traces, axis=0)
        mean_trace = stacked.mean(axis=(0, 2))

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(mean_trace)
        plt.xlabel("Generation")
        plt.ylabel("Mean fitness")
        plt.title("Mean PHC fitness across seeds")
        plt.grid(True)
        plt.savefig("fitness_trace_multi_seed.png", dpi=200, bbox_inches="tight")
        plt.show()