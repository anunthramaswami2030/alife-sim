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
    def __init__(self, mutation_type_prob, num_generations, config):
        self.mutation_type_prob = mutation_type_prob
        self.num_generations = num_generations
        self.config = config

        self.parent_robots = load_robots(num_robots=self.config["simulator"]["n_sims"])
        self.child_robots = copy.deepcopy(self.parent_robots)

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

    def run_sim(self):
        masses = [robot["masses"] for robot in self.child_robots]
        springs = [robot["springs"] for robot in self.child_robots]
        self.simulator.initialize(masses, springs)
        fitness_history = self.simulator.train()
        fitness = fitness_history[:, -1]
        n_sims = self.config["simulator"]["n_sims"]
        params = self.simulator.get_control_params(list(range(n_sims)))
        return fitness, params

    def mutate_robots(self):
        self.child_robots = []
        for r in self.parent_robots:
            mutated_mask = mutate_mask(r["mask"], add_prob=self.mutation_type_prob)
            masses, springs = mask_to_robot(mutated_mask)
            masses = masses * SCALE
            child = {
                "n_masses": int(masses.shape[0]),
                "n_springs": int(springs.shape[0]),
                "masses": masses,
                "springs": springs,
                "mask": mutated_mask,
            }
            self.child_robots.append(child)

    def run(self):
        n_sims = self.config["simulator"]["n_sims"]
        self.fitness_trace = np.zeros((self.num_generations + 1, n_sims), dtype=np.float32)
        self.child_robots = copy.deepcopy(self.parent_robots)
        fs, params = self.run_sim()
        self.fitness_scores = fs
        self.best_control_params = params
        self.fitness_trace[0] = self.fitness_scores
        for gen in range(1, self.num_generations + 1):
            self.mutate_robots()
            fs, params = self.run_sim()

            improved = np.where(fs > self.fitness_scores)[0]
            for j in improved:
                self.parent_robots[j] = self.child_robots[j]
                self.fitness_scores[j] = fs[j]
                self.best_control_params[j] = params[j]

            self.fitness_trace[gen] = self.fitness_scores

    def save(self):
        ranking = np.argsort(self.fitness_scores)[::-1]
        top_3_idxs = ranking[:3].tolist()
        for k, idx in enumerate(top_3_idxs):
            robot = dict(self.parent_robots[idx])
            robot["control_params"] = self.best_control_params[idx]
            robot["max_n_masses"] = self.max_num_masses
            robot["max_n_springs"] = self.max_num_springs
            robot["fitness"] = float(self.fitness_scores[idx])
            np.save(f"robot_{k}.npy", robot, allow_pickle=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    np.random.seed(config["seed"])

    phc = ParallelHillClimber(mutation_type_prob=0.65, num_generations=100, config=config)
    phc.run()
    plot_fitness(phc.fitness_trace, save_path="fitness_trace.png")
    phc.save()
