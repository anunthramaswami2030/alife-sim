from simulator import Simulator
from robot import load_robots, mutate_mask, mask_to_robot, crossover, SCALE, MASK_DIM
import numpy as np
import copy
from utils import load_config
from argparse import ArgumentParser
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 150

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
    plt.title("Parallel Hill Climber With Genetic Crossing")
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

def plot_robot(robot):
    masses = robot["masses"]
    springs = robot["springs"]
    fig = plt.figure(figsize=(6, 6))
    for spring in springs:
        plt.plot([masses[spring[0], 0], masses[spring[1], 0]], [masses[spring[0], 1], masses[spring[1], 1]], color="blue", alpha=0.5)
    plt.scatter(masses[:, 0], masses[:, 1], color="red")
    plt.axis("equal")
    plt.show()

class GeneticAlgorithm:
    def __init__(self, mutation_type_prob, num_generations, n_elites, tournament_size, config, rng):
        self.mutation_type_prob = mutation_type_prob
        self.num_generations = num_generations
        self.n_elites = n_elites
        self.tournament_size = tournament_size
        self.config = config
        self.rng = rng

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

    def tournament_select(self):
        idx = self.rng.choice(len(self.fitness_scores), size=self.tournament_size, replace=False)
        winner = idx[np.argmax(self.fitness_scores[idx])]
        return winner
    
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
        ranking = np.argsort(self.fitness_scores)[::-1]
        top_idxs = ranking[:self.n_elites]
        self.child_robots = [copy.deepcopy(self.parent_robots[i]) for i in top_idxs]
        for i in range(self.n_elites, self.config["simulator"]["n_sims"]):
            p1 = self.parent_robots[self.tournament_select()]
            p2 = self.parent_robots[self.tournament_select()]
            crossed_mask = crossover(p1["mask"], p2["mask"], rng = self.rng)
            mutated_mask = mutate_mask(crossed_mask , add_prob=self.mutation_type_prob,rng = self.rng)
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
            fs_union = np.concatenate([fs, self.fitness_scores])
            robot_union = self.child_robots + self.parent_robots
            params_union = params + self.best_control_params
            best_idx = np.argsort(fs_union)[-n_sims:][::-1]
            self.parent_robots = [robot_union[i] for i in best_idx]
            self.fitness_scores = fs_union[best_idx]
            self.best_control_params = [params_union[i] for i in best_idx]
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
        plot_robot(self.parent_robots[0])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    np.random.seed(config["seed"])
    rng = np.random.default_rng()
    phc = GeneticAlgorithm(mutation_type_prob=0.65, num_generations=10, config=config, tournament_size= 4, n_elites=3, rng = rng)
    phc.run()
    plot_fitness(phc.fitness_trace, save_path="fitness_trace.png")
    phc.save()
