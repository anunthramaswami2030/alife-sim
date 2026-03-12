from flask import Flask, render_template, Response
from argparse import ArgumentParser
from simulator import Simulator
from utils import load_config
import threading
import time
import json
import numpy as np
import os

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "visualizer_two", "templates"),
    static_folder=os.path.join(BASE_DIR, "visualizer_two", "static"),
)

TARGET_FPS = 60.0
NUM_ROBOTS = 2

state_lock = threading.Lock()
app_state = {
    "step_index": 0,
    "actual_fps": 0.0,
}


@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------------------------------------------------------
# Simulation step
# -----------------------------------------------------------------------------
def step_once():
    global simulator, robots, max_steps, n_masses_cached, n_springs_cached

    t = app_state["step_index"]

    if t >= max_steps:
        simulator.reinitialize_robots()
        app_state["step_index"] = 0
        t = 0

    simulator.compute_com(t)
    simulator.nn1(t)
    simulator.nn2(t)
    simulator.apply_spring_force(t)
    simulator.advance(t + 1)

    x_np = simulator.x.to_numpy()
    act_np = simulator.act.to_numpy()

    robot_data = []

    for robot_idx in range(NUM_ROBOTS):
        n_masses = n_masses_cached[robot_idx]
        n_springs = n_springs_cached[robot_idx]

        positions = x_np[robot_idx, t + 1, :n_masses]
        activations = act_np[robot_idx, t, :n_springs]
        center_of_mass = positions.mean(axis=0)

        robot_data.append({
            "positions": positions,
            "activations": activations,
            "center_of_mass": center_of_mass,
        })

    app_state["step_index"] = t + 1

    return robot_data


# -----------------------------------------------------------------------------
# Stream endpoint
# -----------------------------------------------------------------------------
@app.route("/stream")
def stream():
    global robots, n_masses_cached, n_springs_cached

    def event_stream():

        topology = {
            "type": "topology",
            "robots": [],
        }

        for i, robot in enumerate(robots):
            topology["robots"].append({
                "robot_index": i,
                "springs": robot["springs"].tolist(),
                "n_masses": int(n_masses_cached[i]),
                "n_springs": int(n_springs_cached[i]),
            })

        yield f"data: {json.dumps(topology)}\n\n"

        fps_samples = []
        last_fps_update = time.perf_counter()

        while True:

            frame_start = time.perf_counter()
            target_interval = 1.0 / TARGET_FPS

            robot_data = step_once()

            payload = {
                "type": "step",
                "robots": [],
                "step": app_state["step_index"],
                "fps": app_state["actual_fps"],
            }

            for i, data in enumerate(robot_data):
                payload["robots"].append({
                    "robot_index": i,
                    "positions": data["positions"].tolist(),
                    "activations": data["activations"].tolist(),
                    "center_of_mass": data["center_of_mass"].tolist(),
                })

            yield f"data: {json.dumps(payload)}\n\n"

            frame_end = time.perf_counter()
            work_time = frame_end - frame_start

            sleep_time = target_interval - work_time
            if sleep_time > 0.001:
                time.sleep(sleep_time)

            total_frame_time = time.perf_counter() - frame_start
            if total_frame_time > 0:
                fps_samples.append(1.0 / total_frame_time)

            current_time = time.perf_counter()

            if current_time - last_fps_update >= 0.5:
                if fps_samples:
                    with state_lock:
                        app_state["actual_fps"] = sum(fps_samples) / len(fps_samples)

                    fps_samples = []
                    last_fps_update = current_time

    response = Response(event_stream(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"

    return response


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------
def get_max_dims(robot):
    if "max_n_masses" in robot and "max_n_springs" in robot:
        return int(robot["max_n_masses"]), int(robot["max_n_springs"])

    return int(robot["n_masses"]), int(robot["n_springs"])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--input1", type=str, required=True)
    parser.add_argument("--input2", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")

    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load robots
    # -------------------------------------------------------------------------
    print(f"Loading robot 1 from {args.input1}")
    robot1 = np.load(args.input1, allow_pickle=True).item()

    print(f"Loading robot 2 from {args.input2}")
    robot2 = np.load(args.input2, allow_pickle=True).item()

    robots = [robot1, robot2]

    print(f"Robot1: {robot1['n_masses']} masses, {robot1['n_springs']} springs")
    print(f"Robot2: {robot2['n_masses']} masses, {robot2['n_springs']} springs")

    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------
    config = load_config(args.config)

    max_masses_all = []
    max_springs_all = []

    for robot in robots:
        mm, ms = get_max_dims(robot)
        max_masses_all.append(mm)
        max_springs_all.append(ms)

    config["simulator"]["n_masses"] = max(max_masses_all)
    config["simulator"]["n_springs"] = max(max_springs_all)
    config["simulator"]["n_sims"] = NUM_ROBOTS

    print(
        f"Simulator dims: "
        f"n_sims={NUM_ROBOTS}, "
        f"n_masses={config['simulator']['n_masses']}, "
        f"n_springs={config['simulator']['n_springs']}"
    )

    # -------------------------------------------------------------------------
    # Simulator
    # -------------------------------------------------------------------------
    print("Initializing simulator...")

    simulator = Simulator(
        sim_config=config["simulator"],
        taichi_config=config["taichi"],
        seed=config["seed"],
        needs_grad=False,
    )

    simulator.initialize(
        [robot1["masses"], robot2["masses"]],
        [robot1["springs"], robot2["springs"]],
    )

    # -------------------------------------------------------------------------
    # Control params
    # -------------------------------------------------------------------------
    control_robot_indices = []
    control_params = []

    for i, robot in enumerate(robots):

        if "control_params" in robot:
            control_robot_indices.append(i)
            control_params.append(robot["control_params"])

            print(f"Loaded control params for robot {i}")

        else:
            print(f"No control params for robot {i}")

    if control_robot_indices:
        simulator.set_control_params(control_robot_indices, control_params)

    # -------------------------------------------------------------------------
    # Cache sizes
    # -------------------------------------------------------------------------
    max_steps = simulator.steps[None]

    n_masses_cached = [int(simulator.n_masses[i]) for i in range(NUM_ROBOTS)]
    n_springs_cached = [int(simulator.n_springs[i]) for i in range(NUM_ROBOTS)]

    print("Cached masses:", n_masses_cached)
    print("Cached springs:", n_springs_cached)

    print(f"\nVisualizer running at http://localhost:{args.port}\n")

    app.run(
        host="0.0.0.0",
        port=args.port,
        debug=args.debug,
        threaded=False,
        use_reloader=False,
    )