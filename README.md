# HIMoS

HIMoS is the codebase for **Hierarchical Multi-Modal Planning for Fixed-Altitude Sparse Target Search and Sampling**. It simulates an AUV searching for sparse coral targets on fixed-altitude reef maps, using a hierarchical planner that combines long-horizon region selection with short-horizon multi-modal sensing and sampling.

<p align="center">
  <img src="fig/introduction.png" alt="HIMoS motivation and overview" width="560">
</p>

## Project Overview

HIMoS targets sparse benthic search-and-sampling missions where exhaustive coverage is inefficient and high-altitude vision is unreliable. The simulator models a fixed-altitude AUV with three sensing modalities:

- **FLS**: forward-looking sonar for long-range substrate scouting.
- **FLC**: front-looking camera for mid-range coral target scouting.
- **DLC**: down-looking camera for close-range deterministic target confirmation.

The planner is hierarchical:

- **Global Planner**: selects promising regions by solving an orienteering-style routing problem over the substrate belief map.
- **Local Planner**: optimizes short-horizon trajectories that trade off sonar exploration, visual target search, and DLC confirmation.

The robot runs in a closed loop: sense, update belief, locally replan, and request a new global target when the current region is reached.

## Visuals

**Simulation and performance**

<p align="center">
  <img src="fig/himos.gif" alt="HIMoS simulation dashboard" width="580">
</p>

**Field data and planning maps**

<table>
  <tr>
    <td width="50%" align="center">
      <img src="fig/experiment_field.png" alt="Experiment field">
    </td>
    <td width="50%" align="center">
      <img src="fig/easy_medium_hard_maps.png" alt="Easy, medium, and hard planning maps">
    </td>
  </tr>
  <tr>
    <td align="center">Survey field and dataset source</td>
    <td align="center">Easy, medium, and hard planning maps</td>
  </tr>
</table>

## Repository Structure

```text
.
├── main_game.py        # Main simulator and autonomous planning loop
├── param.py            # Sensor, robot, planner, visualization, and logging parameters
├── himos.yml           # Conda environment file with runtime dependencies
├── planner/
│   ├── global_planner.py      # Adaptive global planner and OP target selection
│   ├── local_planner.py       # CasADi-based local trajectory optimizer
│   ├── belief_map.py          # Bayesian belief update and local map snapshots
│   ├── op_solver.py           # Numba-accelerated OP heuristic solver
│   └── mcts_planner.py        # MCTS baseline utilities
├── simulation/
│   ├── coral_map.py           # Ground-truth map and sensor observation simulation
│   ├── robot.py               # Omnidirectional AUV kinematics
│   └── input_controller.py    # Manual control helper
└── map/
    ├── README.md              # Map data and planning-map generation notes
    ├── process_segmentation_50m.py
    └── planning_maps/         # Generated 50 m x 50 m planning maps
```

## Installation

Create the environment from the provided conda file:

```bash
conda env create -f himos.yml
conda activate ml
```

The environment file includes the packages used by the simulator and planners:
`numpy`, `pygame`, `matplotlib`, `scikit-learn`, `numba`, and `casadi`.
You can verify the installation with:

```bash
python -c "import numpy, pygame, matplotlib, sklearn, numba, casadi; print('HIMoS dependencies OK')"
```

If the `ml` environment already exists, update it instead:

```bash
conda env update -f himos.yml --prune
conda activate ml
```

The code is written for Python 3.8. `casadi` is required by the local nonlinear optimizer, `numba` accelerates the orienteering solver, `scikit-learn` provides the Gaussian-process utilities used by the global planner, and `pygame` provides the simulator dashboard.

## Quick Start

The project entry point is [main_game.py](main_game.py). Core experiment and planner parameters are collected in [param.py](param.py). Map data and map-generation details are described separately in [map/README.md](map/README.md).

Run the default autonomous HIMoS simulation:

```bash
python main_game.py
```

The default run uses:

- time budget: `2000` seconds
- map: `map/planning_maps/Area_2_map_1/map.npy`
- start pose: index `0`

You can change these from the command line:

```bash
python main_game.py -budget 1000 -map_index 5 -start_pose 2
```

Arguments:

- `-budget`: total mission time budget in simulated seconds.
- `-map_index`: selects `map/planning_maps/Area_2_map_<index>/map.npy`.
- `-start_pose`: selects one of four corner start poses, indexed `0` to `3`.

While the Pygame window is open, press `G` to toggle the global-planner interest-grid overlay. After a run ends, press `Q`, `Esc`, or close the window to exit.

## Headless or Batch Runs

For non-interactive runs, edit [param.py](param.py):

```python
SHOW_VIS = False
```

When `SHOW_VIS` is disabled, `main_game.py` sets the SDL video driver to `dummy`, so the simulator can run without opening a display window. `pygame` is still required because the simulator uses Pygame surfaces for rendering and saving output images.

## Troubleshooting

If you see `ModuleNotFoundError`, make sure the conda environment is active:

```bash
conda activate ml
```

If `python` is not found before activation, use the command only after activating the conda environment. The default run expects the included map file at `map/planning_maps/Area_2_map_1/map.npy`.

## Outputs

Each run creates an experiment folder under `experiment_results/`, named like:

```text
experiment_results/exp_map<map_index>_start<start_pose>_budget<budget>_<timestamp>/
```

The saved files include:

- `summary.txt`: final confirmation ratio, path length, final pose, and run metadata.
- `coral_timeseries.csv`: confirmed-coral count and robot pose over simulated time.
- `trajectory.npy` and `trajectory.txt`: executed robot trajectory.
- `gt_with_trajectory.png`: ground-truth map with executed trajectory.
- `simulation_observed.png`: final observed map.
- `frames/`, `frames_full_observed/`, `frames_global_planner/`: optional frame outputs, saved only when `RECORD_FRAMES = True` in [param.py](param.py).

## Important Parameters

Most experiment knobs are in [param.py](param.py):

- `FLS_RANGE`, `FLS_FOV_DEG`: sonar range and field of view.
- `FLC_RANGE`, `FLC_FOV_DEG`: forward camera range and field of view.
- `DLC_FOOTPRINT`: down-looking confirmation footprint size.
- `SHOW_VIS`, `RECORD_FRAMES`, `OUTPUT_DIR`: visualization and logging options.

The total mission budget is set from the command line in [main_game.py](main_game.py), then converted to the global-planner path budget with:

```python
path_budget = TOTAL_TIME_BUDGET_SECS / GRID_TIME_SEC
```

## Paper

```bibtex
@misc{chen2026hierarchicalmultimodalplanningfixedaltitude,
      title={Hierarchical Multi-Modal Planning for Fixed-Altitude Sparse Target Search and Sampling}, 
      author={Lingpeng Chen and Yuchen Zheng and Apple Pui-Yi Chui and Junfeng Wu and Ziyang Hong},
      year={2026},
      eprint={2603.08336},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2603.08336}, 
}
```
