import numpy as np

DEBUG = False
SHOW_VIS = True
DEBUG_SHOW_INTEREST_GRIDS = False
RECORD_FRAMES = False
OUTPUT_DIR = "experiment_results"


GAME_FPS = 2

# 1. Map parameters
CELL_SIZE = 0.25  # meters


# 2. Robot sensor parameters


# FLS: long-range substrate scouting
FLS_RANGE = 6.0   # meters
FLS_FOV_DEG = 90  # degrees

# FLC: mid-range target scouting
FLC_RANGE = 2.5   # meters
FLC_FOV_DEG = 60  # degrees

# DLC: downward-looking confirmation footprint, square side length in meters
DLC_FOOTPRINT = 1.0


# FLS detection model
def fls_true_positive_rate(d_norm):
    return 1.0 - 0.1 * d_norm


def fls_false_positive_rate(d_norm):
    return 0.0 + 0.1 * d_norm


# FLC detection model
def flc_true_positive_rate(d_norm):
    return 1.0 - 0.15 * d_norm


def flc_false_positive_rate(d_norm):
    return 0.0 + 0.15 * d_norm


# Backward compat aliases (deprecated: prefer fls_* / flc_*)
true_positive_rate = flc_true_positive_rate
false_positive_rate = flc_false_positive_rate

CONF_THRESHOLD = 0.8  # threshold for rendering / binary mask


# 3. Local planner parameters
NEXEC = 4  # Number of local-planner control steps executed before replanning

MAX_VELOCITY = 0.5  # m/s
MAX_ANGULAR_VELOCITY = 1.0  # rad/s

# Entropy-map snapshot padding, in cells
SUBSTRATE_ENTROPY_MAP_PADDING = int(0.8 * FLS_RANGE / CELL_SIZE)
CORAL_ENTROPY_MAP_PADDING = int(0.5 * FLC_RANGE / CELL_SIZE)

N_CANDIDATES_TO_OPTIMIZE = 15  # Keep only the nearest target candidates
PLANNING_TIMESTEP_SIZE = 0.5  # seconds
ENTROPY_MAP_DOWNSAMPLE = 2  # (deprecated) for backward compatibility
FLC_ENTROPY_MAP_DOWNSAMPLE = 2
FLS_ENTROPY_MAP_DOWNSAMPLE = 4

N_INFO = 10  # Number of information-field rollout steps

# Number of observations per second. Sonar and camera use the same frequency.
OBSERVATION_FREQUENCY = 2
OBSERVATION_TIMES_PER_STEP = PLANNING_TIMESTEP_SIZE * OBSERVATION_FREQUENCY

GRID_TIME_SEC = 6  # Seconds assigned to each global-grid transition
# One global-grid step corresponds to a local-planner time budget. This value should
# cover both travel time and sensing/search time inside the target grid.

# Long-distance cruise mode threshold for the local planner
CRUISE_DIST_THRESHOLD = 6.0  # meters


from dataclasses import dataclass


@dataclass
class GlobalPlannerConfig:
    """Hyperparameters used by the global planner."""

    # Grid-node sampling interval, in meters
    grid_interval_m: float = 2

    # Path budget in global-grid steps. It is set from CLI budget in main_game.py.
    path_budget: float = 0

    # Exploration-exploitation tradeoff
    ucb_beta: float = 0.6

    gp_length_scale_m: float = 1.0
    gp_signal_var: float = 0.16

    node_var_threshold: float = 0.2

    dist_decay_factor: float = 0.0


# 5. Manual control parameters (not used in autonomous mode)
KEY_VELOCITY = 0.5  # m/s
KEY_ANGULAR_VELOCITY = 1.0  # rad/s


# 6. Parameters for MCTS
SAVE_MCTS_TRAJ_IMAGES = False
SHOW_MCTS_TRAJ_PLOT = True
SEARCH_DEPTH = 20
MCTS_SUBSTRATE_ENTROPY_MAP_PADDING = int(5 * FLS_RANGE / CELL_SIZE)
MCTS_CORAL_ENTROPY_MAP_PADDING = int(5 * FLC_RANGE / CELL_SIZE)


# 7. Parameters for planner with prior
SOLVER_TIME_LIMIT_SEC = 100  # seconds per solver call
GRID_TIME_SEC_FOR_PRIOR = 5  # seconds per grid for prior-map experiments
