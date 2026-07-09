import math
import argparse
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp

    _HAS_OR_TOOLS = True
except Exception:
    _HAS_OR_TOOLS = False


@dataclass
class PlannerConfig:
    """
    Configuration for the prior-only global planner.

    Notes:
    - grid_res: number of grid cells per dimension.
    - grid_size: meters per grid cell (grid_interval_m).
    - budget: path budget in grid units (not meters).
    """

    grid_res: int
    grid_size: float
    budget: float

    step_penalty_grid: float = 0.5
    reward_min: float = 0.0

    use_or_tools: bool = True
    objective_use_distance_cost: bool = True
    cost_scale_factor: int = 100
    prize_scale_factor: int = 10000
    solver_time_limit_sec: int = 3

    debug: bool = False
    random_seed: Optional[int] = None


class GlobalPlannerPrior:
    """
    One-shot global planner using static substrate prior.

    This is adapted from global_planner_prior_op_example.py to match the
    main_game_with_prior.py interface.
    """

    def __init__(
        self,
        budget: float,
        grid_res: int,
        grid_size: float,
        start: Tuple[int, int],
        step_penalty_grid: float = 0.5,
        reward_min: float = 0.0,
        use_or_tools: bool = True,
        objective_use_distance_cost: bool = True,
        cost_scale_factor: int = 100,
        prize_scale_factor: int = 10000,
        solver_time_limit_sec: int = 3,
        # Deprecated: kept for backward compatibility. Use show_vis instead.
        visualize: bool = False,
        show_vis: bool = False,
        debug: bool = False,
        random_seed: Optional[int] = None,
    ) -> None:
        if random_seed is not None:
            np.random.seed(int(random_seed))

        self.cfg = PlannerConfig(
            grid_res=int(grid_res),
            grid_size=float(grid_size),
            budget=float(budget),
            step_penalty_grid=float(step_penalty_grid),
            reward_min=float(reward_min),
            use_or_tools=bool(use_or_tools),
            objective_use_distance_cost=bool(objective_use_distance_cost),
            cost_scale_factor=int(cost_scale_factor),
            prize_scale_factor=int(prize_scale_factor),
            solver_time_limit_sec=int(solver_time_limit_sec),
            debug=bool(debug),
            random_seed=random_seed,
        )

        self._grid_res = self.cfg.grid_res
        self.grid_size = self.cfg.grid_size
        self.budget = self.cfg.budget
        self.remaining_budget = float(self.budget)

        self.step_penalty_grid = self.cfg.step_penalty_grid
        self.reward_min = self.cfg.reward_min
        self.use_or_tools = bool(self.cfg.use_or_tools and _HAS_OR_TOOLS)
        self.objective_use_distance_cost = self.cfg.objective_use_distance_cost
        self.cost_scale_factor = self.cfg.cost_scale_factor
        self.prize_scale_factor = self.cfg.prize_scale_factor
        self.solver_time_limit_sec = self.cfg.solver_time_limit_sec
        # Keep `self.visualize` as alias for compatibility with existing callers.
        self.visualize = bool(visualize)
        # `show_vis` controls interactive display only; plotting itself is always generated.
        self.show_vis = bool(show_vis or self.visualize)
        self.debug = self.cfg.debug

        self.current = (int(start[1]), int(start[0]))
        self.visited: Set[Tuple[int, int]] = {self.current}

        self._cell_size: Optional[float] = None
        self.prob_substrate: Optional[np.ndarray] = None
        self.reward_grid: Optional[np.ndarray] = None

        self._path_nodes_m: List[Tuple[float, float]] = []
        self._path_cells: List[Tuple[int, int]] = []
        self._path_grid_units: List[Tuple[float, float]] = []
        self._path_idx: int = 0

        self.target_m: Optional[Tuple[float, float]] = None
        self.target_in_grid_unit: Optional[Tuple[int, int]] = None
        self.target_grids: Optional[Set[Tuple[int, int]]] = None
        self.last_path: Optional[List[Tuple[float, float]]] = None

        self._fig = None
        self._ax = None

    # -------------------------
    # Public API (compat layer)
    # -------------------------
    def plan_global_path(self, prob_substrate: np.ndarray):
        if self._path_nodes_m:
            return self._fig
        self.prob_substrate = np.asarray(prob_substrate, dtype=np.float32)
        self._infer_cell_size(self.prob_substrate)

        nodes_m, prizes, node_cells = self._build_nodes_and_prizes(self.prob_substrate)

        path_indices = None
        if self.use_or_tools and len(nodes_m) > 1:
            path_indices = self._solve_orienteering(prizes, nodes_m)

        if not path_indices:
            path_indices = self._build_path_greedy(prizes, nodes_m)

        self._set_path_from_indices(path_indices, nodes_m, node_cells)
        # Always generate/update figure for downstream saving.
        self._update_visualization()
        return self._fig

    def retrieve_target(self) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[int, int]], float]:
        if not self._path_nodes_m or self._path_idx + 1 >= len(self._path_nodes_m):
            self.target_in_grid_unit = None
            self.target_m = None
            self.target_grids = None
            self.last_path = None
            return None, None, 0.0

        next_node_m = self._path_nodes_m[self._path_idx + 1]
        next_cell = self._path_cells[self._path_idx + 1]

        self.target_in_grid_unit = next_cell
        self.target_m = next_node_m
        self.target_grids = {next_cell}
        self.last_path = self._path_grid_units[self._path_idx :]

        current_m = self._path_nodes_m[self._path_idx]
        dist_m = math.hypot(next_node_m[0] - current_m[0], next_node_m[1] - current_m[1])
        local_budget_grid = dist_m / self.grid_size + self.step_penalty_grid

        return self.target_m, self.target_in_grid_unit, local_budget_grid

    def check_reached_target(self, robot_x: float, robot_y: float) -> bool:
        if not self.target_grids:
            return False
        grid_r, grid_c = self._grid_from_world(robot_x, robot_y)
        reached = (grid_r, grid_c) in self.target_grids
        if reached:
            self.current = (grid_r, grid_c)
            self.visited.add(self.current)
            if self._path_idx + 1 < len(self._path_cells):
                if self._path_cells[self._path_idx + 1] == self.current:
                    self._path_idx += 1
            self.last_path = self._path_grid_units[self._path_idx :]
        return reached

    def sync_budget_from_time(self, remaining_time_sec: float, grid_time_sec: float) -> float:
        if grid_time_sec <= 0.0:
            raise ValueError("grid_time_sec must be positive.")
        self.remaining_budget = max(0.0, float(remaining_time_sec) / float(grid_time_sec))
        return self.remaining_budget

    # -------------------------
    # Path construction helpers
    # -------------------------
    def _infer_cell_size(self, prob_substrate: np.ndarray) -> None:
        h, w = prob_substrate.shape
        if h % self._grid_res != 0 or w % self._grid_res != 0:
            raise ValueError(
                f"Belief map shape {prob_substrate.shape} not divisible by grid_res={self._grid_res}."
            )
        map_size_m = self._grid_res * self.grid_size
        cell_size_x = map_size_m / float(w)
        cell_size_y = map_size_m / float(h)
        if abs(cell_size_x - cell_size_y) > 1e-6:
            raise ValueError(
                f"Non-square cell size inferred: {cell_size_x:.6f} vs {cell_size_y:.6f}."
            )
        self._cell_size = cell_size_x

    def _build_nodes_and_prizes(
        self, prob_substrate: np.ndarray
    ) -> Tuple[List[Tuple[float, float]], np.ndarray, List[Tuple[int, int]]]:
        h, w = prob_substrate.shape
        block_h = h // self._grid_res
        block_w = w // self._grid_res
        area = float(block_h * block_w)

        reward_grid = np.zeros((self._grid_res, self._grid_res), dtype=np.float32)
        nodes_m: List[Tuple[float, float]] = []
        prizes: List[float] = []
        node_cells: List[Tuple[int, int]] = []

        start_c = self.current[1]
        start_r = self.current[0]
        start_m = ((start_c + 0.5) * self.grid_size, (start_r + 0.5) * self.grid_size)
        nodes_m.append(start_m)
        prizes.append(0.0)
        node_cells.append(self.current)

        for r in range(self._grid_res):
            r0 = r * block_h
            r1 = r0 + block_h
            for c in range(self._grid_res):
                if (r, c) == self.current:
                    # Start node has already been inserted as index 0.
                    continue
                c0 = c * block_w
                c1 = c0 + block_w

                block = prob_substrate[r0:r1, c0:c1]
                weight_sum = float(block.sum())
                if weight_sum <= 0.0:
                    continue

                prize = weight_sum / area
                reward_grid[r, c] = prize
                if prize < self.reward_min:
                    continue

                # Use geometric center of each global grid cell, not weighted centroid.
                node_x_m = (c + 0.5) * self.grid_size
                node_y_m = (r + 0.5) * self.grid_size

                nodes_m.append((node_x_m, node_y_m))
                prizes.append(prize)
                node_cells.append((r, c))

        self.reward_grid = reward_grid
        return nodes_m, np.asarray(prizes, dtype=np.float32), node_cells

    def _solve_orienteering(self, prizes: np.ndarray, nodes_m: List[Tuple[float, float]]) -> Optional[List[int]]:
        if not _HAS_OR_TOOLS:
            return None
        if len(nodes_m) <= 1:
            return [0]

        coords = np.asarray(nodes_m, dtype=np.float32)
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        cost_matrix = (dist * self.cost_scale_factor).astype(int)

        manager = pywrapcp.RoutingIndexManager(cost_matrix.shape[0], 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(cost_matrix[from_node][to_node])

        distance_callback_index = routing.RegisterTransitCallback(distance_callback)

        if self.objective_use_distance_cost:
            routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)
        else:
            zero_callback_index = routing.RegisterTransitCallback(lambda _f, _t: 0)
            routing.SetArcCostEvaluatorOfAllVehicles(zero_callback_index)

        scaled_prizes = (prizes * self.prize_scale_factor).astype(int)
        for node_idx in range(1, len(scaled_prizes)):
            routing.AddDisjunction([manager.NodeToIndex(node_idx)], int(scaled_prizes[node_idx]))

        budget_m = max(0.0, self.budget) * self.grid_size
        budget_scaled = int(budget_m * self.cost_scale_factor)

        routing.AddDimension(
            distance_callback_index,
            0,
            budget_scaled,
            True,
            "Distance",
        )

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_params.time_limit.FromSeconds(self.solver_time_limit_sec)

        solution = routing.SolveWithParameters(search_params)
        if not solution:
            return None

        path_indices: List[int] = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            path_indices.append(node_idx)
            index = solution.Value(routing.NextVar(index))

        if self.debug:
            print(f"[GlobalPlannerPrior] OR-Tools path indices: {path_indices}")

        return path_indices

    def _build_path_greedy(self, prizes: np.ndarray, nodes_m: List[Tuple[float, float]]) -> List[int]:
        remaining_m = max(0.0, self.budget) * self.grid_size
        path: List[int] = [0]
        visited = {0}
        current_idx = 0

        while True:
            best_idx = None
            best_score = -1.0
            best_cost = 0.0

            for i in range(1, len(nodes_m)):
                if i in visited:
                    continue
                reward = float(prizes[i])
                if reward < self.reward_min:
                    continue
                dist = math.hypot(
                    nodes_m[i][0] - nodes_m[current_idx][0],
                    nodes_m[i][1] - nodes_m[current_idx][1],
                )
                cost = dist + self.step_penalty_grid * self.grid_size
                if cost > remaining_m:
                    continue
                score = reward / max(cost, 1e-6)
                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_cost = cost

            if best_idx is None:
                break

            path.append(best_idx)
            visited.add(best_idx)
            remaining_m -= best_cost
            current_idx = best_idx

        if self.debug:
            print(f"[GlobalPlannerPrior] Greedy path indices: {path}")

        return path

    def _set_path_from_indices(
        self,
        path_indices: List[int],
        nodes_m: List[Tuple[float, float]],
        node_cells: List[Tuple[int, int]],
    ) -> None:
        if not path_indices:
            self._path_nodes_m = []
            self._path_cells = []
            self._path_grid_units = []
            self.last_path = None
            return

        raw_path_nodes_m = [nodes_m[i] for i in path_indices]
        raw_path_cells = [node_cells[i] for i in path_indices]
        dedup_nodes_m: List[Tuple[float, float]] = []
        dedup_cells: List[Tuple[int, int]] = []
        for node_m, cell in zip(raw_path_nodes_m, raw_path_cells):
            if dedup_cells and cell == dedup_cells[-1]:
                continue
            dedup_nodes_m.append(node_m)
            dedup_cells.append(cell)

        self._path_nodes_m = dedup_nodes_m
        self._path_cells = dedup_cells
        self._path_grid_units = [
            (node[0] / self.grid_size, node[1] / self.grid_size)
            for node in self._path_nodes_m
        ]
        self._path_idx = 0
        self.last_path = self._path_grid_units[:]

    # -------------------------
    # Visualization
    # -------------------------
    def _update_visualization(self) -> None:
        if self.reward_grid is None:
            return
        if self._fig is None or self._ax is None:
            if self.show_vis:
                plt.ion()
            else:
                # Headless-safe fallback when caller does not preconfigure MPLBACKEND.
                try:
                    plt.switch_backend("Agg")
                except Exception:
                    pass
            self._fig, self._ax = plt.subplots(1, 1, figsize=(7, 6))

        ax = self._ax
        ax.clear()

        reward = self.reward_grid
        im = ax.imshow(
            reward,
            origin="lower",
            extent=(0, self._grid_res, 0, self._grid_res),
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title("Global Planner Prior (Centroid Nodes)")
        ax.set_xlabel("Grid Col")
        ax.set_ylabel("Grid Row")
        ax.set_xlim(0, self._grid_res)
        ax.set_ylim(self._grid_res, 0)
        ax.set_aspect("equal", adjustable="box")

        if self._path_grid_units:
            xs = [p[0] for p in self._path_grid_units]
            ys = [p[1] for p in self._path_grid_units]
            ax.plot(xs, ys, "-o", color="white", markersize=4, linewidth=1.5)
            ax.plot(xs[0], ys[0], marker="*", color="red", markersize=10)

        self._fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self._fig.tight_layout()
        self._fig.canvas.draw()
        if self.show_vis:
            plt.show(block=False)
            self._fig.canvas.flush_events()

    def get_visualization_figure(self):
        return self._fig

    # -------------------------
    # Geometry helpers
    # -------------------------
    def _grid_from_world(self, x_m: float, y_m: float) -> Tuple[int, int]:
        c = int(x_m / self.grid_size)
        r = int(y_m / self.grid_size)
        c = max(0, min(self._grid_res - 1, c))
        r = max(0, min(self._grid_res - 1, r))
        return r, c


if __name__ == "__main__":
    from param import MAP_DIR
    parser = argparse.ArgumentParser(description="Smoke test for GlobalPlannerPrior.")
    # parser.add_argument("--map-npy", type=str, required=True, help="Path to map npy file.")
    parser.add_argument("--map-npy", type=str, default=MAP_DIR, help="Path to map npy file.")
    parser.add_argument("--cell-size", type=float, default=0.25, help="Map cell size in meters.")
    parser.add_argument("--grid-size", type=float, default=2.5, help="Global grid size in meters.")
    parser.add_argument("--budget", type=float, default=150.0, help="Budget in grid units.")
    parser.add_argument("--start-x", type=float, default=0.0, help="Start x in meters.")
    parser.add_argument("--start-y", type=float, default=0.0, help="Start y in meters.")
    parser.add_argument("--solver-time-limit-sec", type=int, default=3, help="OR-Tools solver time limit.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="(Deprecated) Alias of --show-vis.",
    )
    parser.add_argument("--show-vis", action="store_true", help="Show interactive matplotlib window.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs.")
    args = parser.parse_args()

    map_array = np.load(args.map_npy)
    prob_substrate = (map_array > 0).astype(np.float32)

    map_size_m = map_array.shape[1] * args.cell_size
    grid_res = int(round(map_size_m / args.grid_size))
    grid_res = max(1, grid_res)

    start_r = int(args.start_y / args.grid_size) if args.grid_size > 0 else 0
    start_c = int(args.start_x / args.grid_size) if args.grid_size > 0 else 0
    start_r = max(0, min(grid_res - 1, start_r))
    start_c = max(0, min(grid_res - 1, start_c))

    planner = GlobalPlannerPrior(
        budget=float(args.budget),
        grid_res=grid_res,
        grid_size=float(args.grid_size),
        start=(start_c, start_r),
        show_vis=bool(args.show_vis),
        visualize=bool(args.visualize),
        debug=bool(args.debug),
        solver_time_limit_sec=int(args.solver_time_limit_sec),
    )
    planner.plan_global_path(prob_substrate)

    print(
        f"[GlobalPlannerPrior] grid_res={grid_res}, budget={planner.budget}, "
        f"path_nodes={len(planner._path_nodes_m)} (or-tools={planner.use_or_tools})"
    )

    if planner.show_vis:
        plt.ioff()
        plt.show()
