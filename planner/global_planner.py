import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set


import numpy as np
from param import SHOW_VIS
import matplotlib
if not SHOW_VIS:
    matplotlib.use("Agg")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# from planner.op_solver import NumbaTDOP_Solver  # type: ignore
try:
    from planner.op_solver import NumbaTDOP_Solver  
except:
    from op_solver import NumbaTDOP_Solver  

@dataclass
class Node:
    kind: str  # "macro" or "micro"
    cells: List[Tuple[int, int]]  # covered 1x1 grid cells
    center: Tuple[float, float]  # center in grid coordinates (x, y)


class _HeteroGP:
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        noise_var: np.ndarray,
        length_scale: float=1.0,
        signal_var: float=0.16,
    ):
        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.y_train = np.asarray(y_train, dtype=np.float64)
        self.noise_var = np.asarray(noise_var, dtype=np.float64)
        self.length_scale = float(length_scale)
        self.signal_var = float(signal_var)

        K = self._rbf_kernel(self.X_train, self.X_train, self.length_scale, self.signal_var)
        K[np.diag_indices_from(K)] += self.noise_var + 1e-6
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += np.eye(K.shape[0]) * 1e-3
            L = np.linalg.cholesky(K)
        # Force Fortran-contiguous matrices for stable/fast LAPACK calls.
        Lf = np.asfortranarray(L)
        self.L = Lf
        self.alpha = np.linalg.solve(Lf.T, np.linalg.solve(Lf, self.y_train))

    def predict(self, X_query: np.ndarray, return_std: bool = True):
        X_query = np.asarray(X_query, dtype=np.float64)
        K_s = self._rbf_kernel(self.X_train, X_query, self.length_scale, self.signal_var)
        K_s = np.asfortranarray(K_s)
        mean = K_s.T @ self.alpha
        v = np.linalg.solve(self.L, K_s)
        var = self.signal_var - np.sum(v * v, axis=0)
        var = np.maximum(var, 1e-6)
        if return_std:
            return mean.astype(np.float32), np.sqrt(var).astype(np.float32)
        return mean.astype(np.float32)

    @staticmethod
    def _rbf_kernel(
        X1: np.ndarray, X2: np.ndarray, length_scale: float, signal_var: float
    ) -> np.ndarray:
        x1_sq = np.sum(X1 * X1, axis=1, keepdims=True)
        x2_sq = np.sum(X2 * X2, axis=1, keepdims=True).T
        sqdist = x1_sq + x2_sq - 2.0 * (X1 @ X2.T)
        denom = max(length_scale, 1e-6) ** 2
        return signal_var * np.exp(-0.5 * sqdist / denom)

class DensityMap:
    def __init__(
        self,
        map_path: str,
        grid_res: int = 20,
        add_gaussian: bool = True,
        gaussian_center: Tuple[float, float] = (15.0, 2.0),
        gaussian_sigma: float = 1.0,
        gaussian_amplitude: float = 1.0,
    ):
        self.grid_res = int(grid_res)
        self.density = self._load_density_grid(map_path, self.grid_res)
        if add_gaussian:
            self.density = self._add_gaussian_to_density(
                self.density,
                center_xy=gaussian_center,
                amplitude=gaussian_amplitude,
                sigma=gaussian_sigma,
            )

    def sample_3x3(
        self, pos: Tuple[int, int], sampled: Dict[Tuple[int, int], float]
    ) -> None:
        r0, c0 = pos
        h, w = self.density.shape
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < h and 0 <= c < w:
                    if (r, c) not in sampled:
                        sampled[(r, c)] = float(self.density[r, c])

    @staticmethod
    def _load_density_grid(map_path: str, grid_res: int) -> np.ndarray:
        raw = np.load(map_path)
        h, w = raw.shape
        block = h // grid_res
        if h % grid_res != 0 or w % grid_res != 0:
            raise ValueError(f"Map shape {raw.shape} not divisible by {grid_res}.")

        rock_mask = (raw == 1) | (raw == 2)
        rock_mask = rock_mask.astype(np.float32)

        reshaped = rock_mask.reshape(grid_res, block, grid_res, block)
        density = reshaped.sum(axis=(1, 3)) / float(block * block)

        return density

    @staticmethod
    def _add_gaussian_to_density(
        density: np.ndarray,
        center_xy: Tuple[float, float],
        amplitude: float = 1.0,
        sigma: float = 1.5,
    ) -> np.ndarray:
        grid_res = density.shape[0]
        xs = np.arange(0, grid_res, dtype=np.float32) + 0.5
        ys = np.arange(0, grid_res, dtype=np.float32) + 0.5
        xx, yy = np.meshgrid(xs, ys)
        dx = xx - float(center_xy[0])
        dy = yy - float(center_xy[1])
        bump = amplitude * np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
        out = density + bump
        return np.clip(out, 0.0, 1.0)


class GlobalPlanner:
    def __init__(
        self,
        budget: float = 150.0,
        beta: float = 0.2,
        grid_res: int = 20,
        visualize: bool = True,
        random_seed: int = None,
        grid_size: float = 2.5,

        start: Tuple[int, int] = (0, 0), # The start grid (col, row) for global planner, which is the current robot grid cell. Note the order is (col, row) to be consistent with (x, y) in meters.

        node_var_threshold: float = 0.2,
        use_real: bool = False,
        map_path: str = None,
        dist_decay_factor: float = 0.05,
        gp_length_scale_m: float = 1.0,
        gp_signal_var: float = 0.16,
    ):
        if random_seed is not None:
            np.random.seed(int(random_seed))
        self.random_seed = random_seed

        self._grid_res = grid_res           # 20 means a 20*20 grid
        self.grid_size = float(grid_size)   # each cell is 2.5m x 2.5m

        self.budget = float(budget)
        self.beta = float(beta)

        # `visualize` controls whether figure data is rendered at all.
        # SHOW_VIS only controls whether a live matplotlib window is shown.
        self.visualize = bool(visualize)
        self._show_live_window = bool(SHOW_VIS)
        self.node_var_threshold = float(node_var_threshold)

        self.dist_decay_factor = float(dist_decay_factor)
        self.gp_length_scale_m = float(gp_length_scale_m)
        self.gp_signal_var = float(gp_signal_var)

        if use_real: # here in our real experiment, map is only used for drawing the figure
            if map_path is None: 
                raise ValueError("map_path need to be provided to draw figure.")
            self.map = DensityMap(
                map_path=map_path,
                grid_res=grid_res,
                add_gaussian=False,
            )
        else:   # here map is used for simulation, for example sampling
                # we add a gaussian bump to simulate higher density area
            self.map = DensityMap(
                map_path=map_path,
                grid_res=grid_res,
                add_gaussian=True,
                gaussian_center=(15.0, 2.0),
                gaussian_sigma=1,
                gaussian_amplitude=1,
            )
        
        
        def _grid_points() -> np.ndarray:
            xs = np.arange(0, self._grid_res, dtype=np.float32) + 0.5
            ys = np.arange(0, self._grid_res, dtype=np.float32) + 0.5
            xx, yy = np.meshgrid(xs, ys)
            grid_X = np.stack([xx.ravel(), yy.ravel()], axis=1)
            return grid_X
        self._grid_X = _grid_points()
        self._node_mean_grid: Optional[np.ndarray] = None
        self._node_var_grid: Optional[np.ndarray] = None
        self._node_means: Optional[np.ndarray] = None
        self._node_vars: Optional[np.ndarray] = None
        self._obs_idx: Optional[np.ndarray] = None
        self._use_real = use_real
        self.prob_substrate: Optional[np.ndarray] = None
        self.target_m: Optional[Tuple[float, float]] = None
        self.target_in_grid_unit: Optional[Tuple[int, int]] = None  # single target grid coordinate
        self.target_grids: Optional[Set[Tuple[int, int]]] = None  # set of target grids (macro=4, micro=1)
        self.last_path: Optional[List[Tuple[float, float]]] = None

        # initial macro nodes
        self.macros = []
        for r in range(0, self._grid_res, 2):
            for c in range(0, self._grid_res, 2):
                if self._macro_cells(r, c):
                    self.macros.append((r, c))

        self.sampled: Dict[Tuple[int, int], float] = {}
        self.visited: set = set()
        self.split_macros = set()

        # starting node
        self.current = (int(start[1]), int(start[0]))  
        self.visited.add(self.current)

        print(f"[GlobalPlanner] Starting at grid cell {self.current}.")
        self.remaining_budget = float(budget)

        self.step_count = 0
        self.robot_traj: List[Tuple[float, float]] = [
            (self.current[1] + 0.5, self.current[0] + 0.5)
        ]

        self.fig = None
        self.ax_belief = None
        self.ax_gt = None
        self.ax_mean = None
        self.ax_std = None
        
        if self.visualize:
            self.fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            self.ax_belief, self.ax_gt = axes[0, 0], axes[0, 1]
            self.ax_mean, self.ax_std = axes[1, 0], axes[1, 1]
            if self._show_live_window:
                plt.ion()
                plt.show(block=False)
                plt.pause(0.001)

    def plan(self) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[int, int]], float]:
        if self._use_real:
            self.split_macros = self._update_split_macros_real()
            gp = self._train_gp_hetero()
        else:
            self.split_macros = self._update_split_macros_sim()
            gp = self._train_gp()
        nodes = self._build_nodes()

        if gp is None and self._use_real:
            mean, var = self._belief_stats_for_nodes(nodes)
        else:
            mean, var = self._predict_nodes(gp, nodes)
        rewards = self._compute_rewards(nodes, mean, var)
        path, _score = self._solve_path(nodes, rewards)
        self.last_path = path

        if len(path) >= 2:
            target_center = path[1]
            target_r = int(np.clip(int(target_center[1]), 0, self._grid_res - 1))
            target_c = int(np.clip(int(target_center[0]), 0, self._grid_res - 1))
            self.target_in_grid_unit = (target_r, target_c)
            self.target_m = self._grid_center_to_m(target_center)

            target_grids = None
            for node in nodes:
                # Find the node corresponding to the target center, and use its grids as target_grids.
                if abs(node.center[0] - target_center[0]) < 1e-6 and abs(node.center[1] - target_center[1]) < 1e-6:
                    target_grids = set(node.cells)
                    break
            if target_grids is None:
                target_grids = {(target_r, target_c)}
            self.target_grids = target_grids

            dist = math.hypot(
                target_center[1] - (self.current[0] + 0.5),
                target_center[0] - (self.current[1] + 0.5),
            )
        else:
            self.target_in_grid_unit = None
            self.target_m = None
            self.target_grids = None
            dist = 0.0

        if self.visualize:
            self._update_visualization(gp, nodes, path)

        # Take a step toward the next node in the path (simulation-only)
        if not self._use_real and len(path) >= 2:
            r, c = self.current
            cur_x = c + 0.5
            cur_y = r + 0.5
            dx = path[1][0] - cur_x
            dy = path[1][1] - cur_y
            dr = int(math.copysign(1, dy)) if abs(dy) > 1e-6 else 0
            dc = int(math.copysign(1, dx)) if abs(dx) > 1e-6 else 0
            next_pos = (r + dr, c + dc)
            dist = math.hypot(next_pos[0] - self.current[0], next_pos[1] - self.current[1])
            self.remaining_budget -= dist
            self.current = next_pos
            self.robot_traj.append((self.current[1] + 0.5, self.current[0] + 0.5))
            self.step_count += 1
            print(
                f"[GlobalPlanner] step={self.step_count} heading to node indices {self.current} with distance {dist:.2f}, remaining_budget={self.remaining_budget:.2f}"
            )
        # We add an extra 0.5 grid_size here:
        # - Global planning triggers only when the robot reaches the boundary of the current cell,
        #   so the robot is at the cell boundary (not the center).
        # - dist is computed from the current cell center to the target cell center.
        # - For the local planner, it must first move from the boundary to the current cell center
        #   (half a cell), then travel dist to the target center, so we add 0.5 grid_size.
        # 这里需要额外加 0.5 个 grid_size：
        # - 全局规划只在机器人到达当前网格边界时触发，因此此时机器人位于“当前格子的边界”。
        # - dist 的计算是“当前格子中心”到“目标格子中心”的距离。
        # - 对局部规划而言，还需要先从边界走到当前格子中心（半个格子），
        #   再沿 dist 前往目标格子中心，所以局部规划的预算需要补上 0.5 个 grid_size的预算。
        local_budget_grid = (dist + 0.5)  # local budget in grid units

        return self.target_m, self.target_in_grid_unit, local_budget_grid

    def sync_budget_from_time(self, remaining_time_sec: float, grid_time_sec: float) -> float:
        """
        Sync remaining_budget (grid units) from a real-time budget.
        This keeps the global planner budget consistent with time-based constraints.
        """
        if grid_time_sec <= 0.0:
            raise ValueError("grid_time_sec must be positive.")
        budget_grid = max(0.0, float(remaining_time_sec) / float(grid_time_sec))
        self.remaining_budget = budget_grid
        return budget_grid


    def check_reached_target(self, robot_x: float, robot_y: float) -> bool:
        if not self.target_grids:
            return False
        grid_r_robot, grid_c_robot = self._grid_from_world(robot_x, robot_y)
        reached = (grid_r_robot, grid_c_robot) in self.target_grids

        if reached:
            self.current = (grid_r_robot, grid_c_robot)
            self.visited.add(self.current)
            self.robot_traj.append((self.current[1] + 0.5, self.current[0] + 0.5))
            self.step_count += 1
            return True
        return False

    def update_belief_sim(self, pos: Optional[Tuple[int, int]] = None) -> None:
        if pos is None:
            pos = self.current
        self.map.sample_3x3(pos, self.sampled)
        self.visited.add(pos)

    def update_belief_real(self, prob_substrate: np.ndarray) -> None:
        self.prob_substrate = prob_substrate
        map_h, map_w = prob_substrate.shape
        if map_h % self._grid_res != 0 or map_w % self._grid_res != 0:
            raise ValueError(
                f"Belief map shape {prob_substrate.shape} not divisible by grid_res={self._grid_res}."
            )
        grid_interval_y = map_h // self._grid_res
        grid_interval_x = map_w // self._grid_res
        if grid_interval_y <= 0 or grid_interval_x <= 0:
            raise ValueError("Invalid grid interval computed from belief map.")
        grid_interval_half_y = grid_interval_y // 2
        grid_interval_half_x = grid_interval_x // 2

        mean_grid = np.zeros((self._grid_res, self._grid_res), dtype=np.float32)
        var_grid = np.zeros((self._grid_res, self._grid_res), dtype=np.float32)

        ry = 0
        for cy in range(grid_interval_half_y, map_h - grid_interval_half_y + 1, grid_interval_y):
            cx_idx = 0
            for cx in range(grid_interval_half_x, map_w - grid_interval_half_x + 1, grid_interval_x):
                min_y, max_y = cy - grid_interval_half_y, cy + grid_interval_half_y
                min_x, max_x = cx - grid_interval_half_x, cx + grid_interval_half_x
                sub_slice = prob_substrate[min_y:max_y, min_x:max_x]
                mean_sub = float(np.mean(sub_slice))
                var_sub = float(np.mean(sub_slice * (1.0 - sub_slice)))
                mean_grid[ry, cx_idx] = mean_sub
                var_grid[ry, cx_idx] = var_sub
                cx_idx += 1
            ry += 1


        self._node_mean_grid = mean_grid
        self._node_var_grid = var_grid
        self._node_means = mean_grid.ravel()
        self._node_vars = var_grid.ravel()
        self._obs_idx = np.where(self._node_vars < self.node_var_threshold)[0]

        if self._obs_idx.size > 0:
            obs_cells = {
                (int(idx // self._grid_res), int(idx % self._grid_res)) for idx in self._obs_idx
            }
            self.sampled = {
                (r, c): float(mean_grid[r, c]) for (r, c) in obs_cells
            }
        else:
            self.sampled = {}

    def _update_split_macros_sim(self) -> set:
        split = set()
        sampled = set(self.sampled.keys())
        for r0, c0 in self.macros:
            cells = self._macro_cells(r0, c0)
            if not cells:
                continue
            count = sum((cell in sampled) for cell in cells)
            if count >= 1:
                split.add((r0, c0))
        return split
    
    def _update_split_macros_real(self) -> set:
        split = set()
        if self._obs_idx is None or self._obs_idx.size == 0:
            return split
        obs_cells = {
            (int(idx // self._grid_res), int(idx % self._grid_res)) for idx in self._obs_idx
        }
        for r0, c0 in self.macros:
            cells = self._macro_cells(r0, c0)
            if not cells:
                continue
            count = sum((cell in obs_cells) for cell in cells)
            # Keep the original threshold (=2) for regular 2x2 macros,
            # but allow 1x1 edge macros to be treated consistently.
            split_threshold = min(2, len(cells))
            if count >= split_threshold:
                split.add((r0, c0))
        return split

    def _build_nodes(self) -> List[Node]:
        nodes: List[Node] = []
        for r0, c0 in self.macros:
            cells = self._macro_cells(r0, c0)
            if not cells:
                continue
            if (r0, c0) in self.split_macros and len(cells) > 1:
                for r, c in cells:
                    center = (c + 0.5, r + 0.5)
                    nodes.append(Node("micro", [(r, c)], center))
            else:
                xs = [c + 0.5 for (_, c) in cells]
                ys = [r + 0.5 for (r, _) in cells]
                center = (float(np.mean(xs)), float(np.mean(ys)))
                nodes.append(Node("macro", cells, center))
        return nodes

    def _macro_cells(self, r0: int, c0: int) -> List[Tuple[int, int]]: # Here, cell means the minimum grid unit (1x1), whose size equals to micro node
        cells: List[Tuple[int, int]] = []
        for dr in (0, 1):
            for dc in (0, 1):
                r = r0 + dr
                c = c0 + dc
                if 0 <= r < self._grid_res and 0 <= c < self._grid_res:
                    cells.append((r, c))
        return cells

    def _solve_path(self, nodes: List[Node], rewards: np.ndarray):
        solver_budget = min(self.remaining_budget, 150)
        coords = np.array([node.center for node in nodes], dtype=np.float32)
        current_center = (self.current[1] + 0.5, self.current[0] + 0.5)
        return NumbaTDOP_Solver.solveOP(
            coords, rewards, current_center, solver_budget, seed=self.random_seed
        )

### Getting reward model
    def _train_gp(self):
        if not self.sampled:
            return None
        X = np.array(
            [[c + 0.5, r + 0.5] for (r, c) in self.sampled.keys()], dtype=np.float32
        )
        y = np.array(list(self.sampled.values()), dtype=np.float32)
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(
            noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1)
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, normalize_y=True, optimizer=None
        )
        gp.fit(X, y)
        return gp

    def _train_gp_hetero(self):
        if self._node_means is None or self._node_vars is None:
            return None
        if self._obs_idx is None or self._obs_idx.size == 0:
            return None

        obs_idx = self._obs_idx

        X_train = self._grid_X[obs_idx]
        y_train = self._node_means[obs_idx]
        noise_var = self._node_vars[obs_idx]

        return _HeteroGP(
            X_train,
            y_train,
            noise_var,
            length_scale=self.gp_length_scale_m,
            signal_var=self.gp_signal_var,
        )

    @staticmethod
    def _predict_nodes(gp, nodes: List[Node]) -> Tuple[np.ndarray, np.ndarray]:
        X = np.array([node.center for node in nodes], dtype=np.float32)
        if gp is None:
            mean = np.zeros(len(nodes), dtype=np.float32)
            var = np.ones(len(nodes), dtype=np.float32)
            return mean, var
        mean, std = gp.predict(X, return_std=True)
        var = std**2
        return mean.astype(np.float32), var.astype(np.float32)

    def _belief_stats_for_nodes(self, nodes: List[Node]) -> Tuple[np.ndarray, np.ndarray]:
        if self._node_mean_grid is None or self._node_var_grid is None:
            mean = np.zeros(len(nodes), dtype=np.float32)
            var = np.ones(len(nodes), dtype=np.float32)
            return mean, var
        mean = np.zeros(len(nodes), dtype=np.float32)
        var = np.zeros(len(nodes), dtype=np.float32)
        for i, node in enumerate(nodes):
            r = int(np.clip(int(node.center[1]), 0, self._grid_res - 1))
            c = int(np.clip(int(node.center[0]), 0, self._grid_res - 1))
            mean[i] = self._node_mean_grid[r, c]
            var[i] = self._node_var_grid[r, c]
        return mean, var

    def _compute_rewards(self, nodes: List[Node], mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        rewards = (mean + self.beta * np.sqrt(var)).astype(np.float32)

        # 获取机器人当前在 Grid 坐标系下的中心位置 (x, y)
        # self.current 是 (row, col)，对应 (y, x)
        robot_x = self.current[1] + 0.5
        robot_y = self.current[0] + 0.5

        for i, node in enumerate(nodes):
            # Unified visited handling for both macro and micro nodes.
            # A node is considered exhausted when all of its cells have been visited.
            visited_count = sum((cell in self.visited) for cell in node.cells)
            total_cells = max(1, len(node.cells))
            if visited_count >= total_cells:
                rewards[i] = 0.0
                continue

            # Macro nodes are upweighted, but shrink with remaining unvisited fraction.
            if node.kind == "macro":
                unvisited_ratio = float(total_cells - visited_count) / float(total_cells)
                rewards[i] *= 2.0 * unvisited_ratio

            # === 新增：距离惩罚逻辑 ===
            # 获取节点中心坐标
            node_x, node_y = node.center
            
            # 计算欧几里得距离
            dist = math.hypot(node_x - robot_x, node_y - robot_y)
            
            # 应用指数衰减
            # self.dist_decay_factor 越大，远处节点的奖励降得越快
            decay = math.exp(-self.dist_decay_factor * dist)
            rewards[i] *= decay
            
        return rewards


### For visualization
    @staticmethod
    def _clear_overlays(ax) -> None:
        for coll in list(ax.collections):
            coll.remove()
        for line in list(ax.lines):
            line.remove()
        for patch in list(ax.patches):
            patch.remove()

    def _predict_grid_stats(self, gp) -> Tuple[np.ndarray, np.ndarray]:
        if gp is None:
            if self._use_real and self._node_mean_grid is not None and self._node_var_grid is not None:
                mean = self._node_mean_grid.ravel().astype(np.float32)
                std = np.sqrt(np.maximum(self._node_var_grid, 1e-6)).ravel().astype(np.float32)
            else:
                mean = np.zeros((self._grid_res * self._grid_res,), dtype=np.float32)
                std = np.ones((self._grid_res * self._grid_res,), dtype=np.float32)
        else:
            mean, std = gp.predict(self._grid_X, return_std=True)

            mean = mean.astype(np.float32)
            std = std.astype(np.float32)
        return (
            mean.reshape(self._grid_res, self._grid_res),
            std.reshape(self._grid_res, self._grid_res),
        )

    def _visualize_mean_std(self, ax, data: np.ndarray, visited: set, title: str, cmap: str) -> None:
        if not hasattr(ax, "_im"):
            ax._im = ax.imshow(
                data,
                origin="lower",
                extent=(0, self._grid_res, 0, self._grid_res),
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
            )
            ax._cbar = ax.figure.colorbar(ax._im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlim(0, self._grid_res)
            ax.set_ylim(self._grid_res, 0)
            ax.set_aspect("equal", adjustable="box")
        else:
            ax._im.set_data(data)

        self._clear_overlays(ax)
        if visited:
            sx = [c + 0.5 for (_, c) in visited]
            sy = [r + 0.5 for (r, _) in visited]
            ax.scatter(sx, sy, s=10, color="#ffffff", alpha=0.7)
        ax.set_title(title)

    def _visualize_step(
        self,
        ax_belief,
        ax_gt,
        ucb_grid: np.ndarray,
        nodes: List[Node],
        path_centers: List[Tuple[float, float]],
        robot_traj: List[Tuple[float, float]],
        robot_pos: Tuple[int, int],
        sampled: Dict[Tuple[int, int], float],
    ) -> None:
        if not hasattr(ax_belief, "_im"):
            ax_belief._im = ax_belief.imshow(
                ucb_grid,
                origin="lower",
                extent=(0, self._grid_res, 0, self._grid_res),
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            ax_belief._cbar = ax_belief.figure.colorbar(
                ax_belief._im, ax=ax_belief, fraction=0.046, pad=0.04
            )
            ax_belief.set_xlim(0, self._grid_res)
            ax_belief.set_ylim(self._grid_res, 0)
            ax_belief.set_aspect("equal", adjustable="box")
        else:
            ax_belief._im.set_data(ucb_grid)

        self._clear_overlays(ax_belief)

        if path_centers and len(path_centers) >= 2:
            px = [p[0] for p in path_centers]
            py = [p[1] for p in path_centers]
            ax_belief.plot(px, py, color="#00bcd4", linewidth=2.0, label="Global Path")

        for node in nodes:
            r0, c0 = node.cells[0]
            if node.kind == "macro":
                rect = Rectangle(
                    (c0, r0), 2.0, 2.0, fill=False, edgecolor="#ffa726", linewidth=1.0
                )
            else:
                rect = Rectangle(
                    (c0, r0), 1.0, 1.0, fill=False, edgecolor="#29b6f6", linewidth=0.6
                )
            ax_belief.add_patch(rect)

        if sampled:
            sx = [c + 0.5 for (_, c) in sampled.keys()]
            sy = [r + 0.5 for (r, _) in sampled.keys()]
            ax_belief.scatter(sx, sy, s=10, color="#ffffff", alpha=0.6, label="Sampled")

        robot_center = (robot_pos[1] + 0.5, robot_pos[0] + 0.5)
        ax_belief.scatter(
            [robot_center[0]], [robot_center[1]], s=60, color="#ff5252", label="Robot"
        )

        ax_belief.set_title("UCB + Nodes + Global Path")
        # ax_belief.legend(loc="lower right")

        if not hasattr(ax_gt, "_im"):
            ax_gt._im = ax_gt.imshow(
                self.map.density,
                origin="lower",
                extent=(0, self._grid_res, 0, self._grid_res),
                cmap="cividis",
                vmin=0.0,
                vmax=1.0,
            )
            ax_gt._cbar = ax_gt.figure.colorbar(
                ax_gt._im, ax=ax_gt, fraction=0.046, pad=0.04
            )
            ax_gt.set_xlim(0, self._grid_res)
            ax_gt.set_ylim(self._grid_res, 0)
            ax_gt.set_aspect("equal", adjustable="box")
        else:
            ax_gt._im.set_data(self.map.density)

        self._clear_overlays(ax_gt)
        if robot_traj:
            rx = [p[0] for p in robot_traj]
            ry = [p[1] for p in robot_traj]
            ax_gt.plot(rx, ry, color="#ffb300", linewidth=2.0, label="Robot Trajectory")

        if path_centers and len(path_centers) >= 2:
            px = [p[0] for p in path_centers]
            py = [p[1] for p in path_centers]
            ax_gt.plot(
                px, py, color="#26a69a", linestyle="--", linewidth=1.5, label="Planned Path"
            )

        robot_center = (robot_pos[1] + 0.5, robot_pos[0] + 0.5)
        ax_gt.scatter([robot_center[0]], [robot_center[1]], s=60, color="#e53935")

        ax_gt.set_title("GT Map + Robot Trajectory + Planned Path")

    def _update_visualization(self, gp, nodes: List[Node], path: List[Tuple[float, float]]) -> None:
        if self.fig is None:
            return
        mean_grid, std_grid = self._predict_grid_stats(gp)
        ucb_grid = mean_grid + self.beta * std_grid
        self._visualize_step(
            self.ax_belief,
            self.ax_gt,
            ucb_grid,
            nodes,
            path,
            self.robot_traj,
            self.current,
            self.sampled,
        )
        if self.ax_mean is not None:
            self._visualize_mean_std(self.ax_mean, mean_grid, self.visited, "GP Mean", "viridis")
        if self.ax_std is not None:
            self._visualize_mean_std(self.ax_std, std_grid, self.visited, "GP Std", "magma")
        self.fig.canvas.draw()
        if self._show_live_window:
            plt.pause(0.1)

    def _finalize_visualization(self) -> None:
        if self._show_live_window:
            plt.ioff()
            plt.show()
        elif self.fig is not None:
            plt.close(self.fig)

### utils
    def _grid_center_to_m(self, center_xy: Tuple[float, float]) -> Tuple[float, float]:
        return (center_xy[0] * self.grid_size, center_xy[1] * self.grid_size)

    def _grid_from_world(self, x_m: float, y_m: float) -> Tuple[int, int]:
        # Convert world coordinates (meters) to grid cell indices (row, col).
        r = int(y_m / self.grid_size) 
        c = int(x_m / self.grid_size) 
        r = int(np.clip(r, 0, self._grid_res - 1))
        c = int(np.clip(c, 0, self._grid_res - 1))
        return r, c


def main_loop(planner: GlobalPlanner) -> Dict[Tuple[int, int], float]:
    try:
        while planner.remaining_budget > 0.0:
            planner.update_belief_sim()
            target_m, _target_cell, _ = planner.plan()
            if target_m is None:
                break

    finally:
        if planner.visualize:
            planner._finalize_visualization()
    return planner.sampled


if __name__ == "__main__":
    map_dir = "map/planning_maps/Area_2_map_1/map.npy"
    # map is 50*50 meters with 0.25m resolution, so grid_res=20 means we have a 20*20 grid where each cell is 2.5m*2.5m
    planner = GlobalPlanner(grid_res=20, random_seed=42, start=(0, 19), use_real=False, map_path=map_dir)
    main_loop(planner)
