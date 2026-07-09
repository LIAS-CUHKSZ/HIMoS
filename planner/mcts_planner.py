"""MCTS planner focused on information gain for coral exploration.

核心目标：
- 不引入全局代价或启发式成本
- 仅使用信息增益（熵降低 + 置信确认）作为奖励
"""

import math
import time
from typing import List, Optional, Tuple

import numpy as np

# 类型别名：只用于可读性（不改变数据结构）
Pose = np.ndarray
Action = np.ndarray

# 奖励权重（保持与原逻辑一致）
SUBSTRATE_ENTROPY_WEIGHT = 10.0
CORAL_ENTROPY_WEIGHT = 100.0
CONFIRMATION_REWARD = 1000.0
CONFIRMATION_DIST_REWARD = 1000.0
CONFIRMATION_THRESHOLD = 0.6


def _local_frame(pose: Pose, grid_x: np.ndarray, grid_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """将世界坐标网格转到机器人局部坐标系，返回(local_x, local_y)。"""
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    dx = grid_x - x
    dy = grid_y - y
    cos_t = math.cos(-theta)
    sin_t = math.sin(-theta)
    local_x = dx * cos_t - dy * sin_t
    local_y = dx * sin_t + dy * cos_t
    return local_x, local_y


def _sector_mask_from_grid(
    pose: Pose,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    max_dist: float,
    fov_rad: float,
) -> np.ndarray:
    """基于局部坐标系，返回扇形视场内的mask。"""
    local_x, local_y = _local_frame(pose, grid_x, grid_y)
    dist_sq = local_x * local_x + local_y * local_y
    angle = np.abs(np.arctan2(local_y, local_x))
    return (dist_sq <= max_dist * max_dist) & (local_x > 0.0) & (angle <= (fov_rad / 2.0))


def _square_mask_from_grid(
    pose: Pose,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    half_side: float,
) -> np.ndarray:
    """基于局部坐标系，返回机器人中心的正方形足迹mask。"""
    local_x, local_y = _local_frame(pose, grid_x, grid_y)
    return (np.abs(local_x) <= half_side) & (np.abs(local_y) <= half_side)


class MCTSNode:
    """Lightweight MCTS node storing only pose and tree stats (no map copies)."""

    def __init__(
        self,
        pose: Pose,
        depth: int,
        parent: Optional["MCTSNode"] = None,
        action: Optional[Action] = None,
        untried_actions: Optional[List[Action]] = None,
    ):
        self.pose = np.array(pose, dtype=np.float32)
        self.depth = depth
        self.parent = parent
        self.action = action
        self.children: List[MCTSNode] = []
        self.visit_count = 0
        self.total_value = 0.0
        self.untried_actions = list(untried_actions) if untried_actions is not None else []

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0


class MCTSPlanner:
    """
    Monte Carlo Tree Search planner focused on Pure Information Gain.
    - No global waypoints, no heuristic cost-to-go.
    - Reward is driven only by expected confirmation + entropy reduction.
    """

    def __init__(
        self,
        cell_size: float,
        fls_range: float,
        fls_fov_deg: float,
        flc_range: float,
        flc_fov_deg: float,
        dlc_footprint: float,
        max_velocity: float,
        max_angular_velocity: float,
        dt: float,
        search_depth: int = 8,
        uct_c: float = 1.2,
        debug: bool = False,
        seed: Optional[int] = None,
    ):
        self.cell_size = float(cell_size)
        self.fls_range = float(fls_range)
        self.flc_range = float(flc_range)
        self.fls_fov_rad = math.radians(fls_fov_deg)
        self.flc_fov_rad = math.radians(flc_fov_deg)
        self.dlc_half = float(dlc_footprint) / 2.0
        self.max_velocity = float(max_velocity)
        self.max_angular_velocity = float(max_angular_velocity)
        self.dt = float(dt)
        self.search_depth = int(search_depth)
        self.uct_c = float(uct_c)
        self.debug = bool(debug)
        self.rng = np.random.default_rng(seed)

        self.actions = self.get_available_actions()

        # Cached map data per planning call
        self._log_odds_s = None
        self._prob_coral = None
        self._confirmed_mask = None
        self._confirmation_map = None
        self._confirmation_threshold = None
        self._fls_informed_mask = None
        self._flc_informed_mask = None
        self._last_root = None
        self._entropy_substrate = None
        self._entropy_coral = None
        self._fls_origin = None
        self._flc_origin = None
        self._fls_grid_x = None
        self._fls_grid_y = None
        self._flc_grid_x = None
        self._flc_grid_y = None

        # Stats
        self.last_iterations = 0
        self.last_plan_time_sec = 0.0

    def get_available_actions(self) -> List[np.ndarray]:
        """Return the discrete motion primitives (~17 actions)."""
        v = self.max_velocity
        w = self.max_angular_velocity
        w_half = 0.5 * w
        w_quarter = 0.25 * w
        v_diag = v / math.sqrt(2.0)

        actions = [
            # 8 pure translations (unit-speed, omega=0)
            np.array([v, 0.0, 0.0], dtype=np.float32),    # forward
            # np.array([-v, 0.0, 0.0], dtype=np.float32),   # backward
            # np.array([0.0, v, 0.0], dtype=np.float32),    # left
            # np.array([0.0, -v, 0.0], dtype=np.float32),   # right
            np.array([v_diag, v_diag, 0.0], dtype=np.float32),     # fwd-left
            np.array([v_diag, -v_diag, 0.0], dtype=np.float32),    # fwd-right
            # np.array([-v_diag, v_diag, 0.0], dtype=np.float32),    # back-left
            # np.array([-v_diag, -v_diag, 0.0], dtype=np.float32),   # back-right
            # 4 curving motions (forward + turn)
            # np.array([v, 0.0, w_quarter], dtype=np.float32),
            # np.array([v, 0.0, -w_quarter], dtype=np.float32),
            np.array([v, 0.0, w_half], dtype=np.float32),
            np.array([v, 0.0, -w_half], dtype=np.float32),
            np.array([v, 0.0, w], dtype=np.float32),
            np.array([v, 0.0, -w], dtype=np.float32),
            # # 4 spot turns
            # np.array([0.0, 0.0, w_quarter], dtype=np.float32),
            # np.array([0.0, 0.0, -w_quarter], dtype=np.float32),
            # np.array([0.0, 0.0, w_half], dtype=np.float32),
            # np.array([0.0, 0.0, -w_half], dtype=np.float32),
            # np.array([0.0, 0.0, w], dtype=np.float32),
            # np.array([0.0, 0.0, -w], dtype=np.float32),
            # 1 stop
            # np.array([0.0, 0.0, 0.0], dtype=np.float32),
        ]
        return actions

    def plan(self, current_pose: Pose, map_snapshot: dict, time_budget: float) -> Action:
        """在给定时间预算内运行 MCTS，返回最佳首动作。"""
        if time_budget <= 0.0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # 每次计划调用都刷新地图快照，保持与当前belief一致（不在树节点中存储地图）
        if not self._load_map_snapshot(map_snapshot):
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        root, iterations, elapsed = self._run_mcts_root(current_pose, time_budget)
        self._last_root = root
        self.last_iterations = iterations
        self.last_plan_time_sec = elapsed

        if not root.children:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # 选择访问次数最多的子节点（在噪声奖励下更稳健）
        best_child = max(root.children, key=lambda c: c.visit_count)
        if self.debug:
            self._debug_confirmation_diagnostics(current_pose, best_child.action)
            print(f"[MCTS] rollouts={iterations}")
        return np.array(best_child.action, dtype=np.float32)

    def _run_mcts_root(
        self, current_pose: Pose, time_budget: float
    ) -> Tuple[MCTSNode, int, float]:
        """Run MCTS loop and return (root, iterations, elapsed_sec)."""
        root = MCTSNode(current_pose, depth=0, parent=None, action=None, untried_actions=self.actions)

        start_t = time.time()
        iterations = 0

        while (time.time() - start_t) < time_budget:
            node = root

            # 1) Selection (UCT)
            while node.is_fully_expanded() and node.depth < self.search_depth and node.children:
                node = self._select_child(node)

            # 2) Expansion
            if node.depth < self.search_depth and node.untried_actions:
                idx = int(self.rng.integers(len(node.untried_actions)))
                action = node.untried_actions.pop(idx)
                next_pose = self._simulate_pose(node.pose, action)
                child = MCTSNode(
                    next_pose,
                    depth=node.depth + 1,
                    parent=node,
                    action=action,
                    untried_actions=self.actions,
                )
                node.children.append(child)
                node = child

            # 3) Simulation (rollout)
            reward = self.rollout(node)

            # 4) Backpropagation
            self._backpropagate(node, reward)
            iterations += 1

        return root, iterations, time.time() - start_t

    def rollout(self, node: MCTSNode) -> float:
        """从当前节点开始随机rollout，累计信息增益奖励。"""
        pose = np.array(node.pose, dtype=np.float32)
        total_reward = 0.0

        # rollout内的局部覆盖标记，避免重复计分
        visited_fls = np.zeros(self._entropy_substrate.shape, dtype=bool)
        visited_flc = np.zeros(self._entropy_coral.shape, dtype=bool)
        visited_dlc = np.array(self._confirmed_mask, dtype=bool)

        for _ in range(node.depth, self.search_depth):
            action = self.actions[int(self.rng.integers(len(self.actions)))]
            pose = self._simulate_pose(pose, action)
            total_reward += self._info_gain_at_pose(pose, visited_fls, visited_flc, visited_dlc)

        return float(total_reward)

    def _info_gain_at_pose(
        self,
        pose: Pose,
        visited_fls: np.ndarray,
        visited_flc: np.ndarray,
        visited_dlc: np.ndarray,
    ) -> float:
        """
        Information gain from one pose:
        - DLC: confirmation reward for high-confidence cells (P(coral) > threshold)
        - FLS/FLC: entropy reduction for unobserved cells (substrate/coral)
        """
        reward = 0.0

        # FLS（地质基底探索）：熵降低（当前权重为0，仅保留结构）
        fls_mask = self._sector_mask(pose, self._fls_grid_x, self._fls_grid_y, self.fls_range, self.fls_fov_rad)
        if fls_mask is not None:
            new_mask = fls_mask & (~visited_fls)
            if np.any(new_mask):
                reward += SUBSTRATE_ENTROPY_WEIGHT * float(np.sum(self._entropy_substrate[new_mask]))
            visited_fls |= fls_mask

        # FLC（珊瑚探索）：只在FLS已有信息处计入熵降低，避免盲目开采
        flc_mask = self._sector_mask(pose, self._flc_grid_x, self._flc_grid_y, self.flc_range, self.flc_fov_rad)
        if flc_mask is not None:
            if self._flc_informed_mask is None:
                new_mask = np.zeros_like(flc_mask, dtype=bool)
            else:
                new_mask = flc_mask & self._flc_informed_mask & (~visited_flc)
            if np.any(new_mask):
                reward += CORAL_ENTROPY_WEIGHT * float(np.sum(self._entropy_coral[new_mask]))
            visited_flc |= flc_mask

        # DLC（确认）：覆盖到高置信珊瑚单元时给定确认奖励
        dlc_mask = self._square_mask(pose, self._flc_grid_x, self._flc_grid_y, self.dlc_half)
        if dlc_mask is not None:
            new_mask = dlc_mask & (~visited_dlc)
            confirm_mask = new_mask & self._confirmation_map
            if np.any(confirm_mask):
                reward += CONFIRMATION_REWARD * float(np.sum(confirm_mask))
            elif CONFIRMATION_DIST_REWARD > 0.0:
                # 距离塑形：用尚未访问过的确认点计算最小距离（稠密奖励）
                target_mask = self._confirmation_map & (~visited_dlc)
                if np.any(target_mask):
                    dx = self._flc_grid_x[target_mask] - float(pose[0])
                    dy = self._flc_grid_y[target_mask] - float(pose[1])
                    min_dist = float(np.sqrt(np.min(dx * dx + dy * dy)))
                    reward += CONFIRMATION_DIST_REWARD / (1.0 + min_dist)
            visited_dlc |= dlc_mask

        return reward

    def _debug_confirmation_diagnostics(
        self, pose: Pose, best_action: Optional[Action]
    ) -> None:
        if self._confirmation_map is None or self._flc_grid_x is None or self._flc_grid_y is None:
            print("[MCTS] confirmation diagnostics skipped (map not initialized).")
            return

        total_confirm = int(np.sum(self._confirmation_map))
        threshold = self._confirmation_threshold
        step_max = self.max_velocity * self.dt

        if total_confirm == 0:
            print(
                f"[MCTS] confirmation cells=0 (threshold={threshold}, step_max={step_max:.2f} m)."
            )
            return

        mask = self._confirmation_map
        dx = self._flc_grid_x[mask] - float(pose[0])
        dy = self._flc_grid_y[mask] - float(pose[1])
        if dx.size == 0:
            min_dist = float("inf")
        else:
            min_dist = float(np.sqrt(np.min(dx * dx + dy * dy)))
        reachable_one_step = min_dist <= (step_max + self.dlc_half)

        action_hits = []
        for idx, action in enumerate(self.actions):
            next_pose = self._simulate_pose(pose, action)
            dlc_mask = self._square_mask(next_pose, self._flc_grid_x, self._flc_grid_y, self.dlc_half)
            hits = int(np.sum(self._confirmation_map & dlc_mask)) if dlc_mask is not None else 0
            action_hits.append((hits, idx))

        num_nonzero = sum(1 for hits, _idx in action_hits if hits > 0)
        top_hits = sorted(action_hits, key=lambda t: t[0], reverse=True)[:3]
        top_hits_str = ", ".join([f"idx={idx},hits={hits}" for hits, idx in top_hits])

        best_hits = None
        if best_action is not None:
            best_pose = self._simulate_pose(pose, best_action)
            dlc_mask = self._square_mask(best_pose, self._flc_grid_x, self._flc_grid_y, self.dlc_half)
            best_hits = int(np.sum(self._confirmation_map & dlc_mask)) if dlc_mask is not None else 0

        print(
            "[MCTS] confirmation cells="
            f"{total_confirm}, min_dist={min_dist:.2f} m, step_max={step_max:.2f} m, "
            f"reachable_one_step={reachable_one_step}, actions_with_hits={num_nonzero}, "
            f"best_action_hits={best_hits}, top_hits=[{top_hits_str}]"
        )

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        log_n = math.log(max(1, node.visit_count))
        best_score = -float("inf")
        best_child = None
        for child in node.children:
            if child.visit_count == 0:
                score = float("inf")
            else:
                exploit = child.total_value / child.visit_count
                explore = self.uct_c * math.sqrt(log_n / child.visit_count)
                score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child if best_child is not None else node

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        cur = node
        while cur is not None:
            cur.visit_count += 1
            cur.total_value += reward
            cur = cur.parent

    def _simulate_pose(self, pose: Pose, action: Action) -> Pose:
        x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
        vx, vy, omega = float(action[0]), float(action[1]), float(action[2])

        x += (vx * math.cos(theta) - vy * math.sin(theta)) * self.dt
        y += (vx * math.sin(theta) + vy * math.cos(theta)) * self.dt
        theta += omega * self.dt
        theta = math.atan2(math.sin(theta), math.cos(theta))
        return np.array([x, y, theta], dtype=np.float32)

    def _sector_mask(
        self,
        pose: Pose,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        max_dist: float,
        fov_rad: float,
    ) -> Optional[np.ndarray]:
        if grid_x.size == 0 or grid_y.size == 0:
            return None
        return _sector_mask_from_grid(pose, grid_x, grid_y, max_dist, fov_rad)

    def _square_mask(
        self,
        pose: Pose,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        half_side: float,
    ) -> Optional[np.ndarray]:
        if grid_x.size == 0 or grid_y.size == 0:
            return None
        return _square_mask_from_grid(pose, grid_x, grid_y, half_side)

    def _load_map_snapshot(self, map_snapshot: dict) -> bool:
        if map_snapshot is None:
            return False

        log_odds_s = map_snapshot.get("log_odds_s")
        prob_coral = map_snapshot.get("prob_coral")
        if log_odds_s is None or prob_coral is None:
            return False

        # 基础地图数据（每次计划周期都会刷新）
        self._log_odds_s = np.array(log_odds_s, dtype=np.float32)
        self._prob_coral = np.array(prob_coral, dtype=np.float32)
        visited_mask = map_snapshot.get("visited_mask")

        if visited_mask is None:
            visited_mask = np.zeros_like(self._prob_coral, dtype=bool)
        visited_mask = np.array(visited_mask, dtype=bool)
        # DLC已访问的单元格，用于rollout中避免重复计分
        self._confirmed_mask = visited_mask

        confirmation_threshold = CONFIRMATION_THRESHOLD
        self._confirmation_threshold = confirmation_threshold
        self._confirmation_map = (self._prob_coral > confirmation_threshold) & (~visited_mask)

        # 计算熵与已信息化区域
        self._entropy_substrate = self._entropy_from_prob(self._sigmoid(self._log_odds_s))
        self._entropy_coral = self._entropy_from_prob(self._prob_coral)
        entropy_max = math.log(2.0) # entropy_max = log(2) 对应完全不确定（p=0.5）的最大熵
        self._fls_informed_mask = self._entropy_substrate < (entropy_max - 1e-3)

        self._fls_origin = np.array(map_snapshot.get("fls_origin", [0.0, 0.0]), dtype=np.float32)
        self._flc_origin = np.array(map_snapshot.get("flc_origin", [0.0, 0.0]), dtype=np.float32)

        self._build_grids()
        self._cache_flc_informed_mask()
        return True

    def _build_grids(self) -> None:
        """构建FLS/FLC在世界坐标系下的网格。"""
        def build_grid(origin: np.ndarray, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
            rows, cols = shape
            if rows > 0 and cols > 0:
                xs = origin[0] + (np.arange(cols) + 0.5) * self.cell_size
                ys = origin[1] + (np.arange(rows) + 0.5) * self.cell_size
                return np.meshgrid(xs, ys)
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

        # FLS grid
        self._fls_grid_x, self._fls_grid_y = build_grid(self._fls_origin, self._log_odds_s.shape)
        # FLC grid
        self._flc_grid_x, self._flc_grid_y = build_grid(self._flc_origin, self._prob_coral.shape)

    def _cache_flc_informed_mask(self) -> None:
        """将FLS信息化区域投影到FLC网格，用于限制FLC熵奖励。"""
        if (
            self._fls_informed_mask is None
            or self._flc_grid_x is None
            or self._flc_grid_y is None
        ):
            self._flc_informed_mask = None
            return

        rows_c, cols_c = self._prob_coral.shape
        if rows_c == 0 or cols_c == 0:
            self._flc_informed_mask = None
            return

        rows_s, cols_s = self._fls_informed_mask.shape
        if rows_s == 0 or cols_s == 0:
            self._flc_informed_mask = None
            return

        fls_origin_x, fls_origin_y = float(self._fls_origin[0]), float(self._fls_origin[1])
        col_idx = np.floor((self._flc_grid_x - fls_origin_x) / self.cell_size).astype(np.int32)
        row_idx = np.floor((self._flc_grid_y - fls_origin_y) / self.cell_size).astype(np.int32)

        in_bounds = (
            (row_idx >= 0)
            & (row_idx < rows_s)
            & (col_idx >= 0)
            & (col_idx < cols_s)
        )
        flc_informed = np.zeros_like(self._flc_grid_x, dtype=bool)
        flc_informed[in_bounds] = self._fls_informed_mask[row_idx[in_bounds], col_idx[in_bounds]]
        self._flc_informed_mask = flc_informed

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _entropy_from_prob(p: np.ndarray) -> np.ndarray:
        eps = 1e-6
        p = np.clip(p, eps, 1.0 - eps)
        return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


# =========================
# Demo / visualization utilities
# =========================

def _smooth_field(field: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Simple box-blur to create spatial continuity in synthetic maps."""
    for _ in range(iterations):
        f = np.pad(field, 1, mode="edge")
        field = (
            f[:-2, :-2] + f[:-2, 1:-1] + f[:-2, 2:] +
            f[1:-1, :-2] + f[1:-1, 1:-1] + f[1:-1, 2:] +
            f[2:, :-2] + f[2:, 1:-1] + f[2:, 2:]
        ) / 9.0
    return field


def _build_synthetic_world(map_size_m: float, cell_size: float, seed: int = 3):
    """
    Build a synthetic world with continuous rock/sand and patchy coral.

    Belief initialization:
    - P(substrate)=0.5 everywhere (unknown)
    - P(c|s)=0.5 everywhere
    - P(coral)=P(c|s)*P(substrate)=0.25 everywhere initially
    """
    rows = int(map_size_m / cell_size)
    cols = int(map_size_m / cell_size)

    # Robot starts at map center
    center_x = 0.5 * cols * cell_size
    center_y = 0.5 * rows * cell_size
    robot_pose = np.array([center_x, center_y, 0.0], dtype=np.float32)

    # Belief priors
    p_substrate = np.full((rows, cols), 0.5, dtype=np.float32)
    p_c_given_s = np.full((rows, cols), 0.5, dtype=np.float32)

    rng = np.random.default_rng(seed)
    # Ground-truth substrate: smooth random field -> contiguous rock/sand
    base_field = rng.random((rows, cols))
    smooth_field = _smooth_field(base_field, iterations=4)
    thresh = np.quantile(smooth_field, 0.55)
    gt_substrate = (smooth_field > thresh).astype(np.int8)

    # Ground-truth coral: only where substrate=1, also smooth patches
    gt_coral = np.zeros((rows, cols), dtype=np.int8)
    substrate_one = gt_substrate == 1
    coral_field = _smooth_field(rng.random((rows, cols)), iterations=2)
    if np.any(substrate_one):
        coral_thresh = np.quantile(coral_field[substrate_one], 0.92)
        gt_coral[substrate_one & (coral_field > coral_thresh)] = 1

    prob_coral = p_substrate * p_c_given_s
    log_odds_s = np.log(np.clip(p_substrate, 1e-4, 1.0 - 1e-4) / np.clip(1.0 - p_substrate, 1e-4, 1.0 - 1e-4))

    snapshot = {
        "log_odds_s": log_odds_s.astype(np.float32),
        "prob_coral": prob_coral.astype(np.float32),
        "confirmation_map": np.zeros((rows, cols), dtype=bool),
        "visited_mask": np.zeros((rows, cols), dtype=bool),
        "fls_origin": np.array([0.0, 0.0], dtype=np.float32),
        "flc_origin": np.array([0.0, 0.0], dtype=np.float32),
    }
    return snapshot, robot_pose, gt_substrate, gt_coral



def _extract_best_path(root: MCTSNode, planner: MCTSPlanner, depth: int) -> np.ndarray:
    """Follow the most visited child; extend randomly if tree is shallow."""
    path = [root.pose.copy()]
    node = root
    for _ in range(depth):
        if node.children:
            node = max(node.children, key=lambda c: c.visit_count)
            path.append(node.pose.copy())
        else:
            action = planner.actions[int(planner.rng.integers(len(planner.actions)))]
            path.append(planner._simulate_pose(path[-1], action))
    return np.array(path, dtype=np.float32)


def _apply_sensor_updates(
    p_substrate: np.ndarray,
    p_c_given_s: np.ndarray,
    confirmed: np.ndarray,
    gt_substrate: np.ndarray,
    gt_coral: np.ndarray,
    pose: Pose,
    cell_size: float,
    fls_range: float,
    fls_fov_rad: float,
    flc_range: float,
    flc_fov_rad: float,
    dlc_half: float,
) -> None:
    """
    Apply idealized belief updates following the project model:
    - FLS: sets P(substrate)=0 or 1 (ground truth under sonar)
    - FLC: sets P(c|s)=0 or 1 (ground truth under camera)
    - DLC: confirms true coral under footprint
    """
    rows, cols = p_substrate.shape
    xs = (np.arange(cols) + 0.5) * cell_size
    ys = (np.arange(rows) + 0.5) * cell_size
    grid_x, grid_y = np.meshgrid(xs, ys)

    fls = _sector_mask_from_grid(pose, grid_x, grid_y, fls_range, fls_fov_rad)
    flc = _sector_mask_from_grid(pose, grid_x, grid_y, flc_range, flc_fov_rad)
    dlc = _square_mask_from_grid(pose, grid_x, grid_y, dlc_half)

    # Sonar: update substrate belief to ground truth
    p_substrate[fls] = gt_substrate[fls].astype(np.float32)

    # Camera: update coral-given-substrate to ground truth
    p_c_given_s[flc] = gt_coral[flc].astype(np.float32)

    # Confirmation: DLC confirms true coral
    confirmed[dlc & (gt_coral == 1)] = True


def _snapshot_from_belief(p_substrate: np.ndarray, p_c_given_s: np.ndarray) -> dict:
    """Build a planner snapshot from current belief arrays."""
    prob_coral = p_substrate * p_c_given_s
    log_odds_s = np.log(
        np.clip(p_substrate, 1e-4, 1.0 - 1e-4)
        / np.clip(1.0 - p_substrate, 1e-4, 1.0 - 1e-4)
    )
    return {
        "log_odds_s": log_odds_s.astype(np.float32),
        "prob_coral": prob_coral.astype(np.float32),
        "confirmation_map": np.zeros_like(prob_coral, dtype=bool),
        "visited_mask": np.zeros_like(prob_coral, dtype=bool),
        "fls_origin": np.array([0.0, 0.0], dtype=np.float32),
        "flc_origin": np.array([0.0, 0.0], dtype=np.float32),
    }


def _initialize_visuals(gt_substrate: np.ndarray, gt_coral: np.ndarray, cell_size: float):
    """Create figure, axes, images, and line handles for the receding-horizon demo."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    rows, cols = gt_substrate.shape
    extent = [0.0, cols * cell_size, 0.0, rows * cell_size]
    entropy_max = math.log(2.0)
    cmap = "viridis"

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 10.0))
    ax_ent_sub = axes[0, 0]
    ax_ent_cor = axes[0, 1]
    ax_pc = axes[0, 2]
    ax_ps = axes[1, 0]
    ax_conf = axes[1, 1]
    ax_gt = axes[1, 2]

    for ax in axes.flat:
        ax.set_xlim(0.0, cols * cell_size)
        ax.set_ylim(0.0, rows * cell_size)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    gt_map = np.zeros((rows, cols), dtype=np.int8)
    gt_map[gt_substrate == 1] = 1
    gt_map[gt_coral == 1] = 2
    gt_cmap = ListedColormap([(0.8, 0.7, 0.5), (0.4, 0.4, 0.4), (1.0, 0.2, 0.2)])
    gt_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], gt_cmap.N)

    # Initial belief maps
    p_substrate = np.full((rows, cols), 0.5, dtype=np.float32)
    p_c_given_s = np.full((rows, cols), 0.5, dtype=np.float32)
    p_coral = p_substrate * p_c_given_s
    ent_sub = MCTSPlanner._entropy_from_prob(p_substrate) / entropy_max
    ent_cor = MCTSPlanner._entropy_from_prob(p_coral) / entropy_max

    im_ent_sub = ax_ent_sub.imshow(ent_sub, origin="lower", extent=extent, cmap=cmap, vmin=0.0, vmax=1.0)
    im_ent_cor = ax_ent_cor.imshow(ent_cor, origin="lower", extent=extent, cmap=cmap, vmin=0.0, vmax=1.0)
    im_pc = ax_pc.imshow(p_coral, origin="lower", extent=extent, cmap=cmap, vmin=0.0, vmax=1.0)
    im_ps = ax_ps.imshow(p_substrate, origin="lower", extent=extent, cmap=cmap, vmin=0.0, vmax=1.0)
    im_conf = ax_conf.imshow(np.zeros_like(p_coral), origin="lower", extent=extent, cmap=cmap, vmin=0.0, vmax=1.0)
    im_gt = ax_gt.imshow(gt_map, origin="lower", extent=extent, cmap=gt_cmap, norm=gt_norm)

    ax_ent_sub.set_title("Entropy Substrate")
    ax_ent_cor.set_title("Entropy Coral (FLC)")
    ax_pc.set_title("P(coral)")
    ax_ps.set_title("P(substrate)")
    ax_conf.set_title("Confirmed coral (DLC)")
    ax_gt.set_title("GT map: sand/rock/coral")

    fig.colorbar(im_ent_sub, ax=ax_ent_sub, fraction=0.046, pad=0.04)
    fig.colorbar(im_ent_cor, ax=ax_ent_cor, fraction=0.046, pad=0.04)
    fig.colorbar(im_pc, ax=ax_pc, fraction=0.046, pad=0.04)
    fig.colorbar(im_ps, ax=ax_ps, fraction=0.046, pad=0.04)
    fig.colorbar(im_conf, ax=ax_conf, fraction=0.046, pad=0.04)
    cbar_gt = fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04, ticks=[0, 1, 2])
    cbar_gt.ax.set_yticklabels(["sand", "rock", "coral"])

    # Executed path (solid) and planned path (dashed)
    exec_lines = [
        ax_ent_sub.plot([], [], color="cyan", linewidth=2.0)[0],
        ax_ent_cor.plot([], [], color="cyan", linewidth=2.0)[0],
        ax_pc.plot([], [], color="cyan", linewidth=2.0)[0],
        ax_ps.plot([], [], color="cyan", linewidth=2.0)[0],
        ax_conf.plot([], [], color="cyan", linewidth=2.0)[0],
        ax_gt.plot([], [], color="cyan", linewidth=2.0)[0],
    ]
    plan_lines = [
        ax_ent_sub.plot([], [], color="yellow", linewidth=1.5, linestyle="--")[0],
        ax_ent_cor.plot([], [], color="yellow", linewidth=1.5, linestyle="--")[0],
        ax_pc.plot([], [], color="yellow", linewidth=1.5, linestyle="--")[0],
        ax_ps.plot([], [], color="yellow", linewidth=1.5, linestyle="--")[0],
        ax_conf.plot([], [], color="yellow", linewidth=1.5, linestyle="--")[0],
        ax_gt.plot([], [], color="yellow", linewidth=1.5, linestyle="--")[0],
    ]

    return {
        "fig": fig,
        "axes": (ax_ent_sub, ax_ent_cor, ax_pc, ax_ps, ax_conf, ax_gt),
        "images": (im_ent_sub, im_ent_cor, im_pc, im_ps, im_conf),
        "exec_lines": exec_lines,
        "plan_lines": plan_lines,
        "entropy_max": entropy_max,
        "extent": extent,
    }


def _receding_horizon_visual_loop(
    planner: MCTSPlanner,
    gt_substrate: np.ndarray,
    gt_coral: np.ndarray,
    start_pose: Pose,
    cell_size: float,
    num_plans: int = 10,
    time_budget: float = 1.0,
) -> np.ndarray:
    """
    For each iteration:
    1) Run MCTS and draw the full planned path.
    2) Execute one step, update belief, and redraw.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.transforms as transforms

    rows, cols = gt_substrate.shape
    map_w = cols * cell_size
    map_h = rows * cell_size

    vis = _initialize_visuals(gt_substrate, gt_coral, cell_size)
    ax_ent_sub, ax_ent_cor, ax_pc, ax_ps, ax_conf, _ax_gt = vis["axes"]
    im_ent_sub, im_ent_cor, im_pc, im_ps, im_conf = vis["images"]
    exec_lines = vis["exec_lines"]
    plan_lines = vis["plan_lines"]
    entropy_max = vis["entropy_max"]

    p_substrate = np.full((rows, cols), 0.5, dtype=np.float32)
    p_c_given_s = np.full((rows, cols), 0.5, dtype=np.float32)
    confirmed = np.zeros((rows, cols), dtype=bool)

    pose = np.array(start_pose, dtype=np.float32)
    executed = [pose.copy()]

    for step in range(num_plans):
        snapshot = _snapshot_from_belief(p_substrate, p_c_given_s)
        action = planner.plan(pose, snapshot, time_budget=time_budget)
        root = planner._last_root
        if root is None or not root.children:
            next_pose = planner._simulate_pose(pose, action)
            planned = np.array([pose.copy(), next_pose.copy()], dtype=np.float32)
        else:
            planned = _extract_best_path(root, planner, planner.search_depth)
        planned[:, 0] = np.clip(planned[:, 0], 0.0, map_w)
        planned[:, 1] = np.clip(planned[:, 1], 0.0, map_h)

        # Show planned path on current belief
        for line in plan_lines:
            line.set_data(planned[:, 0], planned[:, 1])
        ax_ent_sub.set_title(f"Entropy Substrate (plan {step})")
        plt.pause(0.05)

        # Execute one step
        if len(planned) > 1:
            pose = planned[1]
        executed.append(pose.copy())

        # Update belief with sensors
        _apply_sensor_updates(
            p_substrate,
            p_c_given_s,
            confirmed,
            gt_substrate,
            gt_coral,
            pose,
            cell_size,
            planner.fls_range,
            planner.fls_fov_rad,
            planner.flc_range,
            planner.flc_fov_rad,
            planner.dlc_half,
        )

        p_coral = p_substrate * p_c_given_s
        ent_sub = MCTSPlanner._entropy_from_prob(p_substrate) / entropy_max
        ent_cor = MCTSPlanner._entropy_from_prob(p_coral) / entropy_max

        im_ent_sub.set_data(ent_sub)
        im_ent_cor.set_data(ent_cor)
        im_pc.set_data(p_coral)
        im_ps.set_data(p_substrate)
        im_conf.set_data(confirmed.astype(np.float32))

        xs = np.array(executed)[:, 0]
        ys = np.array(executed)[:, 1]
        for line in exec_lines:
            line.set_data(xs, ys)

        # Clear previous sensor overlays
        for ax in (ax_ent_sub, ax_ent_cor, ax_conf):
            for p in list(ax.patches):
                p.remove()

        # FLS on substrate entropy
        fls_wedge = patches.Wedge(
            (pose[0], pose[1]),
            planner.fls_range,
            math.degrees(pose[2] - planner.fls_fov_rad / 2.0),
            math.degrees(pose[2] + planner.fls_fov_rad / 2.0),
            color="lime",
            alpha=0.2,
        )
        ax_ent_sub.add_patch(fls_wedge)

        # FLC on entropy coral
        flc_wedge = patches.Wedge(
            (pose[0], pose[1]),
            planner.flc_range,
            math.degrees(pose[2] - planner.flc_fov_rad / 2.0),
            math.degrees(pose[2] + planner.flc_fov_rad / 2.0),
            color="orange",
            alpha=0.3,
        )
        ax_ent_cor.add_patch(flc_wedge)

        # DLC on confirmed coral
        dlc = patches.Rectangle(
            (pose[0] - planner.dlc_half, pose[1] - planner.dlc_half),
            2 * planner.dlc_half,
            2 * planner.dlc_half,
            color="white",
            alpha=0.2,
        )
        dlc.set_transform(
            transforms.Affine2D().rotate_around(pose[0], pose[1], pose[2]) + ax_conf.transData
        )
        ax_conf.add_patch(dlc)

        plt.pause(0.05)

    plt.tight_layout()
    plt.show()
    return np.array(executed, dtype=np.float32)




if __name__ == "__main__":
    """Standalone demo: synthetic world + one MCTS plan + animated belief updates."""
    cell_size = 0.25
    map_size_m = 10.0

    planner = MCTSPlanner(
        cell_size=cell_size,
        fls_range=6.0,
        fls_fov_deg=90.0,
        flc_range=2.5,
        flc_fov_deg=60.0,
        dlc_footprint=1.0,
        max_velocity=1.0,
        max_angular_velocity=1.0,
        dt=0.5,
        search_depth=20,
        uct_c=1.2,
        seed=42,
    )

    snapshot, pose, gt_substrate, gt_coral = _build_synthetic_world(map_size_m, cell_size, seed=3)
    # Receding-horizon: plan -> visualize full plan -> take 1 step -> update belief -> repeat
    _receding_horizon_visual_loop(
        planner=planner,
        gt_substrate=gt_substrate,
        gt_coral=gt_coral,
        start_pose=pose,
        cell_size=cell_size,
        num_plans=200,
        time_budget=1,
    )
