import numpy as np
from typing import Iterable, Optional, Tuple
from param import (
    fls_true_positive_rate,
    fls_false_positive_rate,
    flc_true_positive_rate,
    flc_false_positive_rate,
)
from param import SUBSTRATE_ENTROPY_MAP_PADDING, CORAL_ENTROPY_MAP_PADDING
from param import MCTS_SUBSTRATE_ENTROPY_MAP_PADDING, MCTS_CORAL_ENTROPY_MAP_PADDING
from param import CONF_THRESHOLD

# Constants
L_MAX = 6.0    # P ~ 0.997
L_MIN = -6.0   # P ~ 0.002


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class BeliefMap:
    def __init__(self, map_size, cell_size, p_substrate_prior=0.5, p_coral_given_substrate_prior=0.5):
        """
        Two-layer belief representation:
        - log_odds_s: P(s_i = rock)
        - log_odds_c_given_s: P(c_i = coral | s_i = rock)
        Without a prior map, both layers start from an unbiased prior.
        """
        self.map_size = map_size
        self.cell_size = cell_size

        self.log_odds_s = np.full(map_size, logit(p_substrate_prior), dtype=np.float32)
        self.log_odds_c_given_s = np.full(map_size, logit(p_coral_given_substrate_prior), dtype=np.float32)  
        # we treat camera observations as log odds of P(c|s) because coral only exists on rock substrate (P(c|s=0)=0)


        self.visited_mask = np.zeros(map_size, dtype=bool)  # DLC footprint visited

        # Derived fields
        self.prob_substrate = np.zeros(map_size, dtype=np.float32)
        self.prob_c_given_s = np.zeros(map_size, dtype=np.float32)
        self.prob_coral = np.zeros(map_size, dtype=np.float32)  # P(c) = P(c|s)*P(s)
        self.reward_map = np.zeros(map_size, dtype=np.float32)       # Expected reward (not visited)
        self.confirmation_map = np.zeros(map_size, dtype=bool)       # Binary high-confidence view

        self._update_probabilities()

    def update_belief(self, observations):
        """
        Update the two-layer belief from FLS, FLC, and DLC observations.
        """
        # --- DLC (deterministic confirmation) ---
        dlc_obs = observations.get('dlc')
        if dlc_obs and dlc_obs['valid']:
            r1, r2, c1, c2 = dlc_obs['bbox']
            mask = dlc_obs['mask']
            detected_coral = dlc_obs['detected_coral']
            detected_empty = dlc_obs['detected_empty']

            local_log_c_given_s = self.log_odds_c_given_s[r1:r2, c1:c2]
            local_log_s = self.log_odds_s[r1:r2, c1:c2]
            local_visited = self.visited_mask[r1:r2, c1:c2]

            local_log_c_given_s[detected_coral] = L_MAX
            local_log_s[detected_coral] = L_MAX  # coral implies rock

            local_log_c_given_s[detected_empty] = L_MIN
            # empty does not force substrate to sand; keep substrate belief

            local_visited[mask] = True

            self.log_odds_c_given_s[r1:r2, c1:c2] = local_log_c_given_s
            self.log_odds_s[r1:r2, c1:c2] = local_log_s
            self.visited_mask[r1:r2, c1:c2] = local_visited

        # --- FLS (substrate scout) ---
        fls_obs = observations.get('fls')
        if fls_obs and fls_obs['valid']:
            r1, r2, c1, c2 = fls_obs['bbox']
            fov_mask = fls_obs['mask']
            d_norm = fls_obs['dists_norm']
            z_is_1 = fls_obs['detections']  # detection => rock-like

            if np.any(fov_mask):
                p_tp = fls_true_positive_rate(d_norm)
                p_fp = fls_false_positive_rate(d_norm)
                eps = 1e-6

                delta_l_detect = np.log((p_tp + eps) / (p_fp + eps))
                delta_l_miss = np.log((1 - p_tp + eps) / (1 - p_fp + eps))

                local_log = self.log_odds_s[r1:r2, c1:c2]
                update_val = np.zeros_like(local_log)

                mask_d = fov_mask & z_is_1
                mask_m = fov_mask & (~z_is_1)
                update_val[mask_d] = delta_l_detect[mask_d]
                update_val[mask_m] = delta_l_miss[mask_m]

                local_log += update_val
                np.clip(local_log, L_MIN, L_MAX, out=local_log)
                self.log_odds_s[r1:r2, c1:c2] = local_log

        # --- FLC (target scout) ---
        flc_obs = observations.get('flc')
        if flc_obs and flc_obs['valid']:
            r1, r2, c1, c2 = flc_obs['bbox']
            fov_mask = flc_obs['mask']
            d_norm = flc_obs['dists_norm']
            z_is_1 = flc_obs['detections']  # detection => coral-like

            if np.any(fov_mask):
                p_tp = flc_true_positive_rate(d_norm)
                p_fp = flc_false_positive_rate(d_norm)
                eps = 1e-6

                delta_l_detect = np.log((p_tp + eps) / (p_fp + eps))
                delta_l_miss = np.log((1 - p_tp + eps) / (1 - p_fp + eps))

                local_log_c_given_s = self.log_odds_c_given_s[r1:r2, c1:c2]
                update_val = np.zeros_like(local_log_c_given_s)

                mask_d = fov_mask & z_is_1
                mask_m = fov_mask & (~z_is_1)
                update_val[mask_d] = delta_l_detect[mask_d]
                update_val[mask_m] = delta_l_miss[mask_m]

                local_log_c_given_s += update_val
                np.clip(local_log_c_given_s, L_MIN, L_MAX, out=local_log_c_given_s)
                self.log_odds_c_given_s[r1:r2, c1:c2] = local_log_c_given_s

        # --- Sync Probability ---
        self._update_probabilities()

    def _update_probabilities(self):
        self.prob_substrate = 1.0 / (1.0 + np.exp(-self.log_odds_s))
        self.prob_c_given_s = 1.0 / (1.0 + np.exp(-self.log_odds_c_given_s))  # P(c|s)
        # Biological prior coupling: P(c) = P(c|s) * P(s)
        self.prob_coral = self.prob_c_given_s * self.prob_substrate

        self.reward_map = self.prob_coral * (~self.visited_mask)
        self.confirmation_map = self.reward_map > CONF_THRESHOLD

    def mask_reward_map(
        self,
        grids_of_interests: Optional[Iterable[Tuple[int, int]]],
        grid_cell_span: Tuple[int, int],
    ) -> None:
        """
        Restrict reward/confirmation to interested global grids.

        :param grids_of_interests: iterable of (grid_row, grid_col)
        :param grid_cell_span: (rows_per_grid, cols_per_grid) in belief cells
        """
        rows_per_grid, cols_per_grid = int(grid_cell_span[0]), int(grid_cell_span[1])
        if rows_per_grid <= 0 or cols_per_grid <= 0:
            raise ValueError("grid_cell_span must be positive.")

        # Always rebuild base reward from probabilities to avoid cumulative masking.
        base_reward = self.prob_coral * (~self.visited_mask)

        interest_mask = np.zeros(self.map_size, dtype=bool)
        if grids_of_interests is not None:
            map_rows, map_cols = self.map_size
            for grid_r, grid_c in grids_of_interests:
                grid_r = int(grid_r)
                grid_c = int(grid_c)
                if grid_r < 0 or grid_c < 0:
                    continue

                r1 = grid_r * rows_per_grid
                c1 = grid_c * cols_per_grid
                if r1 >= map_rows or c1 >= map_cols:
                    continue
                r2 = min((grid_r + 1) * rows_per_grid, map_rows)
                c2 = min((grid_c + 1) * cols_per_grid, map_cols)
                interest_mask[r1:r2, c1:c2] = True

        self.reward_map = base_reward * interest_mask
        self.confirmation_map = self.reward_map > CONF_THRESHOLD

    def get_snapshot(self):
        """Return a copy of the current belief state for planning."""
        return {
            'log_odds_s': self.log_odds_s.copy(),
            'log_odds_c_given_s': self.log_odds_c_given_s.copy(),
            'prob_substrate': self.prob_substrate.copy(),
            'prob_c_given_s': self.prob_c_given_s.copy(),
            'prob_coral': self.prob_coral.copy(),
            'reward_map': self.reward_map.copy(),
            'confirmation_map': self.confirmation_map.copy(),
            'visited_mask': self.visited_mask.copy(),
        }

    def get_sliced_snapshot(self, robot_pose, target_position):
        """
        Return local belief slices around the robot-target pair.

        - FLS (substrate) slice uses larger padding.
        - FLC/DLC (coral + confirmation) slice uses smaller padding.

        :param robot_pose: (x, y, theta)
        :param target_position: (x, y)
        :return:
            map_snapshot (dict),
            robot_pose_in_flc, target_position_in_flc,
            robot_pose_in_fls, target_position_in_fls
        """
        return self._get_sliced_snapshot_with_padding(
            robot_pose,
            target_position,
            SUBSTRATE_ENTROPY_MAP_PADDING,
            CORAL_ENTROPY_MAP_PADDING,
        )

    def get_sliced_snapshot_4mcts(self, robot_pose, target_position):
        """Return local belief slices for MCTS."""

        return self._get_sliced_snapshot_with_padding(
            robot_pose,
            target_position,
            MCTS_SUBSTRATE_ENTROPY_MAP_PADDING,
            MCTS_CORAL_ENTROPY_MAP_PADDING,
        )

    def _get_sliced_snapshot_with_padding(self, robot_pose, target_position, fls_padding, flc_padding):
        """Shared local slicing logic with configurable FLS/FLC padding."""
        robot_x, robot_y, robot_theta = robot_pose
        target_x, target_y = target_position

        r_robot, c_robot = self.world_to_cell((robot_x, robot_y))
        r_target, c_target = self.world_to_cell((target_x, target_y))

        FLS_padding = int(fls_padding)
        FLC_padding = int(flc_padding)
        DLC_padding = 0.6 * FLC_padding  # Conservative DLC footprint estimate (unused for now)

        

        FLS_r1_ = max(0, min(r_robot, r_target) - FLS_padding)
        FLS_r2_ = min(self.map_size[0], max(r_robot, r_target) + FLS_padding)
        FLS_c1_ = max(0, min(c_robot, c_target) - FLS_padding)
        FLS_c2_ = min(self.map_size[1], max(c_robot, c_target) + FLS_padding)

        r1 = max(0, min(r_robot, r_target) - FLC_padding)
        r2 = min(self.map_size[0], max(r_robot, r_target) + FLC_padding)
        c1 = max(0, min(c_robot, c_target) - FLC_padding)
        c2 = min(self.map_size[1], max(c_robot, c_target) + FLC_padding)
   
        # Local P(c) = P(c|s) * P(s), already maintained by _update_probabilities.
        local_prob_coral = self.prob_coral[r1:r2, c1:c2].copy()
        
        # Convert joint probability back to log-odds for planner initialization.
        local_log_c = logit(local_prob_coral) 

        map_snapshot = {
            'log_odds_s': self.log_odds_s[FLS_r1_:FLS_r2_, FLS_c1_:FLS_c2_].copy(), # substrate log-odds (FLS slice)
            'log_odds_c': local_log_c,   # P(c), used to initialize Lambda
            # 'log_odds_c_given_s': self.log_odds_c_given_s[r1:r2, c1:c2].copy(), # P(c|s) only
            
            'prob_substrate': self.prob_substrate[r1:r2, c1:c2].copy(),
            'prob_c_given_s': self.prob_c_given_s[r1:r2, c1:c2].copy(),
            'prob_coral': self.prob_coral[r1:r2, c1:c2].copy(),
            
            'reward_map': self.reward_map[r1:r2, c1:c2].copy(),
            'confirmation_map': self.confirmation_map[r1:r2, c1:c2].copy(),
            'visited_mask': self.visited_mask[r1:r2, c1:c2].copy(),
            'fls_origin': np.array([FLS_c1_ * self.cell_size, FLS_r1_ * self.cell_size], dtype=np.float32),
            'flc_origin': np.array([c1 * self.cell_size, r1 * self.cell_size], dtype=np.float32),
        }

        flc_origin_x = c1 * self.cell_size
        flc_origin_y = r1 * self.cell_size
        fls_origin_x = FLS_c1_ * self.cell_size
        fls_origin_y = FLS_r1_ * self.cell_size

        robot_pose_in_flc = np.array([
            robot_x - flc_origin_x,
            robot_y - flc_origin_y,
            robot_theta
        ])

        target_position_in_flc = np.array([
            target_x - flc_origin_x,
            target_y - flc_origin_y
        ])

        robot_pose_in_fls = np.array([
            robot_x - fls_origin_x,
            robot_y - fls_origin_y,
            robot_theta
        ])

        target_position_in_fls = np.array([
            target_x - fls_origin_x,
            target_y - fls_origin_y
        ])

        return map_snapshot, robot_pose_in_flc, target_position_in_flc, robot_pose_in_fls, target_position_in_fls
 
    def world_to_cell(self, position):
        """
        Convert world coordinates to map-cell indices.

        Convention: x -> col, y -> row
        """
        x, y = position
        c = int(x / self.cell_size)
        r = int(y / self.cell_size)
        c = np.clip(c, 0, self.map_size[1] - 1)
        r = np.clip(r, 0, self.map_size[0] - 1)
        return r, c
