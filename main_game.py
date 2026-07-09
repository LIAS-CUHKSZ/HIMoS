# ==========================================
# Main Loop with Split-Screen Dashboard
# ==========================================

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from param import SHOW_VIS
if not SHOW_VIS:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import time

import numpy as np
import math
import sys
from simulation.robot import OmnidirectionalRobot
from simulation.input_controller import InputController   

from param import (
    FLS_RANGE,
    FLS_FOV_DEG,
    FLC_RANGE,
    FLC_FOV_DEG,
    DLC_FOOTPRINT,
)
from param import GAME_FPS 
from param import CELL_SIZE
from param import KEY_VELOCITY, KEY_ANGULAR_VELOCITY
from param import PLANNING_TIMESTEP_SIZE, NEXEC, GRID_TIME_SEC
from param import DEBUG, RECORD_FRAMES, DEBUG_SHOW_INTEREST_GRIDS
from param import OUTPUT_DIR
from param import GlobalPlannerConfig

from simulation.coral_map import CoralMap
from planner.belief_map import BeliefMap
from planner.global_planner import GlobalPlanner
from planner.local_planner import UnifiedLocalPlanner


class Game:
    """AUV simulation with split-screen dashboard and planners."""
    def __init__(self):
        pygame.init()
        
        # Dashboard layout
        self.SIDEBAR_W = 320
        self.VIEW_W = 700
        self.TARGET_W = 1100
        self.HEIGHT = 720
        self.WIDTH = self.SIDEBAR_W + self.VIEW_W + self.TARGET_W
        
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("AUV Simulation Dashboard")
        
        # Dashboard panels
        self.rect_sidebar = pygame.Rect(0, 0, self.SIDEBAR_W, self.HEIGHT)
        self.rect_center  = pygame.Rect(self.SIDEBAR_W, 0, self.VIEW_W, self.HEIGHT)
        target_x = self.SIDEBAR_W + self.VIEW_W
        self.rect_target = pygame.Rect(target_x, 0, self.TARGET_W, self.HEIGHT)
        # View scales for the simulation and target panels
        self.view_scales = {
            "simulation": 15 * 2,
            "target": 15 * 3.0,
        }
        
        # Minimap shown in the sidebar
        self.minimap_size = (300, 300)
        self.minimap_rect = pygame.Rect(10, 10, *self.minimap_size)
        
        # Color palette
        self.COLOR_SAND = (194, 178, 128)    
        self.COLOR_ROCK = (80, 80, 80)    
        self.COLOR_CORAL_GT = (255, 105, 180)
        self.COLOR_CORAL_CONFIRMED = (255, 50, 50)
        self.COLOR_FLS_TP = (0, 180, 255)
        self.COLOR_FLS_FP = (120, 200, 255)
        self.COLOR_FLC_TP = (255, 140, 0)
        self.COLOR_FLC_FP = (255, 215, 120)

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.title_font = pygame.font.SysFont("Arial", 24, bold=True)

        # Environment state
        self.map_data = CoralMap(map_file=MAP_DIR, map_size=MAP_SIZE, cell_size=CELL_SIZE) 
        self.last_obs = None
        
        # Robot and controller
        self.robot = OmnidirectionalRobot(x=START_X, y=START_Y, theta=START_THETA)
        self.controller = InputController(key_velocity=KEY_VELOCITY, key_angular_velocity=KEY_ANGULAR_VELOCITY)
        
        # Belief Map (no prior map; initialize neutral beliefs)
        self.map_belief = BeliefMap(
            map_size=MAP_SIZE,
            cell_size=CELL_SIZE,
        )

        # Global Planner
        self.global_planner = self._init_global_planner()

        self.local_planner = UnifiedLocalPlanner()
        self.next_target = None
        self.traj_world = []
        self._last_traj_pose = None
        self._traj_min_step = 0.05  # meters
        self._record_pose()
        # --- Experiment logging state ---
        self._exp_start_time = time.time()
        self._results_saved = False
        self._exp_out_dir = None
        self._exp_timestamp = None
        self._time_log_f = None
        self._time_log_path = None
        self._frame_dir = None
        self._full_observed_frame_dir = None
        self._global_planner_frame_dir = None
        self._frame_index = 0
        self._frame_stride = 1
        self._sim_time_elapsed = 0.0
        self._last_progress_pct = None
        self._global_frame_save_warned = False

        # --- Runtime/planning state ---
        self._replan_elapsed_sec_accum = 0.0
        self._printed_status_legend = False

        # Sidebar runtime status
        self.sidebar_remaining_time_sec = None
        self.sidebar_global_budget = None
        self.sidebar_robot_pose = None
        self.sidebar_target = None
        self.sidebar_local_budget_sec = None
        self.sidebar_local_status = None
        self.sidebar_local_solve_time_sec = None
        self.sidebar_local_steps = None
        self.sidebar_replan_info = None
        self.sidebar_exec_index = None
        self.sidebar_exec_target = None
        self.debug_show_interest_grids = bool(DEBUG_SHOW_INTEREST_GRIDS)
        self._last_interest_grids = set()

    # =========================
    # Experiment Logging
    # =========================
    def _init_experiment_logging(self):
        """Create output folder and initialize CSV/video frame recording."""
        if self._exp_out_dir is not None:
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        budget_num = int(TOTAL_TIME_BUDGET_SECS)
        out_dir = os.path.join(
            OUTPUT_DIR,
            f"exp_map{map_index}_start{start_pose_index}_budget{budget_num}_{timestamp}",
        )
        os.makedirs(out_dir, exist_ok=True)

        self._exp_out_dir = out_dir
        self._exp_timestamp = timestamp
        print(f"[Experiment] Data will be saved at {out_dir}")
        self._time_log_path = os.path.join(out_dir, "coral_timeseries.csv")
        self._time_log_f = open(self._time_log_path, "w", encoding="utf-8")
        self._time_log_f.write(
            "sim_time_sec,confirmed_corals,total_corals,confirm_ratio,robot_x,robot_y,robot_theta\n"
        )
        self._time_log_f.flush()

        if RECORD_FRAMES:
            self._frame_dir = os.path.join(out_dir, "frames")
            os.makedirs(self._frame_dir, exist_ok=True)
            self._full_observed_frame_dir = os.path.join(out_dir, "frames_full_observed")
            os.makedirs(self._full_observed_frame_dir, exist_ok=True)
            self._global_planner_frame_dir = os.path.join(out_dir, "frames_global_planner")
            os.makedirs(self._global_planner_frame_dir, exist_ok=True)
        else:
            self._frame_dir = None
            self._full_observed_frame_dir = None
            self._global_planner_frame_dir = None

    def _log_coral_status(self):
        """Append one timestep of coral stats to CSV."""
        if self._time_log_f is None:
            return
        total_corals = int(getattr(self.map_data, "total_corals", 0))
        confirmed = int(getattr(self.map_data, "confirmed_count", 0))
        confirm_ratio = (confirmed / total_corals) if total_corals > 0 else 0.0
        self._time_log_f.write(
            f"{self._sim_time_elapsed:.3f},{confirmed},{total_corals},"
            f"{confirm_ratio:.6f},{self.robot.x:.3f},{self.robot.y:.3f},{self.robot.theta:.6f}\n"
        )
        self._time_log_f.flush()

    def _capture_frame(self):
        """Save a frame of the current screen if frame recording is enabled."""
        if self._frame_dir is None:
            return
        if self._frame_stride <= 0:
            return
        if (self._frame_index % self._frame_stride) != 0:
            self._frame_index += 1
            return
        frame_idx = self._frame_index
        frame_path = os.path.join(self._frame_dir, f"frame_{frame_idx:06d}.png")
        pygame.image.save(self.screen, frame_path)
        if self._full_observed_frame_dir is not None:
            full_obs_surf = self._render_full_observed_map(scale=3, draw_overlay=True)
            full_obs_path = os.path.join(
                self._full_observed_frame_dir, f"full_observed_{frame_idx:06d}.png"
            )
            pygame.image.save(full_obs_surf, full_obs_path)

        # Save global planner matplotlib figure alongside pygame frames when available.
        gp_fig = getattr(self.global_planner, "fig", None)
        if self._global_planner_frame_dir is not None and gp_fig is not None:
            gp_frame_path = os.path.join(
                self._global_planner_frame_dir, f"global_planner_{frame_idx:06d}.png"
            )
            try:
                gp_fig.savefig(gp_frame_path, dpi=140, bbox_inches="tight")
            except Exception as exc:
                if not self._global_frame_save_warned:
                    print(f"[Frame] Failed to save global planner frame: {exc}")
                    self._global_frame_save_warned = True

        self._frame_index += 1

    def _print_terminal_progress(self, remaining_time_sec: float):
        """Print one-line time-budget progress in terminal."""
        total_budget = max(1e-6, float(TOTAL_TIME_BUDGET_SECS))
        remaining = max(0.0, min(float(remaining_time_sec), total_budget))
        progress = 1.0 - (remaining / total_budget)
        pct = int(progress * 100)
        if self._last_progress_pct == pct:
            return
        self._last_progress_pct = pct

        bar_len = 30
        fill_len = int(bar_len * progress)
        bar = "#" * fill_len + "-" * (bar_len - fill_len)
        elapsed = total_budget - remaining
        print(
            f"\r[Progress] |{bar}| {progress * 100:6.2f}% ({elapsed:.1f}/{total_budget:.1f}s)",
            end="",
            flush=True,
        )

    # =========================
    # Initialization Helpers
    # =========================
    def _init_global_planner(self):
        """Instantiate the global planner with proper grid and start pose."""
        grid_interval_m = getattr(GLOBAL_PLANNER_CONFIG, "grid_interval_m", 2.5) # 2.5m
        map_size_m = (self.map_data.width_meters, self.map_data.height_meters) # (50m,50m)
        grid_res = int(round(map_size_m[0] / grid_interval_m))

        start_r = int(self.robot.y / grid_interval_m) if grid_interval_m > 0 else 0
        start_c = int(self.robot.x / grid_interval_m) if grid_interval_m > 0 else 0
        start_r = max(0, min(grid_res - 1, start_r))
        start_c = max(0, min(grid_res - 1, start_c))

        planner = GlobalPlanner(
            budget=getattr(GLOBAL_PLANNER_CONFIG, "path_budget", 150.0),
            beta=getattr(GLOBAL_PLANNER_CONFIG, "ucb_beta", 0.2),
            grid_res=grid_res,
            visualize=(SHOW_VIS or RECORD_FRAMES),
            random_seed=42,
            # GlobalPlanner expects start as (col, row) to match (x, y).
            start=(start_c, start_r),
            grid_size=grid_interval_m,
            node_var_threshold=getattr(GLOBAL_PLANNER_CONFIG, "node_var_threshold", 0.2),
            use_real=True,
            map_path=MAP_DIR,
            dist_decay_factor=getattr(GLOBAL_PLANNER_CONFIG, "dist_decay_factor", 0.0),
            gp_length_scale_m=getattr(GLOBAL_PLANNER_CONFIG, "gp_length_scale_m", 1.0),
            gp_signal_var=getattr(GLOBAL_PLANNER_CONFIG, "gp_signal_var", 0.16),
        )

        return planner

    def _record_pose(self):
        """Track robot trajectory (downsampled by distance)."""
        pose = (self.robot.x, self.robot.y, self.robot.theta)
        if self._last_traj_pose is None:
            self.traj_world.append(pose)
            self._last_traj_pose = pose
            return
        dx = pose[0] - self._last_traj_pose[0]
        dy = pose[1] - self._last_traj_pose[1]
        if dx * dx + dy * dy >= self._traj_min_step * self._traj_min_step:
            self.traj_world.append(pose)
            self._last_traj_pose = pose

    def _get_global_interest_grids(self):
        grids_of_interests = set()
        target_grids = getattr(self.global_planner, "target_grids", None)
        visited = getattr(self.global_planner, "visited", None)
        if target_grids:
            grids_of_interests.update(target_grids)
        if visited:
            grids_of_interests.update(visited)
        return grids_of_interests

    def _mask_reward_map_with_global_interests(self):
        """
        Keep reward/confirmation only inside global planner interested grids:
        target_grids U visited.
        """
        grid_res = int(getattr(self.global_planner, "_grid_res", 0))
        if grid_res <= 0:
            self._last_interest_grids = set()
            return

        map_rows, map_cols = self.map_belief.map_size
        if map_rows % grid_res != 0 or map_cols % grid_res != 0:
            self._last_interest_grids = set()
            return

        rows_per_grid = map_rows // grid_res
        cols_per_grid = map_cols // grid_res

        grids_of_interests = self._get_global_interest_grids()
        self._last_interest_grids = set(grids_of_interests)

        self.map_belief.mask_reward_map(
            grids_of_interests=grids_of_interests,
            grid_cell_span=(rows_per_grid, cols_per_grid),
        )

    def _toggle_interest_grid_debug_overlay(self):
        self.debug_show_interest_grids = not self.debug_show_interest_grids
        state = "ON" if self.debug_show_interest_grids else "OFF"
        print(f"[Debug] Interest-grid overlay: {state} (press G to toggle)")

    # =========================
    # Render-to-File Helpers
    # =========================
    def _render_full_observed_map(self, scale=1, draw_overlay=False):
        rows, cols = self.map_data.rows, self.map_data.cols
        img = np.zeros((cols, rows, 3), dtype=np.uint8)

        for r in range(rows):
            for c in range(cols):
                status = self.map_data.status_mask[r, c]
                val = self.map_data.map[r, c]
                if status == 2:  # Confirmed
                    if val == 0:
                        color = self.COLOR_SAND
                    elif val == 1:
                        color = self.COLOR_ROCK
                    else:
                        color = self.COLOR_CORAL_CONFIRMED
                else:
                    if val == 2:
                        color = (80, 20, 20)    # unexplored coral (darker red)
                    elif val == 0:
                        color = (70, 60, 35)    # unexplored sand (darker sand)
                    else:
                        color = (20, 20, 20)    # unexplored rock (darker gray)

                img[c, r] = color

        if scale > 1:
            img = np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8))
        surf = pygame.surfarray.make_surface(img)

        if draw_overlay:
            max_w = img.shape[0] - 1
            max_h = img.shape[1] - 1

            # Draw trajectory on the full observed map.
            if len(self.traj_world) >= 2:
                pts = []
                for wx, wy, _ in self.traj_world:
                    px = int(wx / self.map_data.cell_size * scale)
                    py = int(wy / self.map_data.cell_size * scale)
                    px = max(0, min(max_w, px))
                    py = max(0, min(max_h, py))
                    pts.append((px, py))
                if len(pts) >= 2:
                    pygame.draw.lines(surf, (0, 220, 255), False, pts, 2)

            self._draw_robot_overlay_full_map(surf, scale)

        return surf

    def _draw_robot_overlay_full_map(self, surf, scale):
        """Draw robot body and sensor footprints directly on a full-map surface."""
        sx = int(self.robot.x / self.map_data.cell_size * scale)
        sy = int(self.robot.y / self.map_data.cell_size * scale)
        m_to_px = float(scale) / float(self.map_data.cell_size)

        # FLS Arc (long-range substrate scout)
        fls_r = FLS_RANGE * m_to_px
        fls_half = math.radians(FLS_FOV_DEG) / 2
        fls_start = self.robot.theta - fls_half
        fls_end = self.robot.theta + fls_half
        fls_end_pos1 = (sx + fls_r * math.cos(fls_start), sy + fls_r * math.sin(fls_start))
        fls_end_pos2 = (sx + fls_r * math.cos(fls_end), sy + fls_r * math.sin(fls_end))
        pygame.draw.line(surf, (0, 200, 0), (sx, sy), fls_end_pos1, 1)
        pygame.draw.line(surf, (0, 200, 0), (sx, sy), fls_end_pos2, 1)
        fls_bbox = pygame.Rect(sx - fls_r, sy - fls_r, fls_r * 2, fls_r * 2)
        pygame.draw.arc(surf, (0, 200, 0), fls_bbox, -fls_end, -fls_start, 2)

        # FLC Arc (mid-range target scout)
        flc_r = FLC_RANGE * m_to_px
        flc_half = math.radians(FLC_FOV_DEG) / 2
        flc_start = self.robot.theta - flc_half
        flc_end = self.robot.theta + flc_half
        flc_end_pos1 = (sx + flc_r * math.cos(flc_start), sy + flc_r * math.sin(flc_start))
        flc_end_pos2 = (sx + flc_r * math.cos(flc_end), sy + flc_r * math.sin(flc_end))
        pygame.draw.line(surf, (255, 140, 0), (sx, sy), flc_end_pos1, 1)
        pygame.draw.line(surf, (255, 140, 0), (sx, sy), flc_end_pos2, 1)
        flc_bbox = pygame.Rect(sx - flc_r, sy - flc_r, flc_r * 2, flc_r * 2)
        pygame.draw.arc(surf, (255, 140, 0), flc_bbox, -flc_end, -flc_start, 2)

        # DLC footprint
        dlc_px = int(DLC_FOOTPRINT * m_to_px)
        footprint_surf = pygame.Surface((dlc_px, dlc_px), pygame.SRCALPHA)
        pygame.draw.rect(footprint_surf, (255, 0, 0, 120), footprint_surf.get_rect(), 2)
        rot_surf = pygame.transform.rotate(footprint_surf, math.degrees(-self.robot.theta))
        rot_rect = rot_surf.get_rect(center=(sx, sy))
        surf.blit(rot_surf, rot_rect)

        # Robot body
        r_rad = 6
        p1 = (sx + r_rad * math.cos(self.robot.theta), sy + r_rad * math.sin(self.robot.theta))
        p2 = (sx + r_rad * math.cos(self.robot.theta + 2.5), sy + r_rad * math.sin(self.robot.theta + 2.5))
        p3 = (sx + r_rad * math.cos(self.robot.theta - 2.5), sy + r_rad * math.sin(self.robot.theta - 2.5))
        pygame.draw.polygon(surf, (0, 255, 255), [p1, p2, p3])

    def _render_full_gt_with_traj(self, scale=1):
        gt_map = self.map_data.map.T  # Shape: (cols, rows)
        map_rgb = np.zeros((*gt_map.shape, 3), dtype=np.uint8)
        map_rgb[gt_map == 0] = self.COLOR_SAND
        map_rgb[gt_map == 1] = self.COLOR_ROCK
        map_rgb[gt_map == 2] = self.COLOR_CORAL_GT
        if scale > 1:
            map_rgb = np.kron(map_rgb, np.ones((scale, scale, 1), dtype=np.uint8))
        surf = pygame.surfarray.make_surface(map_rgb)
        max_w = map_rgb.shape[0] - 1
        max_h = map_rgb.shape[1] - 1

        # Draw trajectory
        if len(self.traj_world) >= 2:
            pts = []
            for wx, wy, _ in self.traj_world:
                px = int(wx / self.map_data.cell_size * scale)
                py = int(wy / self.map_data.cell_size * scale)
                px = max(0, min(max_w, px))
                py = max(0, min(max_h, py))
                pts.append((px, py))
            if len(pts) >= 2:
                pygame.draw.lines(surf, (0, 220, 255), False, pts, 2)

        return surf

    def _save_experiment_results(self):
        """Write summary files, maps, and trajectory after simulation ends."""
        if self._results_saved:
            return
        self._results_saved = True

        if self._exp_out_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            budget_num = int(TOTAL_TIME_BUDGET_SECS)
            out_dir = os.path.join(
                OUTPUT_DIR,
                f"exp_map{map_index}_start{start_pose_index}_budget{budget_num}_{timestamp}",
            )
            os.makedirs(out_dir, exist_ok=True)
            self._exp_out_dir = out_dir
            self._exp_timestamp = timestamp
        out_dir = self._exp_out_dir

        # 1) Full GT map with trajectory
        export_scale = 3
        gt_surf = self._render_full_gt_with_traj(scale=export_scale)
        pygame.image.save(gt_surf, os.path.join(out_dir, "gt_with_trajectory.png"))

        # 2) Full observed map (raw, no sensor footprint)
        sim_surf = self._render_full_observed_map(scale=export_scale)
        pygame.image.save(sim_surf, os.path.join(out_dir, "simulation_observed.png"))

        # 3) Text summary
        elapsed = time.time() - self._exp_start_time
        total_corals = int(getattr(self.map_data, "total_corals", 0))
        confirmed = int(getattr(self.map_data, "confirmed_count", 0))
        confirm_ratio = (confirmed / total_corals) if total_corals > 0 else 0.0

        path_len = 0.0
        for i in range(1, len(self.traj_world)):
            dx = self.traj_world[i][0] - self.traj_world[i-1][0]
            dy = self.traj_world[i][1] - self.traj_world[i-1][1]
            path_len += math.hypot(dx, dy)

        # 3) Trajectory file
        traj_arr = np.array(self.traj_world, dtype=np.float32)
        np.save(os.path.join(out_dir, "trajectory.npy"), traj_arr)
        np.savetxt(os.path.join(out_dir, "trajectory.txt"), traj_arr, fmt="%.4f",
                   header="x(m) y(m) theta(rad)")

        # 4) Text summary
        summary_path = os.path.join(out_dir, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            timestamp = self._exp_timestamp or time.strftime("%Y%m%d_%H%M%S")
            f.write(f"timestamp: {timestamp}\n")
            f.write(f"map_path: {MAP_DIR}\n")
            f.write(f"elapsed_sec: {elapsed:.2f}\n")
            f.write(f"robot_final_x: {self.robot.x:.3f}\n")
            f.write(f"robot_final_y: {self.robot.y:.3f}\n")
            f.write(f"robot_final_theta_deg: {math.degrees(self.robot.theta):.2f}\n")
            f.write(f"total_corals: {total_corals}\n")
            f.write(f"confirmed_corals: {confirmed}\n")
            f.write(f"confirm_ratio: {confirm_ratio:.3f}\n")
            f.write(f"path_length_m: {path_len:.2f}\n")
            f.write(f"traj_points: {len(self.traj_world)}\n")
            if hasattr(self.global_planner, "remaining_budget"):
                f.write(f"global_remaining_budget: {self.global_planner.remaining_budget:.2f}\n")
            if self._time_log_path is not None:
                f.write(f"coral_timeseries: {self._time_log_path}\n")
            if self._frame_dir is not None:
                f.write(f"frames_dir: {self._frame_dir}\n")
                f.write(f"frame_count: {self._frame_index}\n")

        if self._time_log_f is not None:
            self._time_log_f.close()
            self._time_log_f = None

        print(f"[Experiment] Results saved to {out_dir}")

    # =========================
    # Run/Shutdown Flow
    # =========================
    def _wait_for_exit(self):
        """Block UI until user closes the window or presses Q/ESC."""
        if not SHOW_VIS:
            return
        waiting = True
        msg = "Simulation ended. Press Q / ESC or close the window."
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        waiting = False
            self.draw()
            label = self.title_font.render(msg, True, (255, 255, 255))
            bg_rect = pygame.Rect(10, self.HEIGHT - 40, label.get_width() + 20, 30)
            pygame.draw.rect(self.screen, (0, 0, 0), bg_rect)
            self.screen.blit(label, (15, self.HEIGHT - 35))
            pygame.display.flip()
            self.clock.tick(10)

    def _finalize_run(self, user_requested_exit=False):
        """Persist results and close pygame cleanly."""
        if self._last_progress_pct is not None:
            print()
        self._save_experiment_results()
        if not user_requested_exit:
            self._wait_for_exit()
        pygame.quit()
        sys.exit()


    def run_manual(self):
        """Manual control mode."""
        self._init_experiment_logging()
        running = True
        user_requested_exit = False

        total_budget = getattr(GLOBAL_PLANNER_CONFIG, "path_budget", 0.0)
        t_elapsed = 0.0
        need_replan = False

        while running:
            dt = self.clock.tick(GAME_FPS) / 1000.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    user_requested_exit = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                    self._toggle_interest_grid_debug_overlay()
            
            # Update
            ctrl = self.controller.get_control_input()

            obs = self.map_data.get_observations(self.robot.x, self.robot.y, self.robot.theta)
            self.last_obs = obs
            self.map_belief.update_belief(obs)


            self.robot.update(dt, ctrl)
            self._record_pose()
            t_elapsed += dt

            # Render
            self.draw()
            pygame.display.flip()
            self._capture_frame()
            
        self._finalize_run(user_requested_exit=user_requested_exit)

    def run_planner_HIMoS(self):
        """
        Autonomous planning simulation loop using the real-case global planner.
        """
        self._init_experiment_logging()
        running = True
        user_requested_exit = False

        if not self._printed_status_legend:
            print("[Sidebar Legend] LocalPlanner status meanings:")
            print("  Optimal: solver converged successfully.")
            print("  MaxIter: solver hit max iterations, using sub-optimal result.")
            print("  Cruise: far-target cruise mode (simple kinematics).")
            print("  FallbackCruise: optimization failed, fallback to cruise controls.")
            print("  Failed: solver failed (no valid control).")
            self._printed_status_legend = True

        SIM_DT = PLANNING_TIMESTEP_SIZE  # Δt in Algorithm 1
        control_queue = []
        index = 0

        self.next_target = None
        replan_index = 0
        exec_index = 0
        last_replan_budget_sec = None

        remaining_time_sec = float(TOTAL_TIME_BUDGET_SECS)

        # --- First Perception Before start: GetMeasurements + UpdateBelief ---
        obs = self.map_data.get_observations(self.robot.x, self.robot.y, self.robot.theta)
        self.last_obs = obs
        self.map_belief.update_belief(obs)
        self._mask_reward_map_with_global_interests()
        
        while running and remaining_time_sec > 0.0:
            # self.clock.tick(GAME_FPS)
            self.clock.tick(GAME_FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    user_requested_exit = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                    self._toggle_interest_grid_debug_overlay()
            if not running:
                break

            # --- Global Strategic Layer (Real Case) ---
            self.global_planner.update_belief_real(self.map_belief.prob_substrate)
            
            if self.next_target is None:
                need_replan = True
            else:
                need_replan = self.global_planner.check_reached_target(
                    self.robot.x, self.robot.y
                )

            self.global_planner.sync_budget_from_time(remaining_time_sec, GRID_TIME_SEC)
            # print(f"[RunNew] Remaining time budget: {remaining_time_sec:.2f}, "
            #       f"Global planner Budget: {self.global_planner.remaining_budget:.2f}, "
            #       f"robot current location: ({self.robot.x:.2f}, {self.robot.y:.2f})")
            self.sidebar_remaining_time_sec = remaining_time_sec
            self._print_terminal_progress(remaining_time_sec)
            self.sidebar_global_budget = self.global_planner.remaining_budget
            self.sidebar_robot_pose = (self.robot.x, self.robot.y, self.robot.theta)

            if need_replan:
                replan_index += 1
                if self._replan_elapsed_sec_accum > 0.0:
                    actual_leg_time_sec = self._replan_elapsed_sec_accum
                    # print(
                    #     "[RunNew] Replan #%d: previous path actual time=%.2fs | planned_budget=%.2fs\n\n"
                    #     % (replan_index, actual_leg_time_sec, last_replan_budget_sec)
                    # )
                    self.sidebar_replan_info = (replan_index, actual_leg_time_sec, last_replan_budget_sec)
                target_m, _target_cell, local_budget_grid = self.global_planner.plan()

                if target_m is None:
                    print("[RunNew] Global Planner found NO PATH.")
                    break
                else:
                    self.next_target = target_m
                    local_budget_time_sec = GRID_TIME_SEC * local_budget_grid
                    local_budget_time_sec = min(local_budget_time_sec, remaining_time_sec)
                    # print(f"[RunNew] New target assigned: {target_m}, with local budget: {local_budget_time_sec:.2f}s")
                    self._replan_elapsed_sec_accum = 0.0
                    last_replan_budget_sec = local_budget_time_sec
                    self.sidebar_target = target_m
                    self.sidebar_local_budget_sec = local_budget_time_sec

                control_queue.clear()

            if self.next_target is None:
                print("[RunNew] No target available; stopping.")
                break

            # --- Local Planning Layer ---
            self._mask_reward_map_with_global_interests()
            robot_pose = np.array([self.robot.x, self.robot.y, self.robot.theta])
            target_position = np.array(self.next_target)

            (map_snapshot,
             robot_pose_in_flc, target_position_in_flc,
             robot_pose_in_fls, target_position_in_fls) = self.map_belief.get_sliced_snapshot(
                robot_pose, target_position
            )
            logodds_s_snapshot = map_snapshot['log_odds_s']
            logodds_c_snapshot = map_snapshot['log_odds_c']
            prob_snapshot = map_snapshot['prob_coral']
            reward_snapshot = map_snapshot['reward_map']
            confirmation_snapshot = map_snapshot['confirmation_map']

            # if DEBUG:
            #     path_dir = f'./planner/local_planner_map/map_{index}'
            #     if not os.path.exists(path_dir):
            #         os.makedirs(path_dir)
            #     np.save(f'{path_dir}/logodds_snapshot.npy', logodds_c_snapshot)
            #     np.save(f'{path_dir}/logodds_s_snapshot.npy', logodds_s_snapshot)
            #     np.save(f'{path_dir}/prob_snapshot.npy', prob_snapshot)
            #     np.save(f'{path_dir}/reward_snapshot.npy', reward_snapshot)
            #     np.save(f'{path_dir}/confirmation_snapshot.npy', confirmation_snapshot)
            #     np.savetxt(f'{path_dir}/robot_pose_in_slice.txt', robot_pose_in_flc)
            #     np.savetxt(f'{path_dir}/target_position_in_slice.txt', target_position_in_flc)
            #     np.savetxt(f'{path_dir}/fls_origin.txt', map_snapshot['fls_origin'])
            #     np.savetxt(f'{path_dir}/flc_origin.txt', map_snapshot['flc_origin'])
            #     print(f"Saved local planner map slice at {path_dir}/.")
            #     index += 1

            res = self.local_planner.plan(
                time_budget=local_budget_time_sec,
                current_pose=robot_pose_in_flc,
                target=target_position_in_flc,
                log_odds_s=logodds_s_snapshot,
                log_odds_c=logodds_c_snapshot,
                confirmation_map=confirmation_snapshot,
                fls_origin=map_snapshot['fls_origin'],
                flc_origin=map_snapshot['flc_origin'],
            )

            if res['success']:
                control_queue = res['control_sequence'].tolist()
                exec_index += NEXEC
                # print(f"Executing {exec_index}-th local plan towards target {self.next_target}.")
                self.sidebar_exec_index = exec_index
                self.sidebar_exec_target = self.next_target
                self.sidebar_local_status = res.get('status', 'ok')
                self.sidebar_local_solve_time_sec = res.get('solve_time_sec')
                self.sidebar_local_steps = res.get('steps', len(res.get('control_sequence', [])))
            else:
                # print("[RunNew] Local Plan Failed.")
                control_queue = [np.array([0.0, 0.0, 0.0])]
                self.sidebar_local_status = "failed"
                self.sidebar_local_solve_time_sec = None
                self.sidebar_local_steps = None

            # --- Execution (Control Horizon) ---
            if control_queue:
                if DEBUG:
                    # Save debug information
                    input("Press Enter to execute...")

                max_steps_by_time = int(remaining_time_sec / SIM_DT)
                if max_steps_by_time <= 0:
                    break
                exec_steps = min(len(control_queue), NEXEC, max_steps_by_time)

                for _ in range(exec_steps):
                    if not control_queue:
                        break

                    current_ctrl = control_queue.pop(0)
                    self.robot.update(SIM_DT, current_ctrl)
                    self._record_pose()
                    self._sim_time_elapsed += SIM_DT

                    # Update local/time budgets during execution
                    local_budget_time_sec = max(0.0, local_budget_time_sec - SIM_DT)
                    remaining_time_sec = max(0.0, remaining_time_sec - SIM_DT)
                    self._print_terminal_progress(remaining_time_sec)
                    self._replan_elapsed_sec_accum += SIM_DT
                    if remaining_time_sec <= 0.0:
                        break

                    obs = self.map_data.get_observations(self.robot.x, self.robot.y, self.robot.theta)
                    self.last_obs = obs
                    self.map_belief.update_belief(obs)
                    self._mask_reward_map_with_global_interests()

                    self.draw()
                    pygame.display.flip()
                    self._log_coral_status()
                    self._capture_frame()

            else:
                # Still refresh UI even if no controls are available for this iteration
                self.draw()
                pygame.display.flip()
                self._capture_frame()


        self._finalize_run(user_requested_exit=user_requested_exit)

    # =========================
    # Rendering
    # =========================
    def draw(self):
        """Draw the full dashboard."""
        # Clear background
        self.screen.fill((10, 15, 20))
        
        # Sidebar: minimap and runtime statistics
        self.draw_sidebar()
        
        # Center panel: observed simulation map
        self.draw_viewport(self.rect_center, mode="simulation", title="Simulation (observed Map)")
        
        # Right panel: target and reward view
        self.draw_viewport(self.rect_target, mode="target", title="Target/Reward (FLC+DLC)")
        
        # Panel separators
        pygame.draw.line(self.screen, (100, 100, 100), (self.SIDEBAR_W, 0), (self.SIDEBAR_W, self.HEIGHT), 2)
        pygame.draw.line(self.screen, (100, 100, 100), (self.rect_center.right, 0), (self.rect_center.right, self.HEIGHT), 2)

    def draw_sidebar(self):
        """Draw the sidebar."""
        # --- A. Minimap (Top Left) ---
        # Build a ground-truth minimap.
        gt_map = self.map_data.map.T # Shape: (cols, rows)
        
        # Build RGB image.
        map_rgb = np.zeros((*gt_map.shape, 3), dtype=np.uint8)
        
        # Sand
        map_rgb[gt_map == 0] = self.COLOR_SAND
        # Rock
        map_rgb[gt_map == 1] = self.COLOR_ROCK
        # Coral
        map_rgb[gt_map == 2] = self.COLOR_CORAL_GT
        
        # Create and scale the surface.
        surf = pygame.surfarray.make_surface(map_rgb)
        surf = pygame.transform.scale(surf, self.minimap_size)
        
        # Draw minimap.
        self.screen.blit(surf, self.minimap_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), self.minimap_rect, 2)
        
        # Draw trajectory on the minimap.
        scale_x = self.minimap_size[0] / self.map_data.width_meters
        scale_y = self.minimap_size[1] / self.map_data.height_meters
        if len(self.traj_world) >= 2:
            pts = []
            for wx, wy, _ in self.traj_world:
                px = self.minimap_rect.x + int(wx * scale_x)
                py = self.minimap_rect.y + int(wy * scale_y)
                pts.append((px, py))
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, (0, 200, 255), False, pts, 2)

        # Draw robot marker on the minimap.
        mini_rx = self.minimap_rect.x + int(self.robot.x * scale_x)
        mini_ry = self.minimap_rect.y + int(self.robot.y * scale_y)
        pygame.draw.circle(self.screen, (0, 255, 0), (mini_rx, mini_ry), 4)

        # Draw the FLS sensor footprint on the minimap.
        fls_r = int(FLS_RANGE * min(scale_x, scale_y))
        fls_half = math.radians(FLS_FOV_DEG) / 2
        fls_start = self.robot.theta - fls_half
        fls_end = self.robot.theta + fls_half
        fls_end_pos1 = (mini_rx + fls_r * math.cos(fls_start), mini_ry + fls_r * math.sin(fls_start))
        fls_end_pos2 = (mini_rx + fls_r * math.cos(fls_end), mini_ry + fls_r * math.sin(fls_end))
        pygame.draw.line(self.screen, (0, 200, 0), (mini_rx, mini_ry), fls_end_pos1, 1)
        pygame.draw.line(self.screen, (0, 200, 0), (mini_rx, mini_ry), fls_end_pos2, 1)
        fls_bbox = pygame.Rect(mini_rx - fls_r, mini_ry - fls_r, fls_r*2, fls_r*2)
        pygame.draw.arc(self.screen, (0, 200, 0), fls_bbox, -fls_end, -fls_start, 2)

        # --- B. Stats (Bottom Left) ---
        start_y = self.minimap_rect.bottom + 10
        line_h = 24
        
        def _fmt_or_null(val, fmt):
            return "null" if val is None else fmt.format(val)

        rx, ry, rtheta = self.sidebar_robot_pose or (self.robot.x, self.robot.y, self.robot.theta)

        remaining_txt = _fmt_or_null(self.sidebar_remaining_time_sec, "{:.2f}s")
        global_txt = _fmt_or_null(self.sidebar_global_budget, "{:.2f}")

        if self.sidebar_target is not None:
            tx, ty = self.sidebar_target
            target_txt = f"({tx:.2f}, {ty:.2f})"
        else:
            target_txt = "null"
        local_budget_txt = _fmt_or_null(self.sidebar_local_budget_sec, "{:.2f}s")

        status = self.sidebar_local_status
        status_label = {
            "optimal": "Optimal",
            "max_iter": "MaxIter",
            "cruise": "Cruise",
            "fallback_cruise": "FallbackCruise",
            "failed": "Failed",
            None: "null",
        }.get(status, status)
        solve_time_txt = _fmt_or_null(self.sidebar_local_solve_time_sec, "{:.3f}s")
        steps_txt = "null" if self.sidebar_local_steps is None else str(self.sidebar_local_steps)

        if self.sidebar_exec_index is not None and self.sidebar_exec_target is not None:
            ex_tx, ex_ty = self.sidebar_exec_target
            exec_txt = f"#{self.sidebar_exec_index} -> ({ex_tx:.2f}, {ex_ty:.2f})"
        else:
            exec_txt = "null"

        if self.sidebar_replan_info is not None:
            idx, actual_sec, planned_sec = self.sidebar_replan_info
            replan_txt = f"#{idx}: actual {actual_sec:.2f}s | planned {planned_sec:.2f}s"
        else:
            replan_txt = "null"

        texts = [
            ("Corals", self.font, (200, 200, 200)),
            (f"total {self.map_data.total_corals} | confirmed {self.map_data.confirmed_count}", self.font, (255, 210, 0)),
            ("Robot", self.font, (200, 200, 200)),
            (f"({rx:.2f}, {ry:.2f}) | head {math.degrees(rtheta):.1f}°", self.font, (200, 200, 200)),
            ("Budget", self.font, (220, 220, 220)),
            (f"remaining {remaining_txt} | global {global_txt}", self.font, (220, 220, 220)),
            ("Target", self.font, (200, 200, 200)),
            (f"{target_txt} | local {local_budget_txt}", self.font, (200, 200, 200)),
            ("LocalPlanner", self.font, (180, 220, 255)),
            (f"{status_label} | {solve_time_txt} | steps {steps_txt}", self.font, (180, 220, 255)),
            ("Executing", self.font, (180, 200, 180)),
            (f"{exec_txt}", self.font, (180, 200, 180)),
            ("Replan", self.font, (255, 200, 150)),
            (f"{replan_txt}", self.font, (255, 200, 150)),
            ("Debug Overlay", self.font, (180, 180, 255)),
            (f"interest grids: {'ON' if self.debug_show_interest_grids else 'OFF'} (G)", self.font, (180, 180, 255)),
        ]
        
        for i, (txt, font_obj, color) in enumerate(texts):
            s = font_obj.render(txt, True, color)
            self.screen.blit(s, (15, start_y + i * line_h))

    def draw_viewport(self, rect, mode, title):
        """
        Draw one dashboard viewport.

        rect: screen-space rectangle for the viewport.
        mode: 'simulation' or 'target'.
        """
        # Clip drawing to the viewport.
        self.screen.set_clip(rect)
        
        # Compute the visible map range.
        scale = self.view_scales.get(mode, 15)
        view_w_m = rect.width / scale
        view_h_m = rect.height / scale

        r_min, c_min = self.map_data.world_to_cell(self.robot.x - view_w_m/2 - 2, self.robot.y - view_h_m/2 - 2)
        r_max, c_max = self.map_data.world_to_cell(self.robot.x + view_w_m/2 + 2, self.robot.y + view_h_m/2 + 2)

        r_min = max(0, r_min); c_min = max(0, c_min)
        r_max = min(self.map_data.rows, r_max); c_max = min(self.map_data.cols, c_max)
        
        cell_px = int(self.map_data.cell_size * scale) + 1
        

        # --- 1. Draw Cells ---
        fls_obs = self.last_obs.get('fls') if self.last_obs and mode == "simulation" else None
        flc_obs = self.last_obs.get('flc') if self.last_obs and mode == "simulation" else None
        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                wx = c * self.map_data.cell_size + self.map_data.cell_size/2
                wy = r * self.map_data.cell_size + self.map_data.cell_size/2
                sx, sy, _ = self._world_to_view_pixel(wx, wy, rect, scale)
                
                # Coarse visibility culling.
                if not (rect.left - cell_px < sx < rect.right and rect.top - cell_px < sy < rect.bottom):
                    continue

                color = None
                is_confirmation_cell = False
                if mode == "simulation":
                    # Rendering priority:
                    # 1. Confirmed cells show their ground-truth class.
                    # 2. FLC detections highlight target detections.
                    # 3. FLS detections highlight substrate detections.
                    # 4. Unknown cells use a dark ground-truth tint.
                    status = self.map_data.status_mask[r, c]
                    val = self.map_data.map[r, c]
                    if status == 2: # Confirmed
                        if val == 0: color = self.COLOR_SAND
                        elif val == 1: color = self.COLOR_ROCK
                        elif val == 2: color = self.COLOR_CORAL_CONFIRMED
                    else:
                        if fls_obs and fls_obs.get('valid'):
                            r1, r2, c1, c2 = fls_obs['bbox']
                            if r1 <= r < r2 and c1 <= c < c2 and fls_obs['detections'][r - r1, c - c1]:
                                if val > 0:
                                    color = self.COLOR_FLS_TP
                                else:
                                    color = self.COLOR_FLS_FP

                        if flc_obs and flc_obs.get('valid'):
                            r1, r2, c1, c2 = flc_obs['bbox']
                            if r1 <= r < r2 and c1 <= c < c2 and flc_obs['detections'][r - r1, c - c1]:
                                if val == 2:
                                    color = self.COLOR_FLC_TP
                                else:
                                    color = self.COLOR_FLC_FP
                        if color is None:
                            if val == 2:
                                color = (80, 20, 20)    # unexplored coral (darker red)
                            elif val == 0:
                                color = (70, 60, 35)    # unexplored sand (darker sand)
                            else:
                                color = (20, 20, 20)    # unexplored rock (darker gray)

                elif mode == "target":
                    # Visualization uses full coral belief (P(coral)),
                    # not masked reward/visited logic.
                    prob_coral = self.map_belief.prob_coral[r, c]
                    intensity = min(int(prob_coral * 255), 255)
                    color = (intensity, 40, 40)
                    is_confirmation_cell = bool(self.map_belief.confirmation_map[r, c])
                else:
                    continue

                pygame.draw.rect(self.screen, color, (sx - cell_px//2, sy - cell_px//2, cell_px, cell_px))
                if mode == "target" and is_confirmation_cell:
                    marker_r = max(1, cell_px // 4)
                    pygame.draw.circle(self.screen, (255, 215, 0), (int(sx), int(sy)), marker_r)

        # --- 1. DRAW TARGET OVERLAY (Target View Only)
        if mode == "target":
            self.draw_target_overlay(rect, scale)

        # --- 2. Draw Robot & Sensors Overlay ---
        self.draw_robot_overlay(rect, scale)

        # --- 3. Draw Title Label ---
        label = self.title_font.render(title, True, (255, 255, 255))
        pygame.draw.rect(self.screen, (0, 0, 0, 150), (rect.x + 10, rect.y + 10, label.get_width()+10, 30))
        self.screen.blit(label, (rect.x + 15, rect.y + 15))
        
        # Release clip.
        self.screen.set_clip(None)

    def draw_robot_overlay(self, rect, scale):
        """Draw the robot and sensor footprints in a viewport."""
        sx, sy, _ = self._world_to_view_pixel(self.robot.x, self.robot.y, rect, scale)
        
        # FLS Arc (long-range substrate scout)
        fls_r = FLS_RANGE * scale
        fls_half = math.radians(FLS_FOV_DEG) / 2
        fls_start = self.robot.theta - fls_half
        fls_end = self.robot.theta + fls_half
        fls_end_pos1 = (sx + fls_r * math.cos(fls_start), sy + fls_r * math.sin(fls_start))
        fls_end_pos2 = (sx + fls_r * math.cos(fls_end), sy + fls_r * math.sin(fls_end))
        pygame.draw.line(self.screen, (0, 200, 0), (sx, sy), fls_end_pos1, 1)
        pygame.draw.line(self.screen, (0, 200, 0), (sx, sy), fls_end_pos2, 1)
        fls_bbox = pygame.Rect(sx - fls_r, sy - fls_r, fls_r*2, fls_r*2)
        pygame.draw.arc(self.screen, (0, 200, 0), fls_bbox, -fls_end, -fls_start, 2)

        # FLC Arc (mid-range target scout)
        flc_r = FLC_RANGE * scale
        flc_half = math.radians(FLC_FOV_DEG) / 2
        flc_start = self.robot.theta - flc_half
        flc_end = self.robot.theta + flc_half
        flc_end_pos1 = (sx + flc_r * math.cos(flc_start), sy + flc_r * math.sin(flc_start))
        flc_end_pos2 = (sx + flc_r * math.cos(flc_end), sy + flc_r * math.sin(flc_end))
        pygame.draw.line(self.screen, (255, 140, 0), (sx, sy), flc_end_pos1, 1)
        pygame.draw.line(self.screen, (255, 140, 0), (sx, sy), flc_end_pos2, 1)
        flc_bbox = pygame.Rect(sx - flc_r, sy - flc_r, flc_r*2, flc_r*2)
        pygame.draw.arc(self.screen, (255, 140, 0), flc_bbox, -flc_end, -flc_start, 2)

        # DLC Box (footprint)
        dlc_px = int(DLC_FOOTPRINT * scale)
        surf = pygame.Surface((dlc_px, dlc_px), pygame.SRCALPHA)
        pygame.draw.rect(surf, (255, 0, 0, 120), surf.get_rect(), 2)
        rot_surf = pygame.transform.rotate(surf, math.degrees(-self.robot.theta))
        rot_rect = rot_surf.get_rect(center=(sx, sy))
        self.screen.blit(rot_surf, rot_rect)
        
        # Robot Body
        r_rad = 6
        p1 = (sx + r_rad * math.cos(self.robot.theta), sy + r_rad * math.sin(self.robot.theta))
        p2 = (sx + r_rad * math.cos(self.robot.theta + 2.5), sy + r_rad * math.sin(self.robot.theta + 2.5))
        p3 = (sx + r_rad * math.cos(self.robot.theta - 2.5), sy + r_rad * math.sin(self.robot.theta - 2.5))
        pygame.draw.polygon(self.screen, (0, 255, 255), [p1, p2, p3])

    def _world_to_view_pixel(self, wx, wy, view_rect, scale=None):
        """
        Convert world coordinates to viewport pixels.

        The viewport is centered on the robot.
        """
        # View center in screen coordinates.
        cx = view_rect.x + view_rect.width // 2
        cy = view_rect.y + view_rect.height // 2
        
        # Zoom level.
        if scale is None:
            scale = 15 
        
        # Offset from robot-centered view.
        sx = cx + (wx - self.robot.x) * scale
        sy = cy + (wy - self.robot.y) * scale
        return int(sx), int(sy), scale

    def draw_target_overlay(self, rect, scale):
        """
        Draw grid lines and global-planner overlays in the target/reward view.
        """
        # === 1. 绘制物理网格线 (Grid Lines) ===
        step_m = GLOBAL_PLANNER_CONFIG.grid_interval_m
        
        # For performance, only draw grid lines inside the visible range.
        view_w_m = rect.width / scale
        view_h_m = rect.height / scale
        
        # Compute physical bounds of the current viewport.
        min_x_m = self.robot.x - view_w_m / 2
        max_x_m = self.robot.x + view_w_m / 2
        min_y_m = self.robot.y - view_h_m / 2
        max_y_m = self.robot.y + view_h_m / 2

        # Vertical grid lines.
        start_x_idx = int(min_x_m // step_m)
        end_x_idx = int(max_x_m // step_m) + 1
        
        for i in range(start_x_idx, end_x_idx + 1):
            x_m = i * step_m
            sx, _, _ = self._world_to_view_pixel(x_m, self.robot.y, rect, scale)
            pygame.draw.line(self.screen, (255, 255, 255, 80), (sx, rect.top), (sx, rect.bottom), 1)

        # Horizontal grid lines.
        start_y_idx = int(min_y_m // step_m)
        end_y_idx = int(max_y_m // step_m) + 1
        
        for i in range(start_y_idx, end_y_idx + 1):
            y_m = i * step_m
            _, sy, _ = self._world_to_view_pixel(self.robot.x, y_m, rect, scale)
            pygame.draw.line(self.screen, (255, 255, 255, 80), (rect.left, sy), (rect.right, sy), 1)

        # Debug overlay: interest grids (target_grids U visited).
        if self.debug_show_interest_grids and self._last_interest_grids:
            for (r, c) in self._last_interest_grids:
                x1_m = c * step_m
                y1_m = r * step_m
                x2_m = (c + 1) * step_m
                y2_m = (r + 1) * step_m
                sx1, sy1, _ = self._world_to_view_pixel(x1_m, y1_m, rect, scale)
                sx2, sy2, _ = self._world_to_view_pixel(x2_m, y2_m, rect, scale)
                left = min(sx1, sx2)
                top = min(sy1, sy2)
                w = max(1, abs(sx2 - sx1))
                h = max(1, abs(sy2 - sy1))
                grid_rect = pygame.Rect(left, top, w, h)
                if grid_rect.colliderect(rect):
                    pygame.draw.rect(self.screen, (255, 220, 0), grid_rect, 2)

            target_grids = getattr(self.global_planner, "target_grids", None)
            if target_grids:
                for (r, c) in target_grids:
                    x1_m = c * step_m
                    y1_m = r * step_m
                    x2_m = (c + 1) * step_m
                    y2_m = (r + 1) * step_m
                    sx1, sy1, _ = self._world_to_view_pixel(x1_m, y1_m, rect, scale)
                    sx2, sy2, _ = self._world_to_view_pixel(x2_m, y2_m, rect, scale)
                    left = min(sx1, sx2)
                    top = min(sy1, sy2)
                    w = max(1, abs(sx2 - sx1))
                    h = max(1, abs(sy2 - sy1))
                    grid_rect = pygame.Rect(left, top, w, h)
                    if grid_rect.colliderect(rect):
                        pygame.draw.rect(self.screen, (0, 255, 255), grid_rect, 3)

            # debug_label = self.font.render("Debug: yellow=interest, cyan=target", True, (255, 240, 120))
            # self.screen.blit(debug_label, (rect.x + 14, rect.y + 46))

        # Visited grids.
        if hasattr(self.global_planner, "visited") and self.global_planner.visited:
            for (r, c) in self.global_planner.visited:
                wx = (c + 0.5) * step_m
                wy = (r + 0.5) * step_m
                sx, sy, _ = self._world_to_view_pixel(wx, wy, rect, scale)
                if rect.left <= sx <= rect.right and rect.top <= sy <= rect.bottom:
                    pygame.draw.rect(self.screen, (80, 120, 200), (sx - 4, sy - 4, 8, 8), 1)

        # Next global-planner nodes.
        path = getattr(self.global_planner, "last_path", None)
        if path and len(path) >= 2:
            future = path[1:6]
            pts = []
            rx, ry, _ = self._world_to_view_pixel(self.robot.x, self.robot.y, rect, scale)
            pts.append((int(rx), int(ry)))
            for px, py in future:
                wx = px * step_m
                wy = py * step_m
                sx, sy, _ = self._world_to_view_pixel(wx, wy, rect, scale)
                pts.append((int(sx), int(sy)))
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, (0, 200, 255), False, pts, 2)
            for i, (sx, sy) in enumerate(pts[1:]):
                r = 6 if i == 0 else 4
                pygame.draw.circle(self.screen, (0, 200, 255), (sx, sy), r, 2)

        # Next target.
        if self.next_target is None:
            return
        tx, ty = self.next_target
        sx, sy, _ = self._world_to_view_pixel(tx, ty, rect, scale)
        pygame.draw.circle(self.screen, (0, 255, 0), (int(sx), int(sy)), 8, 2)
        pygame.draw.circle(self.screen, (0, 255, 0), (int(sx), int(sy)), 3)


if __name__ == "__main__":
    start_poses = [[0, 0, np.pi / 4], [50, 0, 3*np.pi / 4], [0, 50, -np.pi / 4], [50, 50, -3*np.pi / 4]]
    
    import argparse
    parser = argparse.ArgumentParser(description="AUV simulation runner")
    parser.add_argument("-budget", type=int, default=2000, help="Total time budget in seconds")
    parser.add_argument("-map_index", type=int, default=1, help="Map index for Area_2_map_{index}")
    parser.add_argument("-start_pose", type=int, default=0, help="Start pose index (0-3)")
    args = parser.parse_args()

    TOTAL_TIME_BUDGET_SECS = args.budget
    map_index = args.map_index
    start_pose_index = args.start_pose

    start_pose = start_poses[start_pose_index]
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MAP_DIR = os.path.join(BASE_DIR, "map", "planning_maps", f"Area_2_map_{map_index}", "map.npy")
    MAP_SIZE = np.load(MAP_DIR).shape  # (rows, cols) cells



    # Keep the startup pose away from map boundaries to avoid early local-planner infeasibility.
    START_MARGIN_M = 0.1
    def _clip_start_to_map(v, upper_bound_m, margin_m):
        upper_safe = max(margin_m, upper_bound_m - margin_m)
        return float(np.clip(v, margin_m, upper_safe))
    START_X = _clip_start_to_map(start_pose[0], MAP_SIZE[1] * CELL_SIZE, START_MARGIN_M)
    START_Y = _clip_start_to_map(start_pose[1], MAP_SIZE[0] * CELL_SIZE, START_MARGIN_M)
    START_THETA = start_pose[2]

    GLOBAL_PLANNER_CONFIG = GlobalPlannerConfig(path_budget=TOTAL_TIME_BUDGET_SECS / GRID_TIME_SEC)


    game = Game()
    # game.run_manual()
    game.run_planner_HIMoS()
