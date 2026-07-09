import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import casadi as ca
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import os

# --- 引入项目特定的配置 ---
# 为了保证代码在没有特定环境时也能运行，做了简单的 try-except 处理
# 实际使用时会使用 param 中的值
from param import FLS_RANGE, FLS_FOV_DEG, FLC_RANGE, FLC_FOV_DEG, DLC_FOOTPRINT
from param import CELL_SIZE
from param import fls_true_positive_rate, fls_false_positive_rate
from param import flc_true_positive_rate, flc_false_positive_rate
from param import MAX_ANGULAR_VELOCITY, MAX_VELOCITY

from param import (
    PLANNING_TIMESTEP_SIZE,
    N_CANDIDATES_TO_OPTIMIZE,
    OBSERVATION_TIMES_PER_STEP,
    FLC_ENTROPY_MAP_DOWNSAMPLE,
    FLS_ENTROPY_MAP_DOWNSAMPLE,
    N_INFO,
)
from param import CRUISE_DIST_THRESHOLD
from param import NEXEC
from param import DEBUG
from param import CONF_THRESHOLD

class UnifiedLocalPlanner:
    def __init__(self):
        """
        Unified Receding Horizon Information Search Planner (Omnidirectional Version)
        """
        # --- 地图与网格参数 ---
        self.cell_size = CELL_SIZE        
        
        # --- 规划视界参数 ---
        self.dt = PLANNING_TIMESTEP_SIZE     # 规划时间步长
        
        # --- 机器人物理限制 ---
        self.v_max = MAX_VELOCITY      # m/s
        self.omega_max = MAX_ANGULAR_VELOCITY    # rad/s

        # --- 声呐参数 (Exploration) ---
        self.sonar_range = FLS_RANGE
        self.sonar_fov_rad = np.deg2rad(FLS_FOV_DEG)
        self.nu_entropy = 2.0   # 熵代理参数

        # --- 前向相机参数 (Exploration) ---
        self.fwd_range = FLC_RANGE
        self.fwd_fov_rad = np.deg2rad(FLC_FOV_DEG)
        
        # --- 相机参数 (Exploitation) ---
        self.cam_L = 0.8 * DLC_FOOTPRINT    # 视场边长 (m) 因为是soft-square，所以取0.8倍footprint，这样才能保证能框住目标
        self.eta_cam = 2.0      # 视觉信息率
        self.lambda_sat = 1.0   # 饱和系数
        self.alpha_soft = 0.15  # Soft-Square 平滑因子
        
        # --- 权重参数 ---
        self.w_sub = 10        # FLS 探索权重
        self.w_search = 100     # FLC 探索权重
        self.w_confirm = 50.0   # 确认权重 (Exploitation 优先级高)
        self.w_goal = 1       # 目标引导
        self.w_energy = 0.0    # 节能
        self.w_smooth = 0.0    # 平滑 (Jerk penalty)
        # self.w_energy = 0.05    # 节能
        # self.w_smooth = 1.0    # 平滑 (Jerk penalty)

    def _sector_field_proxy(self, x, y, theta, grid_x, grid_y, sensor_range, sensor_fov_rad, p_tp_func, p_fp_func):
        """
        Differentiable Sector FOV proxy for FLS / FLC.
        """
        gamma_d = 3.0
        gamma_a = 15.0

        # 1. 距离场
        d_sq = (grid_x - x)**2 + (grid_y - y)**2 + 1e-6
        d = ca.sqrt(d_sq)
        f_dist = 1.0 / (1.0 + ca.exp(gamma_d * (d - sensor_range)))

        # 2. 角度场
        vec_x = grid_x - x
        vec_y = grid_y - y
        robot_dir_x = ca.cos(theta)
        robot_dir_y = ca.sin(theta)

        dot_prod = vec_x * robot_dir_x + vec_y * robot_dir_y
        cos_phi = dot_prod / d
        cos_beta = np.cos(sensor_fov_rad / 2.0)

        f_ang = 1.0 / (1.0 + ca.exp(gamma_a * (cos_beta - cos_phi)))
        alpha = f_dist * f_ang  # 几何观测强度

        # 3. 计算动态信息率 eta(d)
        d_norm = ca.fmin(d / sensor_range, 0.99)
        p_tp = p_tp_func(d_norm)
        p_fp = p_fp_func(d_norm)

        # 防止数值不稳定 (log 0)
        p_tp = ca.fmax(p_tp, 1e-3)
        p_fp = ca.fmax(p_fp, 1e-3)
        p_tn = 1.0 - p_fp
        p_fn = 1.0 - p_tp

        # Eq. (8) Symmetric Information Rate
        term1 = ca.fabs(ca.log(p_tp / p_fp))
        term2 = ca.fabs(ca.log(p_tn / p_fn))
        eta_d = 0.5 * (term1 + term2)

        return eta_d * alpha

    def _sonar_field_proxy(self, x, y, theta, grid_x, grid_y):
        """
        论文 Sec. 3.3.1: Acoustic Scouting Model (Sector FOV)
        """
        return self._sector_field_proxy(
            x, y, theta, grid_x, grid_y,
            self.sonar_range, self.sonar_fov_rad,
            fls_true_positive_rate, fls_false_positive_rate
        )

    def _flc_field_proxy(self, x, y, theta, grid_x, grid_y):
        """
        论文 Sec. 3.3.1: Forward Camera Scouting Model (Sector FOV)
        """
        return self._sector_field_proxy(
            x, y, theta, grid_x, grid_y,
            self.fwd_range, self.fwd_fov_rad,
            flc_true_positive_rate, flc_false_positive_rate
        )

    def _camera_field_proxy(self, x, y, theta, target_x, target_y):
        """
        论文 Sec. 3.3.2: Visual Confirmation Model (Oriented Soft-Square)
        """
        # 转换到 Body Frame
        dx = target_x - x
        dy = target_y - y
        cos_t = ca.cos(theta)
        sin_t = ca.sin(theta)
        
        # 全向机器人 Body Frame 投影
        d_lon = cos_t * dx + sin_t * dy 
        d_lat = -sin_t * dx + cos_t * dy
        
        half_L = self.cam_L / 2.0
        eps = 1e-4
        
        abs_lon = ca.sqrt(d_lon**2 + eps)
        abs_lat = ca.sqrt(d_lat**2 + eps)
        
        f_lon = 1.0 / (1.0 + ca.exp(-(half_L - abs_lon) / self.alpha_soft))
        f_lat = 1.0 / (1.0 + ca.exp(-(half_L - abs_lat) / self.alpha_soft))
        
        return f_lon * f_lat

    def _plan_cruise(self, current_pose, target, map_dims_m):
        """
        Lightweight cruise controller for long-distance travel.
        Generates a short control sequence without CasADi optimization.
        """
        n_steps = max(1, int(NEXEC))

        x, y, theta = float(current_pose[0]), float(current_pose[1]), float(current_pose[2])
        traj = [np.array([x, y, theta], dtype=np.float32)]
        u_seq = []

        for k in range(n_steps):
            dx = target[0] - x
            dy = target[1] - y
            dist = float(np.hypot(dx, dy))

            if dist < 1e-3:
                u = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            else:
                steps_left = max(1, n_steps - k)
                v_des = min(self.v_max, dist / (self.dt * steps_left))
                vx_w = v_des * dx / (dist + 1e-6)
                vy_w = v_des * dy / (dist + 1e-6)

                # World -> Body
                c = np.cos(theta)
                s = np.sin(theta)
                vx_b =  c * vx_w + s * vy_w
                vy_b = -s * vx_w + c * vy_w

                # Heading control (no extra hyper-params)
                desired_heading = np.arctan2(dy, dx)
                heading_err = (desired_heading - theta + np.pi) % (2 * np.pi) - np.pi
                omega = np.clip(heading_err, -self.omega_max, self.omega_max)

                vx_b = np.clip(vx_b, -self.v_max, self.v_max)
                vy_b = np.clip(vy_b, -self.v_max, self.v_max)
                u = np.array([vx_b, vy_b, omega], dtype=np.float32)

            # Forward integrate (omni)
            x += (u[0] * np.cos(theta) - u[1] * np.sin(theta)) * self.dt
            y += (u[0] * np.sin(theta) + u[1] * np.cos(theta)) * self.dt
            theta += u[2] * self.dt

            # Keep within map bounds (same margins as optimizer)
            if map_dims_m is not None:
                x = float(np.clip(x, 0.2, map_dims_m[0] - 0.2))
                y = float(np.clip(y, 0.2, map_dims_m[1] - 0.2))

            traj.append(np.array([x, y, theta], dtype=np.float32))
            u_seq.append(u)

        return {
            'success': True,
            'control_sequence': np.array(u_seq, dtype=np.float32),
            'trajectory': np.array(traj, dtype=np.float32),
            'candidates': np.empty((0, 2), dtype=np.float32),
            'final_entropy_map': None,
            'final_entropy_map_substrate': None,
        }
    
    def plan(self, time_budget, current_pose, target, 
             log_odds_s, log_odds_c, confirmation_map,
             fls_origin, flc_origin):
        """
        Unified Planning Function
        :param current_pose: np.array([x, y, theta]) in FLC slice frame
        :param target: np.array([x, y]) in FLC slice frame
        :param log_odds_s: FLS log-odds slice (larger)
        :param log_odds_c: FLC log-odds slice (smaller, P(c))
        :param confirmation_map: DLC candidate mask (binary, FLC slice)
        :param fls_origin: FLS slice origin in world frame (x, y)
        :param flc_origin: FLC slice origin in world frame (x, y)
        """
        # --- 规划参数 N ---
        N = max(int(np.ceil(time_budget / self.dt)), 2)  # 至少规划2步
        N_info = int(min(N, N_INFO))
        obs_times_per_step = OBSERVATION_TIMES_PER_STEP

        t0 = time.time()
        # --- 0. 基础地图信息 ---
        # FLC slice as the planning frame
        rows_c, cols_c = log_odds_c.shape
        sliced_map_size_m = (cols_c * self.cell_size, rows_c * self.cell_size) # map size (meters)

        # --- 0.1 远距离赶路模式 ---
        dist_to_target = np.linalg.norm(target - current_pose[:2])
        if dist_to_target > CRUISE_DIST_THRESHOLD:
            res = self._plan_cruise(current_pose, target, map_dims_m=sliced_map_size_m)
            res['status'] = "cruise"
            res['solve_time_sec'] = time.time() - t0
            res['steps'] = len(res.get('control_sequence', []))
            if DEBUG:
                print(f"[UnifiedLocalPlanner] Cruise mode: dist={dist_to_target:.2f}m, steps={len(res['control_sequence'])}")
            return res

        # --- 0.2 网格坐标生成 ---

        x_c = np.linspace(self.cell_size/2, sliced_map_size_m[0]-self.cell_size/2, cols_c)
        y_c = np.linspace(self.cell_size/2, sliced_map_size_m[1]-self.cell_size/2, rows_c)
        self.X_grid_c, self.Y_grid_c = np.meshgrid(x_c, y_c) # FLC coordinate grids (meters)

        # FLS slice grid (larger), shift into FLC frame if origins provided
        rows_s, cols_s = log_odds_s.shape
        fls_offset = np.array([0.0, 0.0], dtype=np.float32)
        if fls_origin is not None and flc_origin is not None:
            fls_offset = np.array(fls_origin, dtype=np.float32) - np.array(flc_origin, dtype=np.float32)

        x_s = np.linspace(self.cell_size/2, cols_s * self.cell_size - self.cell_size/2, cols_s)
        y_s = np.linspace(self.cell_size/2, rows_s * self.cell_size - self.cell_size/2, rows_s)
        X_grid_s, Y_grid_s = np.meshgrid(x_s, y_s)
        X_grid_s = X_grid_s + fls_offset[0]
        Y_grid_s = Y_grid_s + fls_offset[1]

        # --- 1. 数据预处理 ---
        # FLC/FLS use different downsample factors
        downsample_c = FLC_ENTROPY_MAP_DOWNSAMPLE
        downsample_s = FLS_ENTROPY_MAP_DOWNSAMPLE
        # 1.1 FLS & FLC: 提取初始置信度幅度 |l|
        idx_slice_rs = slice(0, rows_s, downsample_s)
        idx_slice_cs = slice(0, cols_s, downsample_s)
        opt_X_grid_s = X_grid_s[idx_slice_rs, idx_slice_cs].flatten()
        opt_Y_grid_s = Y_grid_s[idx_slice_rs, idx_slice_cs].flatten()
        opt_lambda_s_init = np.abs(log_odds_s)[idx_slice_rs, idx_slice_cs].flatten()

        idx_slice_rc = slice(0, rows_c, downsample_c)
        idx_slice_cc = slice(0, cols_c, downsample_c)
        opt_X_grid_c = self.X_grid_c[idx_slice_rc, idx_slice_cc].flatten()
        opt_Y_grid_c = self.Y_grid_c[idx_slice_rc, idx_slice_cc].flatten()
        opt_lambda_c_init = np.abs(log_odds_c)[idx_slice_rc, idx_slice_cc].flatten()

        # 1.2 DLC: 从 confirmation_map 提取候选点
        candidate_indices = np.argwhere(confirmation_map)
        candidates = []
        for idx in candidate_indices: # candidate coordinates in meters
            # numpy: [row(y), col(x)]
            cx = idx[1] * self.cell_size + self.cell_size/2
            cy = idx[0] * self.cell_size + self.cell_size/2
            candidates.append([cx, cy])
        
        candidates = np.array(candidates)
        n_candidates = len(candidates)
        
        # # 只保留距离 <= 2m 的候选点
        # if n_candidates > 0:
        #     dists = np.linalg.norm(candidates - current_pose[:2], axis=1)
        #     candidates = candidates[dists <= 2.0]
        #     n_candidates = len(candidates)

        # 如果仍然过多，则保留最近的 Top K
        if n_candidates > N_CANDIDATES_TO_OPTIMIZE:
            dists = np.linalg.norm(candidates - current_pose[:2], axis=1)
            top_k_idx = np.argsort(dists)[:N_CANDIDATES_TO_OPTIMIZE]
            candidates = candidates[top_k_idx]
            n_candidates = len(candidates)

        # --- 2. 构建 CasADi 优化问题 ---
        opti = ca.Opti()
        
        # 决策变量
        X = opti.variable(3, N+1) # [x, y, theta]
        U = opti.variable(3, N)   # [vx, vy, omega] (全向)
        
        # 初始化约束
        opti.subject_to(X[:, 0] == current_pose)
        
        J_confirm = 0
        J_smooth = 0
        J_energy = 0
        J_goal = 0
        
        # 2.1 动力学与约束循环
        for k in range(N):
            # --- 全向轮动力学 (Omnidirectional) ---
            theta = X[2, k]
            vx_body = U[0, k]
            vy_body = U[1, k]
            omega   = U[2, k]
            
            # Body Frame -> World Frame
            dx = vx_body * ca.cos(theta) - vy_body * ca.sin(theta)
            dy = vx_body * ca.sin(theta) + vy_body * ca.cos(theta)
            
            opti.subject_to(X[0, k+1] == X[0, k] + dx * self.dt)
            opti.subject_to(X[1, k+1] == X[1, k] + dy * self.dt)
            opti.subject_to(X[2, k+1] == X[2, k] + omega * self.dt)
            
            # --- 物理限幅 (Body Frame Velocities) ---
            opti.subject_to(opti.bounded(-self.v_max, vx_body, self.v_max))
            opti.subject_to(opti.bounded(-self.v_max, vy_body, self.v_max))
            opti.subject_to(opti.bounded(-self.omega_max, omega, self.omega_max))
            
            # 地图边界
            opti.subject_to(opti.bounded(0.2, X[0, k+1], sliced_map_size_m[0]-0.2))
            opti.subject_to(opti.bounded(0.2, X[1, k+1], sliced_map_size_m[1]-0.2))
            
            # --- 基础成本 ---
            J_energy += ca.sumsqr(U[:, k])
            
            if k > 0:
                J_smooth += ca.sumsqr(U[:, k] - U[:, k-1]) # Min Jerk
            
            # 过程目标引导
            J_goal += 0.1 * ((X[0, k] - target[0])**2 + (X[1, k] - target[1])**2)

        # 终端目标引导
        J_goal += 10.0 * ((X[0, N] - target[0])**2 + (X[1, N] - target[1])**2)

        # --- 3. 统一信息场计算 ---
        
        # 3.1 FLS Exploration
        sonar_alpha_accum = ca.MX.zeros(opt_X_grid_s.shape[0])

        # 3.2 FLC Exploration
        fwd_alpha_accum = ca.MX.zeros(opt_X_grid_c.shape[0])

        # 3.3 DLC Confirmation
        cam_alpha_accum = ca.MX.zeros(n_candidates) if n_candidates > 0 else 0
        if n_candidates > 0:
            cand_x = ca.DM(candidates[:, 0])
            cand_y = ca.DM(candidates[:, 1])
        
        # 累积信息
        for k in range(1, N_info + 1):
            pose_x = X[0, k]
            pose_y = X[1, k]
            pose_th = X[2, k]
            
            # FLS (Exploration)
            step_info_gain_s = self._sonar_field_proxy(pose_x, pose_y, pose_th, opt_X_grid_s, opt_Y_grid_s)
            sonar_alpha_accum += step_info_gain_s * obs_times_per_step

            # FLC (Exploration)
            step_info_gain_c = self._flc_field_proxy(pose_x, pose_y, pose_th, opt_X_grid_c, opt_Y_grid_c)
            fwd_alpha_accum += step_info_gain_c * obs_times_per_step

            # Camera (Exploitation)
            if n_candidates > 0:
                vis = self._camera_field_proxy(pose_x, pose_y, pose_th, cand_x, cand_y)
                cam_alpha_accum += vis * obs_times_per_step

        # --- 4. 目标函数构建 ---
        
        # 4.1 Exploration: Minimize Final Entropy Proxy (FLS + FLC)
        lambda_sonar_final = opt_lambda_s_init + sonar_alpha_accum
        lambda_fwd_final = opt_lambda_c_init + fwd_alpha_accum

        sigmoid_s = 1.0 / (1.0 + ca.exp(-lambda_sonar_final))
        entropy_s = ca.log(1 + ca.exp(lambda_sonar_final)) - lambda_sonar_final * sigmoid_s
        J_explore_s = ca.sum1(entropy_s)

        sigmoid_c = 1.0 / (1.0 + ca.exp(-lambda_fwd_final))
        entropy_c = ca.log(1 + ca.exp(lambda_fwd_final)) - lambda_fwd_final * sigmoid_c
        J_explore_c = ca.sum1(entropy_c)

        # --- 方法 B: 高斯核近似 (Gaussian Proxy) [已注释] ---
        # nu_entropy 控制高斯的宽度
        # final_entropy_proxy = ca.exp(- (lambda_sonar_final**2) / (2 * self.nu_entropy**2))
        # J_explore = ca.sum1(final_entropy_proxy) 
        
        # 4.2 Confirmation: Maximize Prob
        if n_candidates > 0:
            lambda_cam_final = self.eta_cam * cam_alpha_accum
            J_confirm = ca.sum1(ca.exp(-self.lambda_sat * lambda_cam_final))
        else:
            J_confirm = 0

        J_total = (self.w_sub * J_explore_s +
                   self.w_search * J_explore_c +
                   self.w_confirm * J_confirm + 
                   self.w_goal * J_goal)
        # J_total = (self.w_sub * J_explore_s +
        #            self.w_search * J_explore_c +
        #            self.w_confirm * J_confirm + 
        #            self.w_goal * J_goal + 
        #            self.w_smooth * J_smooth + 
        #            self.w_energy * J_energy)
        
        opti.minimize(J_total)
        
        # --- 5. 求解 ---
        opti.set_initial(X[0, :], np.linspace(current_pose[0], target[0], N+1))
        opti.set_initial(X[1, :], np.linspace(current_pose[1], target[1], N+1))
        heading_to_target = float(np.arctan2(target[1] - current_pose[1], target[0] - current_pose[0]))
        heading_err = (heading_to_target - float(current_pose[2]) + np.pi) % (2 * np.pi) - np.pi
        theta_end_guess = float(current_pose[2]) + heading_err
        opti.set_initial(X[2, :], np.linspace(float(current_pose[2]), theta_end_guess, N + 1))

        dist_to_target_safe = max(1e-6, float(np.linalg.norm(target - current_pose[:2])))
        v_guess = min(self.v_max * 0.6, dist_to_target_safe / max(self.dt * N, self.dt))
        vx_w_guess = v_guess * (target[0] - current_pose[0]) / dist_to_target_safe
        vy_w_guess = v_guess * (target[1] - current_pose[1]) / dist_to_target_safe
        c0 = np.cos(float(current_pose[2]))
        s0 = np.sin(float(current_pose[2]))
        vx_b_guess = float(np.clip(c0 * vx_w_guess + s0 * vy_w_guess, -self.v_max, self.v_max))
        vy_b_guess = float(np.clip(-s0 * vx_w_guess + c0 * vy_w_guess, -self.v_max, self.v_max))
        omega_guess = float(np.clip(heading_err / max(self.dt * N, self.dt), -self.omega_max, self.omega_max))
        opti.set_initial(U[0, :], vx_b_guess)
        opti.set_initial(U[1, :], vy_b_guess)
        opti.set_initial(U[2, :], omega_guess)

        opts = {'print_time': False, 'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'ipopt.max_iter': 80, 'ipopt.tol': 1e-1}
        opti.solver('ipopt', opts)
        

        # 定义一个提取结果的内部函数，因为成功和超时都需要提取同样的变量
        def extract_result(val_func):
            traj = np.vstack((val_func(X[0, :]), val_func(X[1, :]), val_func(X[2, :]))).T
            
            # --- 提取并还原预测的 Final Lambda (FLC) ---
            final_lambda_c_small_flat = val_func(lambda_fwd_final)
            
            # Reshape & Upsample (FLC)
            rows, cols = log_odds_c.shape
            r_indices = np.arange(0, rows, downsample_c)
            c_indices = np.arange(0, cols, downsample_c)
            rows_small = len(r_indices)
            cols_small = len(c_indices)
            
            final_lambda_small = final_lambda_c_small_flat.reshape(rows_small, cols_small)
            
            # Upsample (Repeat)
            final_lambda_large = final_lambda_small.repeat(downsample_c, axis=0).repeat(downsample_c, axis=1)
            final_lambda_large = final_lambda_large[:rows, :cols]
            
            # 计算 Entropy Map
            sigma_L = 1.0 / (1.0 + np.exp(-final_lambda_large))
            pred_entropy_map_c = np.log(1 + np.exp(final_lambda_large)) - final_lambda_large * sigma_L

            # --- 提取并还原预测的 Final Lambda (FLS) ---
            final_lambda_s_small_flat = val_func(lambda_sonar_final)

            rows_s, cols_s = log_odds_s.shape
            r_indices_s = np.arange(0, rows_s, downsample_s)
            c_indices_s = np.arange(0, cols_s, downsample_s)
            rows_s_small = len(r_indices_s)
            cols_s_small = len(c_indices_s)

            final_lambda_s_small = final_lambda_s_small_flat.reshape(rows_s_small, cols_s_small)
            final_lambda_s_large = final_lambda_s_small.repeat(downsample_s, axis=0).repeat(downsample_s, axis=1)
            final_lambda_s_large = final_lambda_s_large[:rows_s, :cols_s]

            sigma_L_s = 1.0 / (1.0 + np.exp(-final_lambda_s_large))
            pred_entropy_map_s = np.log(1 + np.exp(final_lambda_s_large)) - final_lambda_s_large * sigma_L_s

            # shape 转为 (N, 3) -> [[vx0, vy0, w0], [vx1, vy1, w1], ...]
            u_seq = np.vstack((val_func(U[0, :]), val_func(U[1, :]), val_func(U[2, :]))).T
            
            return u_seq, traj, pred_entropy_map_c, pred_entropy_map_s, val_func(cam_alpha_accum) if n_candidates > 0 else None

        res = {}
        try:
            sol = opti.solve()
            # 1. 完美求解成功
            u_opt_seq, traj, pred_entropy_map_c, pred_entropy_map_s, cam_accum_val = extract_result(sol.value)
            res['success'] = True
            solve_time_sec = time.time() - t0
            res['status'] = "optimal"
            # print(f"[UnifiedLocalPlanner] Optimal solution found in {solve_time_sec:.3f}s | "
            #       f"local_budget={time_budget:.2f}s | steps={N}")
            
                
        except Exception as e:
            # 2. 求解失败，判断原因
            err_msg = str(e)
            # 如果仅仅是达到最大迭代次数，我们依然信任当前结果 (opti.debug.value)
            if "Maximum_Iterations_Exceeded" in err_msg:
                solve_time_sec = time.time() - t0
                res['status'] = "max_iter"
                # print(f"[UnifiedLocalPlanner] Warning: Max iter reached ({solve_time_sec:.3f}s). "
                #       f"Using sub-optimal result. local_budget={time_budget:.2f}s | steps={N}")
                u_opt_seq, traj, pred_entropy_map_c, pred_entropy_map_s, cam_accum_val = extract_result(opti.debug.value)
                res['success'] = True # 标记为 True 以便继续执行
            else:
                # 3. 其他严重错误 (如 Infeasible)，使用备用策略
                print(f"[UnifiedLocalPlanner] Opt Failed: {e}")
                fallback = self._plan_cruise(current_pose, target, map_dims_m=sliced_map_size_m)
                fallback['status'] = "fallback_cruise"
                fallback['solve_time_sec'] = time.time() - t0
                fallback['steps'] = len(fallback.get('control_sequence', []))
                if DEBUG:
                    print(
                        "[UnifiedLocalPlanner] Fallback to cruise mode after optimization failure. "
                        f"dist={dist_to_target:.2f}m, steps={fallback['steps']}"
                    )
                return fallback
                

        res['control_sequence'] = u_opt_seq
        res['trajectory'] = traj
        res['candidates'] = candidates
        res['solve_time_sec'] = solve_time_sec
        res['steps'] = N


        res['final_entropy_map'] = pred_entropy_map_c
        res['final_entropy_map_substrate'] = pred_entropy_map_s

        
        # --- 可视化 (生成前后对比图) ---
        if DEBUG:
            self._visualize_comparison(current_pose, target, log_odds_c, log_odds_s,
                                    traj, candidates, cam_accum_val if n_candidates>0 else None,
                                    pred_entropy_map_c=pred_entropy_map_c,
                                    pred_entropy_map_s=pred_entropy_map_s,
                                    fls_offset=fls_offset)
            
        return res

    def _visualize_comparison(self, start, goal, log_odds_c, log_odds_s, trajectory, candidates,
                              final_cam_accum, pred_entropy_map_c=None, pred_entropy_map_s=None,
                              fls_offset=None):
        # 2x3 subplots
        plt.ion()
        
        fig = plt.figure(num="LocalPlanner", figsize=(22, 10), clear=True)

        rows_c, cols_c = log_odds_c.shape
        map_width_c = cols_c * self.cell_size
        map_height_c = rows_c * self.cell_size

        rows_s, cols_s = log_odds_s.shape
        map_width_s = cols_s * self.cell_size
        map_height_s = rows_s * self.cell_size

        if fls_offset is None:
            fls_offset = np.array([0.0, 0.0], dtype=np.float32)

        # --- 1. 数据准备 ---
        # Coral probability & entropy
        prob_map_c = 1.0 / (1.0 + np.exp(-log_odds_c))
        eps = 1e-6
        p_safe_c = np.clip(prob_map_c, eps, 1-eps)
        init_entropy_map_c = -p_safe_c * np.log(p_safe_c) - (1-p_safe_c) * np.log(1-p_safe_c)

        if pred_entropy_map_c is None:
            pred_entropy_map_c = init_entropy_map_c.copy()

        # Substrate probability & entropy
        prob_map_s = 1.0 / (1.0 + np.exp(-log_odds_s))
        p_safe_s = np.clip(prob_map_s, eps, 1-eps)
        init_entropy_map_s = -p_safe_s * np.log(p_safe_s) - (1-p_safe_s) * np.log(1-p_safe_s)

        if pred_entropy_map_s is None:
            pred_entropy_map_s = init_entropy_map_s.copy()

        # --- 辅助函数：绘制机器人三角形 ---
        def draw_robot(ax, pose, color='cyan'):
            rx, ry, rtheta = pose
            # 假设车长 0.8m, 宽 0.4m (你可以根据 self.cell_size 调整)
            tri_len = 0.8
            tri_width = 0.4
            triangle_verts = np.array([
                [tri_len/2, 0],              # 鼻尖
                [-tri_len/2, -tri_width/2],  # 左后
                [-tri_len/2, tri_width/2],   # 右后
            ])
            robot_tri = patches.Polygon(triangle_verts, closed=True, 
                                        facecolor=color, edgecolor='black', linewidth=1.5,
                                        label='Robot', zorder=10)
            t = transforms.Affine2D().rotate(rtheta).translate(rx, ry) + ax.transData
            robot_tri.set_transform(t)
            ax.add_patch(robot_tri)

        # ==========================================
        # Row 1: Coral maps (FLC)
        # ==========================================
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(prob_map_c,
                   origin='upper',
                   extent=[0, map_width_c, map_height_c, 0],
                   cmap='gray', vmin=0, vmax=1)
        ax1.scatter(goal[0], goal[1], c='red', marker='*', s=200, edgecolors='white', label='Goal', zorder=5)
        if len(candidates) > 0:
            ax1.scatter(candidates[:, 0], candidates[:, 1], s=100, facecolors='none',
                        edgecolors='cyan', linewidth=2, label='Candidates')
        draw_robot(ax1, start, color='cyan')
        ax1.set_title("1. Coral Probability Map (Initial)", fontsize=12)
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m] (Down)")
        ax1.legend(loc='lower right')

        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(init_entropy_map_c,
                   origin='upper',
                   extent=[0, map_width_c, map_height_c, 0],
                   cmap='gray', vmin=0, vmax=0.7)
        ax2.scatter(goal[0], goal[1], c='red', marker='*', s=100)
        if len(candidates) > 0:
            ax2.scatter(candidates[:, 0], candidates[:, 1], s=100, facecolors='none',
                        edgecolors='cyan', linewidth=2)
        draw_robot(ax2, start, color='cyan')
        ax2.set_title("2. Coral Entropy Map (Initial)", fontsize=12)
        ax2.set_xlabel("X [m]")

        ax3 = plt.subplot(2, 3, 3)
        ax3.imshow(pred_entropy_map_c,
                   origin='upper',
                   extent=[0, map_width_c, map_height_c, 0],
                   cmap='gray', vmin=0, vmax=0.7)

        # FLC sensor + DLC footprint
        vis_step = 2
        for k in range(0, len(trajectory), vis_step):
            px, py, pth = trajectory[k]
            theta_deg = np.rad2deg(pth)
            fov_deg = np.rad2deg(self.fwd_fov_rad)

            wedge = patches.Wedge((px, py), self.fwd_range,
                                  theta_deg - fov_deg/2, theta_deg + fov_deg/2,
                                  color='orange', alpha=0.15)
            ax3.add_patch(wedge)

            rect = patches.Rectangle((-self.cam_L/2, -self.cam_L/2),
                                     self.cam_L, self.cam_L,
                                     linewidth=1, edgecolor='blue', facecolor='none',
                                     linestyle='--', alpha=0.6)
            t_cam = transforms.Affine2D().rotate(pth).translate(px, py) + ax3.transData
            rect.set_transform(t_cam)
            ax3.add_patch(rect)

        ax3.plot(trajectory[:, 0], trajectory[:, 1], 'b.-', linewidth=1.5, label='Path')

        if len(candidates) > 0 and final_cam_accum is not None:
            raw_confs = 1.0 - np.exp(-self.lambda_sat * self.eta_cam * final_cam_accum)
            confs = np.atleast_1d(raw_confs)
            for i, cand in enumerate(candidates):
                c_prob = confs[i] if i < len(confs) else 0
                col = plt.cm.Reds(c_prob)
                ax3.scatter(cand[0], cand[1], s=150, color=col, edgecolors='k', zorder=5)

        ax3.set_title("3. Coral Entropy (Predicted at Plan End, FLC + DLC)", fontsize=12)
        ax3.set_xlabel("X [m]")

        # ==========================================
        # Row 2: Substrate maps (FLS)
        # ==========================================
        ax4 = plt.subplot(2, 3, 4)
        ax4.imshow(prob_map_s,
                   origin='upper',
                   extent=[fls_offset[0], fls_offset[0] + map_width_s, fls_offset[1] + map_height_s, fls_offset[1]],
                   cmap='gray', vmin=0, vmax=1)
        draw_robot(ax4, start, color='cyan')
        ax4.set_title("4. Substrate Probability Map (Initial)", fontsize=12)
        ax4.set_xlabel("X [m]")
        ax4.set_ylabel("Y [m] (Down)")

        ax5 = plt.subplot(2, 3, 5)
        ax5.imshow(init_entropy_map_s,
                   origin='upper',
                   extent=[fls_offset[0], fls_offset[0] + map_width_s, fls_offset[1] + map_height_s, fls_offset[1]],
                   cmap='gray', vmin=0, vmax=0.7)
        draw_robot(ax5, start, color='cyan')
        ax5.set_title("5. Substrate Entropy Map (Initial)", fontsize=12)
        ax5.set_xlabel("X [m]")

        ax6 = plt.subplot(2, 3, 6)
        ax6.imshow(pred_entropy_map_s,
                   origin='upper',
                   extent=[fls_offset[0], fls_offset[0] + map_width_s, fls_offset[1] + map_height_s, fls_offset[1]],
                   cmap='gray', vmin=0, vmax=0.7)

        # FLS sensor
        for k in range(0, len(trajectory), vis_step):
            px, py, pth = trajectory[k]
            theta_deg = np.rad2deg(pth)
            fov_deg = np.rad2deg(self.sonar_fov_rad)
            wedge = patches.Wedge((px, py), self.sonar_range,
                                  theta_deg - fov_deg/2, theta_deg + fov_deg/2,
                                  color='yellow', alpha=0.15)
            ax6.add_patch(wedge)

        ax6.plot(trajectory[:, 0], trajectory[:, 1], 'b.-', linewidth=1.5, label='Path')
        ax6.set_title("6. Substrate Entropy (Predicted at Plan End, FLS)", fontsize=12)
        ax6.set_xlabel("X [m]")

        # 统一设置 Grid
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.grid(True, linestyle='--', alpha=0.3)
            # 不需要 set_ylim 或 invert_yaxis，因为 extent 已经处理了

        plt.tight_layout()
        fig.canvas.draw_idle()
        # plt.show(block=False)
        # plt.pause(0.001)
        # plt.ioff()
        # plt.show()  # 阻塞直到手动关闭

# --- 简单的测试桩 (Main) ---
if __name__ == "__main__":
    planner = UnifiedLocalPlanner()
    
    # --- 1. 读取数据 ---
    # 模拟文件路径，如果文件不存在则生成模拟数据
    # map_dir = "./local_planner_map/map_0/"
    map_dir = "./planner/local_planner_map/map_102/"

    # Load Data
    robot_pose = np.loadtxt(os.path.join(map_dir, "robot_pose_in_slice.txt"))
    target_position = np.loadtxt(os.path.join(map_dir, "target_position_in_slice.txt"))
    
    
    lo_map_c = np.load(os.path.join(map_dir, "logodds_snapshot.npy"))
    r_map = np.load(os.path.join(map_dir, "reward_snapshot.npy"))

    lo_s_path = os.path.join(map_dir, "logodds_s_snapshot.npy")
    lo_map_s = np.load(lo_s_path) if os.path.exists(lo_s_path) else lo_map_c

    conf_path = os.path.join(map_dir, "confirmation_snapshot.npy")
    confirmation_map = r_map > CONF_THRESHOLD
    # conf_path = os.path.join(map_dir, "confirmation_snapshot.npy")
    # confirmation_map = np.load(conf_path)


    fls_origin_path = os.path.join(map_dir, "fls_origin.txt")
    flc_origin_path = os.path.join(map_dir, "flc_origin.txt")
    fls_origin = np.loadtxt(fls_origin_path) if os.path.exists(fls_origin_path) else None
    flc_origin = np.loadtxt(flc_origin_path) if os.path.exists(flc_origin_path) else None

    # visualize_initial_state(robot_pose, target_position, lo_map, r_map, cell_size=CELL_SIZE)

    # --- 2. 规划 ---
    print(f"Planning from {robot_pose} to {target_position}...")
    time_budget = 6.0  # seconds, adjust as needed for debugging
    res = planner.plan(
        time_budget=time_budget,
        current_pose=robot_pose,
        target=target_position,
        log_odds_s=lo_map_s,
        log_odds_c=lo_map_c,
        confirmation_map=confirmation_map,
        fls_origin=fls_origin,
        flc_origin=flc_origin,
    )
    
    print("Optimization Success:", res['success'])
    print("Control Input (vx_body, vy_body, omega):", res['control_sequence'])
