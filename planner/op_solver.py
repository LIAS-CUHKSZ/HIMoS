import numpy as np
from numba import jit
import matplotlib.pyplot as plt

'''
Numba JIT  (compiled for speed): evaluation and local search helpers
Also provides NumbaTDOP_Solver.solveOP with the same interface as global_planner_solver.solveOP.
A small SimpleGridMap class and __main__ block exist only for local debugging.
'''


# ---------------------------------------------------------
# Numba JIT helpers
# ---------------------------------------------------------
@jit(nopython=True)
def jit_evaluate(path, dist_matrix, base_rewards, budget, decay_rate):
    n = len(path)
    current_time = 0.0
    total_score = 0.0

    for i in range(n - 1):
        u = path[i]
        v = path[i + 1]
        t = dist_matrix[u, v]
        current_time += t

        if current_time > budget:
            return total_score

        decay = 1.0 - decay_rate * current_time
        if decay < 0.0:
            decay = 0.0

        total_score += base_rewards[v] * decay

    return total_score


@jit(nopython=True)
def jit_2opt_search(path, dist_matrix, base_rewards, budget, decay_rate, window):
    best_path = path.copy()
    best_score = jit_evaluate(best_path, dist_matrix, base_rewards, budget, decay_rate)

    n = len(path)
    improved = True
    loop_count = 0

    while improved and loop_count < 10:
        improved = False
        loop_count += 1

        for i in range(1, n - 2):
            limit_j = min(n, i + window)

            for j in range(i + 1, limit_j):
                new_path = np.empty_like(best_path)
                idx = 0

                for k in range(0, i):
                    new_path[idx] = best_path[k]
                    idx += 1

                for k in range(j - 1, i - 1, -1):
                    new_path[idx] = best_path[k]
                    idx += 1

                for k in range(j, n):
                    new_path[idx] = best_path[k]
                    idx += 1

                new_score = jit_evaluate(new_path, dist_matrix, base_rewards, budget, decay_rate)

                if new_score > best_score + 1e-6:
                    best_score = new_score
                    best_path = new_path
                    improved = True
                    break

            if improved:
                break

    return best_path, best_score


@jit(nopython=True)
def jit_try_insert(path, dist_matrix, base_rewards, budget, decay_rate, candidates):
    current_best_path = path.copy()
    current_best_score = jit_evaluate(current_best_path, dist_matrix, base_rewards, budget, decay_rate)

    for node in candidates:
        is_in_path = False
        for p in current_best_path:
            if p == node:
                is_in_path = True
                break
        if is_in_path:
            continue

        best_pos = -1
        n = len(current_best_path)

        for i in range(1, n + 1):
            test_path = np.empty(n + 1, dtype=np.int64)
            idx = 0
            for k in range(i):
                test_path[idx] = current_best_path[k]
                idx += 1
            test_path[idx] = node
            idx += 1
            for k in range(i, n):
                test_path[idx] = current_best_path[k]
                idx += 1

            score = jit_evaluate(test_path, dist_matrix, base_rewards, budget, decay_rate)

            if score > current_best_score + 1e-6:
                current_best_score = score
                best_pos = i

        if best_pos != -1:
            new_len = len(current_best_path) + 1
            new_path = np.empty(new_len, dtype=np.int64)
            idx = 0
            for k in range(best_pos):
                new_path[idx] = current_best_path[k]
                idx += 1
            new_path[idx] = node
            idx += 1
            for k in range(best_pos, len(current_best_path)):
                new_path[idx] = current_best_path[k]
                idx += 1
            current_best_path = new_path

    return current_best_path, current_best_score


# ---------------------------------------------------------
# OP-only solver interface (matches global_planner_solver.solveOP)
# ---------------------------------------------------------
class NumbaTDOP_Solver:
    @staticmethod
    def _randomized_greedy_construction(dist_matrix, rewards, budget, decay_rate, start_idx, k=2):
        path = [start_idx]
        curr_t = 0.0
        n = len(rewards)
        mask = np.ones(n, dtype=bool)
        mask[start_idx] = False

        while True:
            curr = path[-1]
            dists_from_curr = dist_matrix[curr]

            valid_mask = (mask) & (curr_t + dists_from_curr <= budget)
            if not np.any(valid_mask):
                break

            valid_indices = np.where(valid_mask)[0]

            ts = curr_t + dists_from_curr[valid_indices]
            decays = np.maximum(0.0, 1.0 - decay_rate * ts)
            future_rewards = rewards[valid_indices] * decays
            costs = dists_from_curr[valid_indices]

            heuristics = future_rewards / (costs**2 + 1e-5)
            # heuristics = future_rewards / (costs + 1e-5)
            # heuristics = future_rewards / ((costs + 1.0)**2)


            top_k_local_indices = np.argsort(heuristics)[-k:][::-1]
            best_h = heuristics[top_k_local_indices[0]]
            worst_h = heuristics[top_k_local_indices[-1]]
            close_enough = (best_h - worst_h) <= 0.2 * abs(best_h) + 1e-12
            if close_enough:
                selected_local_idx = np.random.choice(top_k_local_indices)
            else:
                selected_local_idx = top_k_local_indices[0]
            best_node = valid_indices[selected_local_idx]

            path.append(best_node)
            mask[best_node] = False
            curr_t += dists_from_curr[best_node]

        return np.array(path, dtype=np.int64)

    @staticmethod
    def _try_insert_nodes(path, dist_matrix, rewards, budget, decay_rate, top_k=100):
        all_indices = np.argsort(rewards)[::-1]
        candidates = all_indices[:top_k].astype(np.int64)
        new_path, new_score = jit_try_insert(
            path, dist_matrix, rewards, float(budget), float(decay_rate), candidates
        )
        return new_path, new_score

    @staticmethod
    def solveOP(
        coords,
        rewards,
        current_pos,
        budget,
        decay_rate=0.0,
        num_restarts=10,
        ils_iters=50,
        seed=None,
    ):
        if seed is not None:
            np.random.seed(int(seed))
        coords = np.asarray(coords, dtype=np.float64)
        rewards = np.asarray(rewards, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must be (N,2)")
        if rewards.ndim != 1 or rewards.shape[0] != coords.shape[0]:
            raise ValueError("rewards shape must match coords")
        if coords.shape[0] == 0:
            return [tuple(current_pos)]
        

        # 1. 计算 current_pos 到所有网格点的距离
        current_pos_arr = np.asarray(current_pos, dtype=np.float64)
        dists_to_start = np.sqrt(np.sum((coords - current_pos_arr)**2, axis=1))
        
        # 找到最近的那个网格点的索引，作为“逻辑起点”
        start_idx = np.argmin(dists_to_start)
        start_coordinate = coords[start_idx]
        # print(f"Chosen start index: {start_idx}, coordinate: {start_coordinate}, distance to actual start: {dists_to_start[start_idx]:.2f}")

        # 2. 准备数据：直接使用原始 coords，不插入新点
        coords_all = coords
        
        # 3. 复制奖励数组，并将起点奖励设为 0 (防止机器人为了吃起点的分而在原地打转)
        rewards_all = rewards.copy()
        rewards_all[start_idx] = 0.0



        diff = coords_all[:, np.newaxis, :] - coords_all[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1)).astype(np.float64)

        d_mat = dist_matrix
        b_rews = rewards_all
        bdg = float(budget)
        d_rate = float(decay_rate)

        global_best_score = -1.0
        global_best_path = None

        for _ in range(num_restarts):
            current_path = NumbaTDOP_Solver._randomized_greedy_construction(
                d_mat, b_rews, bdg, d_rate, start_idx, k=2
            )
            current_score = jit_evaluate(current_path, d_mat, b_rews, bdg, d_rate)
            best_local_path = current_path
            best_local_score = current_score

            for _ in range(ils_iters):
                temp_path = best_local_path.copy()
                if len(temp_path) > 8:
                    num_remove = np.random.randint(2, 6)
                    keep_mask = np.ones(len(temp_path), dtype=bool)
                    remove_indices = np.random.choice(
                        np.arange(1, len(temp_path)), num_remove, replace=False
                    )
                    keep_mask[remove_indices] = False
                    temp_path = temp_path[keep_mask]

                temp_path, temp_score = jit_2opt_search(
                    temp_path, d_mat, b_rews, bdg, d_rate, window=30
                )

                temp_path, temp_score = NumbaTDOP_Solver._try_insert_nodes(
                    temp_path, d_mat, b_rews, bdg, d_rate, top_k=100
                )

                full_window = len(temp_path)
                temp_path, temp_score = jit_2opt_search(
                    temp_path, d_mat, b_rews, bdg, d_rate, window=full_window
                )

                if temp_score > best_local_score:
                    best_local_score = temp_score
                    best_local_path = temp_path
            # print(f"Local best score: {best_local_score:.2f}")

            if best_local_score > global_best_score:
                global_best_score = best_local_score
                global_best_path = best_local_path

        if global_best_path is None:
            raise ValueError("No valid path found")

        result_path = [] 

        for idx in global_best_path:
            result_path.append(tuple(coords_all[idx]))

        return result_path, global_best_score



# ---------------------------------------------------------
# Debug-only utilities (not part of OP solver interface)
# ---------------------------------------------------------
class SimpleGridMap:
    def __init__(self, grid_size=20, grid_res=20):
        self.grid_size = grid_size
        self.grid_res = grid_res
        self.num_nodes = grid_res * grid_res

    def generate_reward_field(self):
        centers = np.linspace(0.0, self.grid_size, self.grid_res)
        xs, ys = np.meshgrid(centers, centers)
        field = np.zeros((self.grid_res, self.grid_res), dtype=np.float64)

        gaussians = [
            (0.25 * self.grid_size, 0.25 * self.grid_size, 0.10 * self.grid_size, 100.0),
            (0.75 * self.grid_size, 0.30 * self.grid_size, 0.12 * self.grid_size, 90.0),
            (0.60 * self.grid_size, 0.75 * self.grid_size, 0.15 * self.grid_size, 100.0),
            (0.10 * self.grid_size, 0.90 * self.grid_size, 0.08 * self.grid_size, 60.0),
            (0.30 * self.grid_size, 0.70 * self.grid_size, 0.08 * self.grid_size, 60.0),
        ]

        for cx, cy, sigma, amp in gaussians:
            dx = xs - cx
            dy = ys - cy
            field += amp * np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))

        field -= field.min()
        if field.max() > 0:
            field = 100.0 * field / field.max()
        return field.ravel()

    def to_op_inputs(self):
        centers = np.linspace(0.5, self.grid_size-0.5, self.grid_res)
        xs, ys = np.meshgrid(centers, centers)
        coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
        rewards = self.generate_reward_field().astype(np.float64)
        current_pos = (self.grid_size * 0.5, self.grid_size * 0.5)
        return coords, rewards, current_pos


if __name__ == "__main__":
    grid = SimpleGridMap(grid_size=20, grid_res=20)
    coords, rewards, current_pos = grid.to_op_inputs()
    random_seed = np.random.randint(0, 10000)
    print(f"Using random seed: {random_seed}") # 2416

    path, score = NumbaTDOP_Solver.solveOP(
        coords,
        rewards,
        current_pos,
        budget=100.0,
        decay_rate=0,
        num_restarts=10,
        ils_iters=50,
        seed=random_seed,
    )
    print(f"Getted score: {score:.2f}")

    reward_grid = rewards.reshape(grid.grid_res, grid.grid_res)
    path_arr = np.asarray(path, dtype=np.float64)
    plt.figure(figsize=(8, 8))
    plt.imshow(
        reward_grid,
        origin="lower",
        extent=(0, grid.grid_size, 0, grid.grid_size),
        cmap="viridis",
    )
    plt.plot(path_arr[:, 0], path_arr[:, 1], color="#2ca02c", linewidth=2.5, label="Best")
    plt.scatter(path_arr[0, 0], path_arr[0, 1], s=50, color="#d62728", label="Start")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Optimized Path")
    plt.colorbar(label="Reward")
    plt.legend()
    plt.tight_layout()
    plt.show()
