import numpy as np
import math
import os

from param import (
    FLS_RANGE,
    FLS_FOV_DEG,
    FLC_RANGE,
    FLC_FOV_DEG,
    DLC_FOOTPRINT,
    fls_true_positive_rate,
    fls_false_positive_rate,
    flc_true_positive_rate,
    flc_false_positive_rate,
)


class CoralMap:
    def __init__(self, map_file=None, map_size=(400, 400), cell_size=0.25):
        """
        Initialize the coral map.

        map_size is (rows, cols) in cells. cell_size is meters per cell.
        """
        self.cell_size = cell_size
        
        if map_file and os.path.exists(map_file):
            print(f"Loading map from {map_file}...")
            self.map = np.load(map_file)  # ground truth map # 0: Sand, 1: Rock, 2: Coral
        else:
            print("Generating random map...")
            # 0: Sand, 1: Rock, 2: Coral
            self.map = np.zeros(map_size, dtype=np.int8)
            # Randomly generate rock cells.
            noise = np.random.rand(*map_size)
            self.map[noise > 0.85] = 1 
            # Randomly generate coral spots on rock.
            coral_spots = np.random.rand(*map_size)
            self.map[(coral_spots > 0.98) & (self.map == 1)] = 2
            
        self.rows, self.cols = self.map.shape
        self.width_meters = self.cols * self.cell_size
        self.height_meters = self.rows * self.cell_size

        # Observation status:
        # 0: unknown, 1: detected by sonar, 2: confirmed by camera/DLC.
        self.status_mask = np.zeros_like(self.map, dtype=np.int8) 
        self.confirmed_count = 0
        self.total_corals = np.sum(self.map == 2)

    def get_substrate_map(self):
        """
        Return the substrate prior map.

        True: rock/hard-substrate cells where coral may exist, map values 1 and 2.
        False: sand cells where coral is impossible, map value 0.
        """
        return (self.map > 0).astype(bool)

    def world_to_cell(self, x, y):
        """Convert world coordinates to map-cell indices."""
        c = int(x / self.cell_size)
        r = int(y / self.cell_size)
        return r, c
    

    def get_sensor_mask(self, robot_x, robot_y, robot_theta, max_dist, fov_angle=None, shape='sector'):
        """Compute a local boolean mask for the sensor footprint."""
        # Search radius needed by the local slice. A rotated square needs extra
        # room for its corners.
        if shape == 'square':
            search_radius = max_dist * 1.415
        else:
            search_radius = max_dist

        # Bounding box that contains the footprint after rotation.
        r_min, c_min = self.world_to_cell(robot_x - search_radius, robot_y - search_radius)
        r_max, c_max = self.world_to_cell(robot_x + search_radius, robot_y + search_radius)
        
        # Clamp to map bounds.
        r_min, r_max = max(0, r_min), min(self.rows, r_max)
        c_min, c_max = max(0, c_min), min(self.cols, c_max)
        
        if r_min >= r_max or c_min >= c_max:
            return None, (0,0,0,0)

        # Local grid coordinates.
        grid_rows = np.arange(r_min, r_max)
        grid_cols = np.arange(c_min, c_max)
        grid_x, grid_y = np.meshgrid(grid_cols * self.cell_size + self.cell_size/2, 
                                     grid_rows * self.cell_size + self.cell_size/2)
        
        # Transform into the robot body frame.
        dx = grid_x - robot_x
        dy = grid_y - robot_y
        
        # Body X forward, Body Y left
        cos_t = math.cos(-robot_theta)
        sin_t = math.sin(-robot_theta)
        
        local_x = dx * cos_t - dy * sin_t
        local_y = dx * sin_t + dy * cos_t
        
        # Footprint mask.
        mask = np.zeros_like(local_x, dtype=bool)
        
        if shape == 'sector': 
            dist_sq = local_x**2 + local_y**2
            angle_mask = np.abs(np.arctan2(local_y, local_x)) <= (fov_angle / 2.0)
            mask = (dist_sq <= max_dist**2) & (local_x > 0) & angle_mask
            
        elif shape == 'square': 
            half_side = max_dist 
            mask = (np.abs(local_x) <= half_side) & (np.abs(local_y) <= half_side)
            
        return mask, (r_min, r_max, c_min, c_max)

    def get_observations(self, robot_x, robot_y, robot_theta):
        """
        Generate sensor observations from the ground-truth map.

        This is a lightweight physical simulation: detections are sampled from
        the true map using distance-dependent sensor probabilities.
        """

        observations = {}

        # Reset transient detection state. Confirmed cells remain persistent.
        self.status_mask[self.status_mask == 1] = 0

        observations['fls'] = {'valid': False}
        observations['flc'] = {'valid': False}
        observations['dlc'] = {'valid': False}

        # --- 1) DLC (Deterministic, down-looking confirmation) ---
        dlc_half = DLC_FOOTPRINT / 2
        dlc_mask, (r1, r2, c1, c2) = self.get_sensor_mask(
            robot_x, robot_y, robot_theta, max_dist=dlc_half, shape='square'
        )
        dlc_bbox = (r1, r2, c1, c2)
        if dlc_mask is not None:
            local_map = self.map[r1:r2, c1:c2]
            local_status = self.status_mask[r1:r2, c1:c2]

            new_corals = dlc_mask & (local_map == 2) & (local_status != 2)
            count = np.sum(new_corals)
            if count > 0:
                self.confirmed_count += count
            # DLC confirmation marks cells as persistent observations.
            update_mask = dlc_mask
            local_status[update_mask] = 2
            self.status_mask[r1:r2, c1:c2] = local_status

            observations['dlc'] = {
                'valid': True,
                'bbox': dlc_bbox,
                'mask': dlc_mask,
                'detected_coral': dlc_mask & (local_map == 2),
                'detected_empty': dlc_mask & (local_map != 2),
            }

        # --- 2) FLS (probabilistic substrate scout) ---
        fls_mask, (r1, r2, c1, c2) = self.get_sensor_mask(
            robot_x,
            robot_y,
            robot_theta,
            max_dist=FLS_RANGE,
            fov_angle=math.radians(FLS_FOV_DEG),
            shape='sector'
        )
        fls_bbox = (r1, r2, c1, c2)
        if fls_mask is not None:
            local_map = self.map[r1:r2, c1:c2]
            local_status = self.status_mask[r1:r2, c1:c2]

            grid_rows = np.arange(r1, r2)
            grid_cols = np.arange(c1, c2)
            gx, gy = np.meshgrid(
                grid_cols * self.cell_size + self.cell_size/2,
                grid_rows * self.cell_size + self.cell_size/2
            )
            dists = np.sqrt((gx - robot_x)**2 + (gy - robot_y)**2)
            dists[~fls_mask] = np.inf
            d_norm = np.clip(dists / FLS_RANGE, 0, 1)

            p_tp = fls_true_positive_rate(d_norm)
            p_fp = fls_false_positive_rate(d_norm)
            rand = np.random.rand(*dists.shape)

            is_rock = (local_map > 0)  # rock or coral
            hit_tp = is_rock & (rand < p_tp)
            hit_fp = (~is_rock) & (rand < p_fp)

            detections = fls_mask & (hit_tp | hit_fp)

            update_mask = detections & (local_status != 2)
            local_status[update_mask] = 1
            self.status_mask[r1:r2, c1:c2] = local_status

            observations['fls'] = {
                'valid': True,
                'bbox': fls_bbox,
                'mask': fls_mask,
                'dists_norm': d_norm,
                'detections': detections,
            }

        # --- 3) FLC (probabilistic target scout) ---
        flc_mask, (r1, r2, c1, c2) = self.get_sensor_mask(
            robot_x,
            robot_y,
            robot_theta,
            max_dist=FLC_RANGE,
            fov_angle=math.radians(FLC_FOV_DEG),
            shape='sector'
        )
        flc_bbox = (r1, r2, c1, c2)
        if flc_mask is not None:
            local_map = self.map[r1:r2, c1:c2]
            local_status = self.status_mask[r1:r2, c1:c2]

            grid_rows = np.arange(r1, r2)
            grid_cols = np.arange(c1, c2)
            gx, gy = np.meshgrid(
                grid_cols * self.cell_size + self.cell_size/2,
                grid_rows * self.cell_size + self.cell_size/2
            )
            dists = np.sqrt((gx - robot_x)**2 + (gy - robot_y)**2)
            dists[~flc_mask] = np.inf
            d_norm = np.clip(dists / FLC_RANGE, 0, 1)

            p_tp = flc_true_positive_rate(d_norm)
            p_fp = flc_false_positive_rate(d_norm)
            rand = np.random.rand(*dists.shape)

            is_coral = (local_map == 2)
            is_sand = (local_map == 0)
            hit_tp = is_coral & (rand < p_tp)
            hit_fp = (~is_coral) & (~is_sand) & (rand < p_fp)

            detections = flc_mask & (hit_tp | hit_fp)

            update_mask = detections & (local_status != 2)
            local_status[update_mask] = 1
            self.status_mask[r1:r2, c1:c2] = local_status

            observations['flc'] = {
                'valid': True,
                'bbox': flc_bbox,
                'mask': flc_mask,
                'dists_norm': d_norm,
                'detections': detections,
            }

        return observations
