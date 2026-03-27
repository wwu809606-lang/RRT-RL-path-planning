import random

import numpy as np
from shapely.geometry import Point

from src.metrics.search_metrics import (
    best_goal_distance,
    compute_invalid_ratio,
    is_expansion_invalid,
    is_increment_invalid,
    is_progress_invalid,
)
from src.utils.geometry import dist_3d, interpolate_segment_3d


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def point_in_collision_3d(p, obstacles):
    """
    p = (x, y, z)
    若点的平面投影落在某建筑底面内，且 z <= 建筑高度，则碰撞
    """
    pt2d = Point(p[0], p[1])
    z = p[2]

    for obs in obstacles:
        if z <= obs["height"] and obs["polygon"].intersects(pt2d):
            return True
    return False


def segment_collision_free_3d(p1, p2, obstacles, resolution=5.0):
    samples = interpolate_segment_3d(p1, p2, resolution=resolution)
    for p in samples:
        if point_in_collision_3d(p, obstacles):
            return False
    return True


class Node3D:
    def __init__(self, x, y, z, parent=None):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.parent = parent

    @property
    def xyz(self):
        return (self.x, self.y, self.z)


class RRTPlanner3D:
    def __init__(
        self,
        obstacles,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
        step_size=45.0,
        goal_sample_rate=0.06,
        max_iter=12000,
        goal_tolerance=35.0,
        resolution=5.0,
        random_seed=42,
        duplicate_threshold=22.5,
        min_progress=6.75,
    ):
        self.obstacles = obstacles

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.z_min = float(z_min)
        self.z_max = float(z_max)

        self.step_size = float(step_size)
        self.goal_sample_rate = float(goal_sample_rate)
        self.max_iter = int(max_iter)
        self.goal_tolerance = float(goal_tolerance)
        self.resolution = float(resolution)
        self.random_seed = int(random_seed)

        self.duplicate_threshold = float(duplicate_threshold)
        self.min_progress = float(min_progress)

        set_seed(self.random_seed)
        self.nodes = []

    def sample_free(self, goal_xyz):
        if random.random() < self.goal_sample_rate:
            return goal_xyz

        while True:
            x = random.uniform(self.x_min, self.x_max)
            y = random.uniform(self.y_min, self.y_max)
            z = random.uniform(self.z_min, self.z_max)
            p = (x, y, z)
            if not point_in_collision_3d(p, self.obstacles):
                return p

    def nearest_index(self, xyz):
        dists = [dist_3d(node.xyz, xyz) for node in self.nodes]
        return int(np.argmin(dists))

    def steer(self, from_xyz, to_xyz):
        dx = to_xyz[0] - from_xyz[0]
        dy = to_xyz[1] - from_xyz[1]
        dz = to_xyz[2] - from_xyz[2]
        d = (dx * dx + dy * dy + dz * dz) ** 0.5

        if d <= self.step_size:
            return (to_xyz[0], to_xyz[1], to_xyz[2])

        ux = dx / d
        uy = dy / d
        uz = dz / d

        return (
            from_xyz[0] + self.step_size * ux,
            from_xyz[1] + self.step_size * uy,
            from_xyz[2] + self.step_size * uz,
        )

    def in_search_bounds(self, p):
        return (
            self.x_min <= p[0] <= self.x_max and
            self.y_min <= p[1] <= self.y_max and
            self.z_min <= p[2] <= self.z_max
        )

    def plan(self, start_xyz, goal_xyz):
        if point_in_collision_3d(start_xyz, self.obstacles):
            raise ValueError(f"起点位于障碍物内: {start_xyz}")
        if point_in_collision_3d(goal_xyz, self.obstacles):
            raise ValueError(f"终点位于障碍物内: {goal_xyz}")

        self.nodes = [Node3D(*start_xyz, parent=None)]

        n_exp_invalid = 0
        n_inc_invalid = 0
        n_prog_invalid = 0

        current_best_dist = dist_3d(start_xyz, goal_xyz)
        d_best_trace = [current_best_dist]
        first_path_iter = None

        for k in range(self.max_iter):
            rnd = self.sample_free(goal_xyz)

            nearest_idx = self.nearest_index(rnd)
            nearest_node = self.nodes[nearest_idx]

            new_xyz = self.steer(nearest_node.xyz, rnd)

            in_bounds_ok = self.in_search_bounds(new_xyz)
            point_ok = in_bounds_ok and (not point_in_collision_3d(new_xyz, self.obstacles))
            segment_ok = point_ok and segment_collision_free_3d(
                nearest_node.xyz, new_xyz, self.obstacles, resolution=self.resolution
            )

            if is_expansion_invalid(in_bounds_ok, point_ok, segment_ok):
                n_exp_invalid += 1
                d_best_trace.append(current_best_dist)
                continue

            inc_invalid = is_increment_invalid(
                new_xyz=new_xyz,
                existing_nodes=self.nodes,
                duplicate_threshold=self.duplicate_threshold,
            )

            new_node = Node3D(*new_xyz, parent=nearest_idx)
            self.nodes.append(new_node)

            new_best_dist = min(current_best_dist, dist_3d(new_node.xyz, goal_xyz))

            if inc_invalid:
                n_inc_invalid += 1
            elif is_progress_invalid(
                prev_best_dist=current_best_dist,
                new_best_dist=new_best_dist,
                min_progress=self.min_progress,
            ):
                n_prog_invalid += 1

            current_best_dist = new_best_dist

            if dist_3d(new_node.xyz, goal_xyz) <= self.goal_tolerance:
                if segment_collision_free_3d(
                    new_node.xyz, goal_xyz, self.obstacles, resolution=self.resolution
                ):
                    goal_node = Node3D(*goal_xyz, parent=len(self.nodes) - 1)
                    self.nodes.append(goal_node)

                    current_best_dist = 0.0
                    d_best_trace.append(current_best_dist)
                    first_path_iter = k + 1

                    path_xyz = self.extract_path(len(self.nodes) - 1)
                    return {
                        "success": True,
                        "iterations": k + 1,
                        "first_path_iter": first_path_iter,
                        "path_xyz": path_xyz,
                        "tree_edges": self.export_tree_edges(),
                        "n_exp_invalid": n_exp_invalid,
                        "n_inc_invalid": n_inc_invalid,
                        "n_prog_invalid": n_prog_invalid,
                        "invalid_ratio": compute_invalid_ratio(
                            n_exp_invalid, n_inc_invalid, n_prog_invalid, k + 1
                        ),
                        "d_best_trace": d_best_trace,
                    }

            d_best_trace.append(current_best_dist)

        return {
            "success": False,
            "iterations": self.max_iter,
            "first_path_iter": None,
            "path_xyz": None,
            "tree_edges": self.export_tree_edges(),
            "n_exp_invalid": n_exp_invalid,
            "n_inc_invalid": n_inc_invalid,
            "n_prog_invalid": n_prog_invalid,
            "invalid_ratio": compute_invalid_ratio(
                n_exp_invalid, n_inc_invalid, n_prog_invalid, self.max_iter
            ),
            "d_best_trace": d_best_trace,
        }

    def extract_path(self, goal_idx):
        path = []
        idx = goal_idx
        while idx is not None:
            node = self.nodes[idx]
            path.append(node.xyz)
            idx = node.parent
        path.reverse()
        return path

    def export_tree_edges(self):
        edges = []
        for node in self.nodes:
            if node.parent is None:
                continue
            parent = self.nodes[node.parent]
            edges.append({
                "from": [parent.x, parent.y, parent.z],
                "to": [node.x, node.y, node.z],
            })
        return edges