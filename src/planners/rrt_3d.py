import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Point

from src.metrics.search_metrics import (
    compute_invalid_ratio,
    is_expansion_invalid,
    is_increment_invalid,
    is_progress_invalid,
)
from src.utils.geometry import dist_3d, interpolate_segment_3d


XYZ = Tuple[float, float, float]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def point_in_collision_3d(p: XYZ, obstacles) -> bool:
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


def segment_collision_free_3d(p1: XYZ, p2: XYZ, obstacles, resolution: float = 5.0) -> bool:
    samples = interpolate_segment_3d(p1, p2, resolution=resolution)
    for p in samples:
        if point_in_collision_3d(p, obstacles):
            return False
    return True


class Node3D:
    def __init__(self, x: float, y: float, z: float, parent: Optional[int] = None):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.parent = parent

    @property
    def xyz(self) -> XYZ:
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
        step_size: float = 45.0,
        goal_sample_rate: float = 0.06,
        max_iter: int = 12000,
        goal_tolerance: float = 35.0,
        resolution: float = 5.0,
        random_seed: int = 42,
        duplicate_threshold: float = 22.5,
        min_progress: float = 6.75,
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

        self.nodes: List[Node3D] = []

        # episode-level states
        self.start_xyz: Optional[XYZ] = None
        self.goal_xyz: Optional[XYZ] = None

        self.iter_count: int = 0
        self.n_exp_invalid: int = 0
        self.n_inc_invalid: int = 0
        self.n_prog_invalid: int = 0

        self.current_best_dist: float = float("inf")
        self.d_best_trace: List[float] = []
        self.first_path_iter: Optional[int] = None
        self.success: bool = False
        self.goal_node_idx: Optional[int] = None

    def sample_free(self, goal_xyz: XYZ) -> XYZ:
        if random.random() < self.goal_sample_rate:
            return goal_xyz

        while True:
            x = random.uniform(self.x_min, self.x_max)
            y = random.uniform(self.y_min, self.y_max)
            z = random.uniform(self.z_min, self.z_max)
            p = (x, y, z)
            if not point_in_collision_3d(p, self.obstacles):
                return p

    def nearest_index(self, xyz: XYZ) -> int:
        dists = [dist_3d(node.xyz, xyz) for node in self.nodes]
        return int(np.argmin(dists))

    def steer(self, from_xyz: XYZ, to_xyz: XYZ) -> XYZ:
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

    def in_search_bounds(self, p: XYZ) -> bool:
        return (
            self.x_min <= p[0] <= self.x_max
            and self.y_min <= p[1] <= self.y_max
            and self.z_min <= p[2] <= self.z_max
        )

    def reset_tree(
        self,
        start_xyz: XYZ,
        goal_xyz: XYZ,
        episode_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        if episode_seed is not None:
            set_seed(int(episode_seed))
        else:
            set_seed(self.random_seed)

        if point_in_collision_3d(start_xyz, self.obstacles):
            raise ValueError(f"起点位于障碍物内: {start_xyz}")
        if point_in_collision_3d(goal_xyz, self.obstacles):
            raise ValueError(f"终点位于障碍物内: {goal_xyz}")

        self.start_xyz = start_xyz
        self.goal_xyz = goal_xyz

        self.nodes = [Node3D(*start_xyz, parent=None)]

        self.iter_count = 0
        self.n_exp_invalid = 0
        self.n_inc_invalid = 0
        self.n_prog_invalid = 0

        self.current_best_dist = dist_3d(start_xyz, goal_xyz)
        self.d_best_trace = [self.current_best_dist]
        self.first_path_iter = None
        self.success = False
        self.goal_node_idx = None

        return self.get_search_state()

    def get_search_state(self) -> Dict[str, Any]:
        if not self.nodes:
            last_node_xyz = None
            tree_size = 0
        else:
            last_node_xyz = self.nodes[-1].xyz
            tree_size = len(self.nodes)

        invalid_ratio = compute_invalid_ratio(
            self.n_exp_invalid,
            self.n_inc_invalid,
            self.n_prog_invalid,
            max(self.iter_count, 1),
        )

        return {
            "iter_count": self.iter_count,
            "tree_size": tree_size,
            "current_best_dist": self.current_best_dist,
            "invalid_ratio": invalid_ratio,
            "n_exp_invalid": self.n_exp_invalid,
            "n_inc_invalid": self.n_inc_invalid,
            "n_prog_invalid": self.n_prog_invalid,
            "first_path_iter": self.first_path_iter,
            "success": self.success,
            "last_node_xyz": last_node_xyz,
            "goal_xyz": self.goal_xyz,
        }

    def extend_once(self, sample_xyz: XYZ) -> Dict[str, Any]:
        if self.start_xyz is None or self.goal_xyz is None:
            raise RuntimeError("planner 尚未 reset_tree()")

        if self.success:
            return {
                "iter": self.iter_count,
                "sample_xyz": sample_xyz,
                "nearest_idx": None,
                "nearest_xyz": None,
                "new_xyz": None,
                "success_expand": False,
                "status": "done",
                "delta_best_dist": 0.0,
                "goal_reached": True,
                "done": True,
                "tree_size": len(self.nodes),
                "current_best_dist": self.current_best_dist,
                "n_exp_invalid": self.n_exp_invalid,
                "n_inc_invalid": self.n_inc_invalid,
                "n_prog_invalid": self.n_prog_invalid,
                "invalid_ratio": compute_invalid_ratio(
                    self.n_exp_invalid,
                    self.n_inc_invalid,
                    self.n_prog_invalid,
                    max(self.iter_count, 1),
                ),
            }

        self.iter_count += 1

        nearest_idx = self.nearest_index(sample_xyz)
        nearest_node = self.nodes[nearest_idx]

        new_xyz = self.steer(nearest_node.xyz, sample_xyz)

        prev_best_dist = self.current_best_dist

        in_bounds_ok = self.in_search_bounds(new_xyz)
        point_ok = in_bounds_ok and (not point_in_collision_3d(new_xyz, self.obstacles))
        segment_ok = point_ok and segment_collision_free_3d(
            nearest_node.xyz,
            new_xyz,
            self.obstacles,
            resolution=self.resolution,
        )

        if is_expansion_invalid(in_bounds_ok, point_ok, segment_ok):
            self.n_exp_invalid += 1
            self.d_best_trace.append(self.current_best_dist)

            return {
                "iter": self.iter_count,
                "sample_xyz": sample_xyz,
                "nearest_idx": nearest_idx,
                "nearest_xyz": nearest_node.xyz,
                "new_xyz": new_xyz,
                "success_expand": False,
                "status": "expansion_invalid",
                "delta_best_dist": 0.0,
                "goal_reached": False,
                "done": self.iter_count >= self.max_iter,
                "tree_size": len(self.nodes),
                "current_best_dist": self.current_best_dist,
                "n_exp_invalid": self.n_exp_invalid,
                "n_inc_invalid": self.n_inc_invalid,
                "n_prog_invalid": self.n_prog_invalid,
                "invalid_ratio": compute_invalid_ratio(
                    self.n_exp_invalid,
                    self.n_inc_invalid,
                    self.n_prog_invalid,
                    self.iter_count,
                ),
            }

        inc_invalid = is_increment_invalid(
            new_xyz=new_xyz,
            existing_nodes=self.nodes,
            duplicate_threshold=self.duplicate_threshold,
        )

        new_node = Node3D(*new_xyz, parent=nearest_idx)
        self.nodes.append(new_node)

        new_best_dist = min(prev_best_dist, dist_3d(new_node.xyz, self.goal_xyz))
        delta_best_dist = prev_best_dist - new_best_dist

        status = "valid"
        if inc_invalid:
            self.n_inc_invalid += 1
            status = "increment_invalid"
        elif is_progress_invalid(
            prev_best_dist=prev_best_dist,
            new_best_dist=new_best_dist,
            min_progress=self.min_progress,
        ):
            self.n_prog_invalid += 1
            status = "progress_invalid"

        self.current_best_dist = new_best_dist
        goal_reached = False
        done = False

        if dist_3d(new_node.xyz, self.goal_xyz) <= self.goal_tolerance:
            if segment_collision_free_3d(
                new_node.xyz,
                self.goal_xyz,
                self.obstacles,
                resolution=self.resolution,
            ):
                goal_node = Node3D(*self.goal_xyz, parent=len(self.nodes) - 1)
                self.nodes.append(goal_node)

                self.goal_node_idx = len(self.nodes) - 1
                self.current_best_dist = 0.0
                self.d_best_trace.append(self.current_best_dist)

                self.first_path_iter = self.iter_count
                self.success = True
                goal_reached = True
                done = True
            else:
                self.d_best_trace.append(self.current_best_dist)
                done = self.iter_count >= self.max_iter
        else:
            self.d_best_trace.append(self.current_best_dist)
            done = self.iter_count >= self.max_iter

        return {
            "iter": self.iter_count,
            "sample_xyz": sample_xyz,
            "nearest_idx": nearest_idx,
            "nearest_xyz": nearest_node.xyz,
            "new_xyz": new_xyz,
            "success_expand": True,
            "status": status,
            "delta_best_dist": delta_best_dist,
            "goal_reached": goal_reached,
            "done": done,
            "tree_size": len(self.nodes),
            "current_best_dist": self.current_best_dist,
            "n_exp_invalid": self.n_exp_invalid,
            "n_inc_invalid": self.n_inc_invalid,
            "n_prog_invalid": self.n_prog_invalid,
            "invalid_ratio": compute_invalid_ratio(
                self.n_exp_invalid,
                self.n_inc_invalid,
                self.n_prog_invalid,
                self.iter_count,
            ),
        }

    def extract_path(self, goal_idx: int) -> List[XYZ]:
        path = []
        idx = goal_idx
        while idx is not None:
            node = self.nodes[idx]
            path.append(node.xyz)
            idx = node.parent
        path.reverse()
        return path

    def export_tree_edges(self) -> List[Dict[str, List[float]]]:
        edges = []
        for node in self.nodes:
            if node.parent is None:
                continue
            parent = self.nodes[node.parent]
            edges.append(
                {
                    "from": [parent.x, parent.y, parent.z],
                    "to": [node.x, node.y, node.z],
                }
            )
        return edges

    def plan(self, start_xyz: XYZ, goal_xyz: XYZ) -> Dict[str, Any]:
        self.reset_tree(start_xyz, goal_xyz)

        for _ in range(self.max_iter):
            rnd = self.sample_free(goal_xyz)
            step_result = self.extend_once(rnd)

            if step_result["goal_reached"]:
                path_xyz = self.extract_path(self.goal_node_idx)
                return {
                    "success": True,
                    "iterations": self.iter_count,
                    "first_path_iter": self.first_path_iter,
                    "path_xyz": path_xyz,
                    "tree_edges": self.export_tree_edges(),
                    "n_exp_invalid": self.n_exp_invalid,
                    "n_inc_invalid": self.n_inc_invalid,
                    "n_prog_invalid": self.n_prog_invalid,
                    "invalid_ratio": compute_invalid_ratio(
                        self.n_exp_invalid,
                        self.n_inc_invalid,
                        self.n_prog_invalid,
                        self.iter_count,
                    ),
                    "d_best_trace": self.d_best_trace,
                }

            if self.iter_count >= self.max_iter:
                break

        return {
            "success": False,
            "iterations": self.max_iter,
            "first_path_iter": None,
            "path_xyz": [],
            "tree_edges": self.export_tree_edges(),
            "n_exp_invalid": self.n_exp_invalid,
            "n_inc_invalid": self.n_inc_invalid,
            "n_prog_invalid": self.n_prog_invalid,
            "invalid_ratio": compute_invalid_ratio(
                self.n_exp_invalid,
                self.n_inc_invalid,
                self.n_prog_invalid,
                self.max_iter,
            ),
            "d_best_trace": self.d_best_trace,
        }