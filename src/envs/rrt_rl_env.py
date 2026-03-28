from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.planners.rrt_3d import RRTPlanner3D, point_in_collision_3d
from src.rewards.reward_fn import RewardConfig, compute_reward
from src.utils.geometry import dist_3d


XYZ = Tuple[float, float, float]


class RRTRLEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        planner_kwargs: Dict[str, Any],
        task_list: List[Dict[str, Any]],
        k_candidates: int = 8,
        reward_cfg: Optional[RewardConfig] = None,
    ):
        super().__init__()

        if len(task_list) == 0:
            raise ValueError("task_list 不能为空")

        self.planner_kwargs = planner_kwargs
        self.task_list = task_list
        self.k_candidates = int(k_candidates)
        self.reward_cfg = reward_cfg or RewardConfig()

        self.planner: Optional[RRTPlanner3D] = None
        self.current_task: Optional[Dict[str, Any]] = None
        self.current_candidates: Optional[np.ndarray] = None
        self.rng: Optional[np.random.Generator] = None

        self.max_iter = int(planner_kwargs.get("max_iter", 12000))
        self.space_diag = self._compute_space_diag(planner_kwargs)

        # 全局特征 5 维
        # 候选点特征 7 维
        self.n_global_feat = 5
        self.n_candidate_feat = 7

        obs_dim = self.n_global_feat + self.k_candidates * self.n_candidate_feat

        # 动作：从 k 个候选点中选一个
        self.action_space = spaces.Discrete(self.k_candidates)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # -------- 候选采样配置（可后续继续调） --------
        # 目标偏置候选数量
        self.n_goal_bias = max(2, self.k_candidates // 4)
        # 树附近局部探索候选数量
        self.n_local_explore = max(2, self.k_candidates // 4)
        # 剩余部分作为全局随机探索
        # self.k_candidates = n_goal_bias + n_local_explore + n_global_random
        self.local_radius = float(self.planner_kwargs.get("step_size", 25.0)) * 3.0
        self.goal_radius = float(self.planner_kwargs.get("step_size", 25.0)) * 2.0

    @staticmethod
    def _compute_space_diag(planner_kwargs: Dict[str, Any]) -> float:
        dx = float(planner_kwargs["x_max"]) - float(planner_kwargs["x_min"])
        dy = float(planner_kwargs["y_max"]) - float(planner_kwargs["y_min"])
        dz = float(planner_kwargs["z_max"]) - float(planner_kwargs["z_min"])
        return max((dx * dx + dy * dy + dz * dz) ** 0.5, 1e-6)

    def _sample_task(self, rng: np.random.Generator) -> Dict[str, Any]:
        idx = int(rng.integers(0, len(self.task_list)))
        return self.task_list[idx]

    def _make_planner(self) -> RRTPlanner3D:
        return RRTPlanner3D(**self.planner_kwargs)

    def _clip_to_bounds(self, p: XYZ) -> XYZ:
        assert self.planner is not None
        x = float(np.clip(p[0], self.planner.x_min, self.planner.x_max))
        y = float(np.clip(p[1], self.planner.y_min, self.planner.y_max))
        z = float(np.clip(p[2], self.planner.z_min, self.planner.z_max))
        return (x, y, z)

    def _sample_free_point(self, rng: np.random.Generator) -> XYZ:
        assert self.planner is not None
        while True:
            x = rng.uniform(self.planner.x_min, self.planner.x_max)
            y = rng.uniform(self.planner.y_min, self.planner.y_max)
            z = rng.uniform(self.planner.z_min, self.planner.z_max)
            p = (float(x), float(y), float(z))
            if not point_in_collision_3d(p, self.planner.obstacles):
                return p

    def _sample_free_point_near(
        self,
        center: XYZ,
        radius: float,
        rng: np.random.Generator,
        max_trials: int = 40,
    ) -> XYZ:
        """
        在 center 附近采样一个自由点；若多次失败则退化为全局自由采样。
        """
        assert self.planner is not None

        radius = max(float(radius), 1e-3)
        for _ in range(max_trials):
            dx = rng.uniform(-radius, radius)
            dy = rng.uniform(-radius, radius)
            dz = rng.uniform(-radius, radius)

            p = self._clip_to_bounds(
                (center[0] + dx, center[1] + dy, center[2] + dz)
            )
            if not point_in_collision_3d(p, self.planner.obstacles):
                return p

        return self._sample_free_point(rng)

    def _sample_goal_biased_candidate(self, rng: np.random.Generator) -> XYZ:
        """
        在 goal 附近采样，而不是直接把 goal 作为候选点。
        这样更稳，也更符合“引导搜索而非硬贴目标”的思路。
        """
        assert self.planner is not None
        assert self.planner.goal_xyz is not None
        return self._sample_free_point_near(
            self.planner.goal_xyz,
            self.goal_radius,
            rng,
        )

    def _sample_local_tree_candidate(self, rng: np.random.Generator) -> XYZ:
        """
        在当前树已有节点附近采样，提高候选质量，让 RL 真正能“选”。
        """
        assert self.planner is not None

        idx = int(rng.integers(0, len(self.planner.nodes)))
        center = self.planner.nodes[idx].xyz
        return self._sample_free_point_near(center, self.local_radius, rng)

    def _sample_k_candidates(self, rng: np.random.Generator) -> np.ndarray:
        """
        混合候选生成：
        1) 目标偏置
        2) 树附近局部探索
        3) 全局随机探索
        """
        candidates: List[XYZ] = []

        n_goal = min(self.n_goal_bias, self.k_candidates)
        n_local = min(self.n_local_explore, self.k_candidates - n_goal)
        n_random = self.k_candidates - n_goal - n_local

        for _ in range(n_goal):
            candidates.append(self._sample_goal_biased_candidate(rng))

        for _ in range(n_local):
            candidates.append(self._sample_local_tree_candidate(rng))

        for _ in range(n_random):
            candidates.append(self._sample_free_point(rng))

        rng.shuffle(candidates)
        return np.asarray(candidates, dtype=np.float32)

    def _candidate_features(self, candidate_xyz: XYZ) -> np.ndarray:
        assert self.planner is not None
        assert self.planner.goal_xyz is not None

        x, y, z = candidate_xyz
        goal_xyz = self.planner.goal_xyz

        nearest_idx = self.planner.nearest_index(candidate_xyz)
        nearest_xyz = self.planner.nodes[nearest_idx].xyz

        dist_goal = dist_3d(candidate_xyz, goal_xyz)
        dist_nearest = dist_3d(candidate_xyz, nearest_xyz)

        steered_xyz = self.planner.steer(nearest_xyz, candidate_xyz)
        expected_best = min(
            self.planner.current_best_dist,
            dist_3d(steered_xyz, goal_xyz),
        )
        expected_delta = self.planner.current_best_dist - expected_best

        min_tree_dist = min(dist_3d(candidate_xyz, node.xyz) for node in self.planner.nodes)

        feat = np.array(
            [
                x / self.space_diag,
                y / self.space_diag,
                z / self.space_diag,
                dist_goal / self.space_diag,
                dist_nearest / self.space_diag,
                expected_delta / self.space_diag,
                min_tree_dist / self.space_diag,
            ],
            dtype=np.float32,
        )
        return feat

    def _build_global_features(self) -> np.ndarray:
        assert self.planner is not None

        state = self.planner.get_search_state()

        iter_ratio = state["iter_count"] / max(self.max_iter, 1)
        tree_ratio = state["tree_size"] / max(self.max_iter, 1)
        best_dist_ratio = state["current_best_dist"] / self.space_diag
        invalid_ratio = state["invalid_ratio"]
        prog_invalid_ratio = state["n_prog_invalid"] / max(state["iter_count"], 1)

        return np.array(
            [
                iter_ratio,
                tree_ratio,
                best_dist_ratio,
                invalid_ratio,
                prog_invalid_ratio,
            ],
            dtype=np.float32,
        )

    def _build_obs(self) -> np.ndarray:
        assert self.current_candidates is not None

        global_feat = self._build_global_features()
        cand_feat = np.concatenate(
            [self._candidate_features(tuple(c)) for c in self.current_candidates],
            axis=0,
        ).astype(np.float32)

        obs = np.concatenate([global_feat, cand_feat], axis=0).astype(np.float32)
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        super().reset(seed=seed)

        # 统一使用同一个 episode 级 RNG
        self.rng = np.random.default_rng(seed)

        self.current_task = self._sample_task(self.rng)
        self.planner = self._make_planner()

        start_xyz = tuple(self.current_task["start"])
        goal_xyz = tuple(self.current_task["goal"])

        episode_seed = int(self.rng.integers(0, 10_000_000))
        self.planner.reset_tree(start_xyz, goal_xyz, episode_seed=episode_seed)

        self.current_candidates = self._sample_k_candidates(self.rng)
        obs = self._build_obs()

        info = {
            "task": self.current_task,
            "planner_state": self.planner.get_search_state(),
        }
        return obs, info

    def step(self, action: int):
        assert self.planner is not None
        assert self.current_candidates is not None
        assert self.rng is not None

        if not self.action_space.contains(action):
            raise ValueError(f"非法动作: {action}")

        selected_xyz = tuple(self.current_candidates[action].tolist())
        step_result = self.planner.extend_once(selected_xyz)

        reward = compute_reward(step_result, self.reward_cfg)

        terminated = bool(step_result["goal_reached"])
        truncated = bool((not terminated) and (self.planner.iter_count >= self.max_iter))

        if not (terminated or truncated):
            self.current_candidates = self._sample_k_candidates(self.rng)

        obs = self._build_obs() if not (terminated or truncated) else self._terminal_obs()

        info = {
            "selected_xyz": selected_xyz,
            "step_result": step_result,
            "planner_state": self.planner.get_search_state(),
        }

        return obs, float(reward), terminated, truncated, info

    def _terminal_obs(self) -> np.ndarray:
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def render(self):
        pass

    def close(self):
        pass