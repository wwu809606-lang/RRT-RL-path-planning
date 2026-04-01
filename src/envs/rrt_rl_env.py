"""
RRT-RL Environment: 强化学习训练的RRT路径规划环境
=================================================

该模块实现了一个Gymnasium环境，用于训练强化学习智能体来改进RRT（快速随机树）路径规划算法。
环境在每个时间步提供k个候选点，让智能体选择最优的一个进行扩展。

Main Components:
- RRTRLEnv: 主环境类，实现Gymnasium接口
- 候选点生成策略：目标偏置、局部探索、全局随机
- 特征提取：全局特征 + 候选点特征
"""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.planners.rrt_3d import RRTPlanner3D, point_in_collision_3d
from src.rewards.reward_fn import RewardConfig, compute_reward
from src.utils.geometry import dist_3d


# Type alias for 3D coordinates
XYZ = Tuple[float, float, float]


class RRTRLEnv(gym.Env):
    """
    RRT强化学习环境

    该环境将RRT算法的节点扩展过程建模为马尔可夫决策过程(MDP)：
    - State: 当前搜索状态 + k个候选点及其特征
    - Action: 从k个候选点中选择一个进行扩展
    - Reward: 基于扩展结果（是否成功、进度、有效性等）

    Attributes:
        k_candidates: 每个时间步的候选点数量
        n_global_feat: 全局特征维度（5维）
        n_candidate_feat: 单个候选点特征维度（7维）
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        planner_kwargs: Dict[str, Any],
        task_list: List[Dict[str, Any]],
        k_candidates: int = 8,
        reward_cfg: Optional[RewardConfig] = None,
    ):
        """
        初始化RRT-RL环境

        Args:
            planner_kwargs: 传递给RRTPlanner3D的参数字典
            task_list: 任务列表，每个任务包含start和goal坐标及障碍物
            k_candidates: 每个时间步生成的候选点数量
            reward_cfg: 奖励配置，如果为None则使用默认配置

        Raises:
            ValueError: 如果task_list为空
        """
        super().__init__()

        # 验证任务列表非空
        if len(task_list) == 0:
            raise ValueError("task_list 不能为空")

        # 保存环境配置参数
        self.planner_kwargs = planner_kwargs
        self.task_list = task_list
        self.k_candidates = int(k_candidates)
        self.reward_cfg = reward_cfg or RewardConfig()

        # 状态变量（在reset时初始化）
        self.planner: Optional[RRTPlanner3D] = None  # RRT规划器实例
        self.current_task: Optional[Dict[str, Any]] = None  # 当前任务
        self.current_candidates: Optional[np.ndarray] = None  # 当前候选点集
        self.rng: Optional[np.random.Generator] = None  # 随机数生成器

        # 从planner参数中提取最大迭代次数
        self.max_iter = int(planner_kwargs.get("max_iter", 12000))
        # 计算空间对角线长度，用于特征归一化
        self.space_diag = self._compute_space_diag(planner_kwargs)

        # -------- 特征维度配置 --------
        # 全局特征（5维）：迭代进度、树大小、最佳距离、无效比例、近期无效比例
        self.n_global_feat = 5
        # 候选点特征（7维）：坐标(3)、到目标距离、到最近树节点距离、预期改进、到树的最小距离
        self.n_candidate_feat = 7

        # 计算观测空间总维度
        obs_dim = self.n_global_feat + self.k_candidates * self.n_candidate_feat

        # 定义动作空间：从k个候选点中选择一个
        self.action_space = spaces.Discrete(self.k_candidates)
        # 定义观测空间：全局特征 + 所有候选点特征
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # -------- 候选采样配置 --------
        # 这些参数控制候选点的生成策略，可调整以平衡探索与利用

        # 目标偏置候选数量：在目标附近采样，引导向目标搜索
        self.n_goal_bias = max(2, self.k_candidates // 4)
        # 树附近局部探索候选数量：在当前树节点附近采样，进行局部精细搜索
        self.n_local_explore = max(2, self.k_candidates // 4)
        # 剩余部分自动作为全局随机探索候选
        # self.k_candidates = n_goal_bias + n_local_explore + n_global_random

        # 局部探索半径：在树节点附近采样的范围
        self.local_radius = float(self.planner_kwargs.get("step_size", 25.0)) * 3.0
        # 目标附近采样半径
        self.goal_radius = float(self.planner_kwargs.get("step_size", 25.0)) * 2.0

    @staticmethod
    def _compute_space_diag(planner_kwargs: Dict[str, Any]) -> float:
        """
        计算搜索空间的对角线长度

        用于归一化特征，确保不同尺度的环境具有相似的特征分布。

        Args:
            planner_kwargs: 包含空间边界参数的字典

        Returns:
            空间对角线长度
        """
        dx = float(planner_kwargs["x_max"]) - float(planner_kwargs["x_min"])
        dy = float(planner_kwargs["y_max"]) - float(planner_kwargs["y_min"])
        dz = float(planner_kwargs["z_max"]) - float(planner_kwargs["z_min"])
        return max((dx * dx + dy * dy + dz * dz) ** 0.5, 1e-6)

    def _sample_task(self, rng: np.random.Generator) -> Dict[str, Any]:
        """
        从任务列表中随机选择一个任务

        Args:
            rng: 随机数生成器

        Returns:
            选中的任务字典
        """
        idx = int(rng.integers(0, len(self.task_list)))
        return self.task_list[idx]

    def _make_planner(self) -> RRTPlanner3D:
        """创建新的RRT规划器实例"""
        return RRTPlanner3D(**self.planner_kwargs)

    def _clip_to_bounds(self, p: XYZ) -> XYZ:
        """
        将点裁剪到搜索空间边界内

        Args:
            p: 输入点坐标 (x, y, z)

        Returns:
            裁剪后的点坐标
        """
        assert self.planner is not None
        x = float(np.clip(p[0], self.planner.x_min, self.planner.x_max))
        y = float(np.clip(p[1], self.planner.y_min, self.planner.y_max))
        z = float(np.clip(p[2], self.planner.z_min, self.planner.z_max))
        return (x, y, z)

    def _sample_free_point(self, rng: np.random.Generator) -> XYZ:
        """
        全局随机采样一个无碰撞点

        持续采样直到找到一个不在障碍物中的点。

        Args:
            rng: 随机数生成器

        Returns:
            无碰撞点的坐标
        """
        assert self.planner is not None
        while True:
            # 在边界内均匀采样
            x = rng.uniform(self.planner.x_min, self.planner.x_max)
            y = rng.uniform(self.planner.y_min, self.planner.y_max)
            z = rng.uniform(self.planner.z_min, self.planner.z_max)
            p = (float(x), float(y), float(z))
            # 检查是否与障碍物碰撞
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
        在指定中心附近采样一个无碰撞点

        在center周围radius范围内采样，如果多次尝试都失败则退化为全局采样。
        这用于在目标附近或树节点附近生成候选点。

        Args:
            center: 采样中心点
            radius: 采样半径
            rng: 随机数生成器
            max_trials: 局部采样最大尝试次数

        Returns:
            无碰撞点的坐标
        """
        assert self.planner is not None

        radius = max(float(radius), 1e-3)
        # 尝试在指定半径内采样
        for _ in range(max_trials):
            dx = rng.uniform(-radius, radius)
            dy = rng.uniform(-radius, radius)
            dz = rng.uniform(-radius, radius)

            # 生成候选点并裁剪到边界
            p = self._clip_to_bounds(
                (center[0] + dx, center[1] + dy, center[2] + dz)
            )
            if not point_in_collision_3d(p, self.planner.obstacles):
                return p

        # 如果局部采样失败，退化为全局采样
        return self._sample_free_point(rng)

    def _sample_goal_biased_candidate(self, rng: np.random.Generator) -> XYZ:
        """
        生成目标偏置候选点

        在目标附近采样，而不是直接把目标作为候选点。
        这种方法更稳定，也符合"引导搜索而非硬贴目标"的设计思路。

        Args:
            rng: 随机数生成器

        Returns:
            目标附近的候选点坐标
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
        生成局部树探索候选点

        在当前树已有节点附近采样，提高候选质量。
        这样可以让RL智能体真正进行"选择"而非盲目选择。

        Args:
            rng: 随机数生成器

        Returns:
            树节点附近的候选点坐标
        """
        assert self.planner is not None

        # 随机选择一个树节点
        idx = int(rng.integers(0, len(self.planner.nodes)))
        center = self.planner.nodes[idx].xyz
        # 在该节点附近采样
        return self._sample_free_point_near(center, self.local_radius, rng)

    def _sample_k_candidates(self, rng: np.random.Generator) -> np.ndarray:
        """
        混合策略生成k个候选点

        候选点生成策略：
        1) 目标偏置（n_goal_bias个）：在目标附近采样，引导向目标搜索
        2) 树附近局部探索（n_local_explore个）：在树节点附近采样，局部精细搜索
        3) 全局随机探索（剩余）：全局采样，保持探索性

        Args:
            rng: 随机数生成器

        Returns:
            形状为(k, 3)的候选点数组
        """
        candidates: List[XYZ] = []

        # 计算各类候选点的数量
        n_goal = min(self.n_goal_bias, self.k_candidates)
        n_local = min(self.n_local_explore, self.k_candidates - n_goal)
        n_random = self.k_candidates - n_goal - n_local

        # 生成目标偏置候选点
        for _ in range(n_goal):
            candidates.append(self._sample_goal_biased_candidate(rng))

        # 生成局部探索候选点
        for _ in range(n_local):
            candidates.append(self._sample_local_tree_candidate(rng))

        # 生成全局随机候选点
        for _ in range(n_random):
            candidates.append(self._sample_free_point(rng))

        # 打乱候选点顺序，避免位置偏差
        rng.shuffle(candidates)
        return np.asarray(candidates, dtype=np.float32)

    def _candidate_features(self, candidate_xyz: XYZ) -> np.ndarray:
        """
        提取单个候选点的特征向量

        特征包括（共7维）：
        1. x, y, z: 归一化的空间坐标
        2. dist_goal: 到目标的距离
        3. dist_nearest: 到最近树节点的距离
        4. expected_delta: 选择该点后预期最佳距离的改进量
        5. min_tree_dist: 到树中任意节点的最小距离

        Args:
            candidate_xyz: 候选点坐标

        Returns:
            7维特征向量
        """
        assert self.planner is not None
        assert self.planner.goal_xyz is not None

        x, y, z = candidate_xyz
        goal_xyz = self.planner.goal_xyz

        # 找到最近的树节点
        nearest_idx = self.planner.nearest_index(candidate_xyz)
        nearest_xyz = self.planner.nodes[nearest_idx].xyz

        # 计算距离特征
        dist_goal = dist_3d(candidate_xyz, goal_xyz)
        dist_nearest = dist_3d(candidate_xyz, nearest_xyz)

        # 模拟steer操作，计算预期的改进
        steered_xyz = self.planner.steer(nearest_xyz, candidate_xyz)
        expected_best = min(
            self.planner.current_best_dist,
            dist_3d(steered_xyz, goal_xyz),
        )
        expected_delta = self.planner.current_best_dist - expected_best

        # 计算到树中所有节点的最小距离
        min_tree_dist = min(dist_3d(candidate_xyz, node.xyz) for node in self.planner.nodes)

        # 归一化并返回特征向量
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
        """
        构建全局搜索状态特征

        特征包括（共5维）：
        1. iter_ratio: 当前迭代次数 / 最大迭代次数
        2. tree_ratio: 树的大小 / 最大迭代次数
        3. best_dist_ratio: 当前最佳距离 / 空间对角线
        4. invalid_ratio: 历史无效扩展比例
        5. prog_invalid_ratio: 近期无效扩展比例

        Returns:
            5维全局特征向量
        """
        assert self.planner is not None

        state = self.planner.get_search_state()

        # 计算各类归一化特征
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
        """
        构建完整的观测向量

        观测 = 全局特征(5维) + k个候选点特征(每个7维)

        Returns:
            完整的观测向量
        """
        assert self.current_candidates is not None

        # 提取全局特征
        global_feat = self._build_global_features()
        # 提取所有候选点的特征并拼接
        cand_feat = np.concatenate(
            [self._candidate_features(tuple(c)) for c in self.current_candidates],
            axis=0,
        ).astype(np.float32)

        # 拼接全局特征和候选点特征
        obs = np.concatenate([global_feat, cand_feat], axis=0).astype(np.float32)
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """
        重置环境到初始状态

        随机选择一个任务，初始化RRT规划器，生成第一批候选点。

        Args:
            seed: 随机种子
            options: 额外选项（当前未使用）

        Returns:
            obs: 初始观测向量
            info: 包含任务和规划器状态的字典
        """
        super().reset(seed=seed)

        # 创建episode级别的随机数生成器
        self.rng = np.random.default_rng(seed)

        # 随机选择一个任务并创建规划器
        self.current_task = self._sample_task(self.rng)
        self.planner = self._make_planner()

        # 提取起点和终点
        start_xyz = tuple(self.current_task["start"])
        goal_xyz = tuple(self.current_task["goal"])

        # 重置规划器树结构
        episode_seed = int(self.rng.integers(0, 10_000_000))
        self.planner.reset_tree(start_xyz, goal_xyz, episode_seed=episode_seed)

        # 生成第一批候选点并构建观测
        self.current_candidates = self._sample_k_candidates(self.rng)
        obs = self._build_obs()

        # 返回初始信息
        info = {
            "task": self.current_task,
            "planner_state": self.planner.get_search_state(),
        }
        return obs, info

    def step(self, action: int):
        """
        执行一步动作

        根据选择的候选点扩展RRT树，计算奖励，并更新状态。

        Args:
            action: 选择的候选点索引 [0, k_candidates)

        Returns:
            obs: 新的观测向量
            reward: 奖励值
            terminated: 是否到达目标
            truncated: 是否超过最大迭代次数
            info: 包含步骤细节的字典

        Raises:
            ValueError: 如果动作不在有效范围内
        """
        assert self.planner is not None
        assert self.current_candidates is not None
        assert self.rng is not None

        # 验证动作有效性
        if not self.action_space.contains(action):
            raise ValueError(f"非法动作: {action}")

        # 获取选中的候选点
        selected_xyz = tuple(self.current_candidates[action].tolist())
        # 执行一次RRT扩展
        step_result = self.planner.extend_once(selected_xyz)

        # 计算奖励
        reward = compute_reward(step_result, self.reward_cfg)

        # 判断是否终止或截断
        terminated = bool(step_result["goal_reached"])
        truncated = bool((not terminated) and (self.planner.iter_count >= self.max_iter))

        # 如果未终止，生成新的候选点
        if not (terminated or truncated):
            self.current_candidates = self._sample_k_candidates(self.rng)

        # 构建观测（终止时返回零观测）
        obs = self._build_obs() if not (terminated or truncated) else self._terminal_obs()

        # 收集步骤信息
        info = {
            "selected_xyz": selected_xyz,
            "step_result": step_result,
            "planner_state": self.planner.get_search_state(),
        }

        return obs, float(reward), terminated, truncated, info

    def _terminal_obs(self) -> np.ndarray:
        """
        返回终止状态的观测

        终止时返回全零向量，避免对已终止状态进行特征计算。

        Returns:
            全零观测向量
        """
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def render(self):
        """
        渲染环境（当前未实现）
        """
        pass

    def close(self):
        """
        关闭环境并释放资源（当前无资源需要释放）
        """
        pass
