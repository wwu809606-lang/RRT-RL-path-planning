from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RewardConfig:
    # 到达目标给明显大额奖励
    goal_reward: float = 50.0

    # 不再给固定“valid step bonus”，避免模型学成“只要不报错就行”
    valid_step_reward: float = 0.0

    # 显著放大进展奖励
    progress_scale: float = 1.0

    # 明显加大无效扩展惩罚
    expansion_invalid_penalty: float = 2.0
    increment_invalid_penalty: float = 1.0
    progress_invalid_penalty: float = 1.5

    # 超时也给负反馈，防止一直拖
    timeout_penalty: float = 3.0


def compute_reward(step_result: Dict[str, Any], cfg: RewardConfig) -> float:
    """
    基于 planner.extend_once() 的输出计算奖励。

    预期 status:
        - expansion_invalid
        - increment_invalid
        - progress_invalid
        - valid
        - done
    """
    status = step_result["status"]
    reward = 0.0

    delta_best_dist = float(step_result.get("delta_best_dist", 0.0))
    positive_progress = max(delta_best_dist, 0.0)

    if status == "expansion_invalid":
        reward -= cfg.expansion_invalid_penalty

    elif status == "increment_invalid":
        reward -= cfg.increment_invalid_penalty

    elif status == "progress_invalid":
        reward -= cfg.progress_invalid_penalty

    elif status == "valid":
        reward += cfg.valid_step_reward
        reward += cfg.progress_scale * positive_progress

    elif status == "done":
        # done 本身不额外奖励，是否成功由 goal_reward 决定
        reward += 0.0

    if step_result.get("goal_reached", False):
        reward += cfg.goal_reward

    if step_result.get("done", False) and (not step_result.get("goal_reached", False)):
        reward -= cfg.timeout_penalty

    return float(reward)