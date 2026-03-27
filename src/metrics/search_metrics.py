from src.utils.geometry import dist_3d


def best_goal_distance(nodes, goal_xyz):
    if not nodes:
        return float("inf")
    return min(dist_3d(node.xyz, goal_xyz) for node in nodes)


def is_expansion_invalid(in_bounds_ok, point_ok, segment_ok):
    """
    扩展无效（candidate expansion invalid）：
    候选新节点 new_xyz 无法作为一次有效扩展加入树。

    判定条件：
    1) new node 越界
    2) new node 本身碰撞
    3) from -> new 的线段碰撞

    注意：
    在当前实现里，sample_free() 已经保证采样点本身在自由空间内，
    因此这里统计的是“扩展失败”，不是“采样失败”。
    """
    return (not in_bounds_ok) or (not point_ok) or (not segment_ok)


def is_increment_invalid(new_xyz, existing_nodes, duplicate_threshold=22.5):
    """
    增量无效（weak increment）：
    新节点虽然可以加入树，但与已有节点过近，说明这一步扩展的新增搜索量很弱。
    """
    for node in existing_nodes:
        if dist_3d(new_xyz, node.xyz) < duplicate_threshold:
            return True
    return False


def is_progress_invalid(prev_best_dist, new_best_dist, min_progress=6.75):
    """
    推进无效（insufficient progress）：
    本轮扩展后，树到目标的当前最优距离改善不足。
    """
    improvement = prev_best_dist - new_best_dist
    return improvement < min_progress


def compute_invalid_ratio(n_exp_invalid, n_inc_invalid, n_prog_invalid, n_iterations):
    if n_iterations <= 0:
        return 0.0
    return float(n_exp_invalid + n_inc_invalid + n_prog_invalid) / float(n_iterations)