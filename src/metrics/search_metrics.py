from src.utils.geometry import dist_3d


def best_goal_distance(nodes, goal_xyz):
    if not nodes:
        return float("inf")
    return min(dist_3d(node.xyz, goal_xyz) for node in nodes)


def is_expansion_invalid(in_bounds_ok, point_ok, segment_ok):
    """
    扩展无效：
    1) new node 越界
    2) new node 本身碰撞
    3) 线段碰撞
    """
    return (not in_bounds_ok) or (not point_ok) or (not segment_ok)


def is_increment_invalid(new_xyz, existing_nodes, duplicate_threshold=22.5):
    """
    增量无效：
    新节点虽然成功加入，但与已有节点过近，说明搜索增量很弱。
    """
    for node in existing_nodes:
        if dist_3d(new_xyz, node.xyz) < duplicate_threshold:
            return True
    return False


def is_progress_invalid(prev_best_dist, new_best_dist, min_progress=6.75):
    """
    推进无效：
    当前最优到目标距离改善不足。
    """
    improvement = prev_best_dist - new_best_dist
    return improvement < min_progress


def compute_invalid_ratio(n_exp_invalid, n_inc_invalid, n_prog_invalid, n_iterations):
    if n_iterations <= 0:
        return 0.0
    return float(n_exp_invalid + n_inc_invalid + n_prog_invalid) / float(n_iterations)