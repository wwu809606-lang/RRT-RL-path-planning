"""
Blender Python 脚本：从 JSON 导入 UAV 路径规划场景

使用方法：
1. 在 Blender 中运行此脚本：Scripting -> Open -> 选择此文件 -> Run Script
2. 或在 Blender Python Console 中：
   import sys
   sys.path.append('/path/to/scripts/blender')
   import import_scene_from_json as isj
   isj.import_from_json('/path/to/scene_blender.json')
"""

import bpy
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Any


# =========================================================
# 材质设置
# =========================================================
def setup_materials():
    """创建基础材质"""
    materials = {}

    # 地面材质 - 灰色
    mat_ground = bpy.data.materials.new(name="GroundMaterial")
    mat_ground.use_nodes = True
    nodes = mat_ground.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.3, 0.3, 0.3, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.8
    materials["ground"] = mat_ground

    # 建筑材质 - 浅灰色
    mat_building = bpy.data.materials.new(name="BuildingMaterial")
    mat_building.use_nodes = True
    nodes = mat_building.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.85, 0.85, 0.85, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.5
    bsdf.inputs["Metallic"].default_value = 0.1
    materials["building"] = mat_building

    # 起点 - 蓝色
    mat_start = bpy.data.materials.new(name="StartMaterial")
    mat_start.use_nodes = True
    nodes = mat_start.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.29, 0.56, 0.89, 1.0)  # #4A90E2
    bsdf.inputs["Roughness"].default_value = 0.3
    bsdf.inputs["Metallic"].default_value = 0.5
    materials["start"] = mat_start

    # 终点 - 红色
    mat_goal = bpy.data.materials.new(name="GoalMaterial")
    mat_goal.use_nodes = True
    nodes = mat_goal.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.8, 0.1, 0.1, 1.0)  # #cb181d
    bsdf.inputs["Roughness"].default_value = 0.3
    bsdf.inputs["Metallic"].default_value = 0.5
    bsdf.inputs["Emission"].default_value = (0.5, 0.05, 0.05, 1.0)
    materials["goal"] = mat_goal

    # Baseline 路径 - 深蓝色
    mat_base_path = bpy.data.materials.new(name="BaselinePathMaterial")
    mat_base_path.use_nodes = True
    nodes = mat_base_path.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.03, 0.32, 0.61, 1.0)  # #08519c
    bsdf.inputs["Roughness"].default_value = 0.4
    materials["baseline_path"] = mat_base_path

    # RL 路径 - 橙色
    mat_rl_path = bpy.data.materials.new(name="RLPathMaterial")
    mat_rl_path.use_nodes = True
    nodes = mat_rl_path.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.85, 0.37, 0.05, 1.0)  # #d95f0e
    bsdf.inputs["Roughness"].default_value = 0.4
    materials["rl_path"] = mat_rl_path

    # Baseline 无效扩展 - 浅蓝色
    mat_base_invalid = bpy.data.materials.new(name="BaselineInvalidMaterial")
    mat_base_invalid.use_nodes = True
    nodes = mat_base_invalid.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.26, 0.57, 0.77, 1.0)  # #4292c6
    bsdf.inputs["Roughness"].default_value = 0.6
    bsdf.inputs["Alpha"].default_value = 0.5
    mat_base_invalid.blend_method = 'HASHED'
    materials["baseline_invalid"] = mat_base_invalid

    # RL 无效扩展 - 浅橙色
    mat_rl_invalid = bpy.data.materials.new(name="RLInvalidMaterial")
    mat_rl_invalid.use_nodes = True
    nodes = mat_rl_invalid.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.9, 0.33, 0.05, 1.0)  # #e6550d
    bsdf.inputs["Roughness"].default_value = 0.6
    bsdf.inputs["Alpha"].default_value = 0.5
    mat_rl_invalid.blend_method = 'HASHED'
    materials["rl_invalid"] = mat_rl_invalid

    return materials


# =========================================================
# 场景创建
# =========================================================
def clear_scene():
    """清空当前场景（保留默认设置）"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def create_ground(bounds: Dict[str, float], materials: Dict[str, bpy.types.Material]):
    """创建地面"""
    x_size = bounds["x_max"] - bounds["x_min"]
    y_size = bounds["y_max"] - bounds["y_min"]

    bpy.ops.mesh.primitive_plane_add(
        size=1,
        location=((bounds["x_min"] + bounds["x_max"]) / 2,
                  (bounds["y_min"] + bounds["y_max"]) / 2,
                  0)
    )
    plane = bpy.context.active_object
    plane.name = "Ground"
    plane.scale = (x_size / 2, y_size / 2, 1)

    # 添加材质
    if "ground" in materials:
        plane.data.materials.append(materials["ground"])

    return plane


def create_building(coords_data: List, height: float,
                    materials: Dict[str, bpy.types.Material],
                    index: int) -> bpy.types.Object:
    """
    创建建筑挤出体

    Args:
        coords_data: 坐标数据，可能是单个多边形或多重多边形
        height: 建筑高度
        materials: 材质字典
        index: 建筑索引

    Returns:
        Blender 对象
    """
    if isinstance(coords_data[0], (list, tuple)):
        # MultiPolygon: 多个多边形
        # 需要为每个子多边形创建对象并合并
        objects_to_join = []

        for i, sub_coords in enumerate(coords_data):
            obj = _create_single_building(sub_coords, height, materials, f"Building_{index}_{i}")
            objects_to_join.append(obj)

        if len(objects_to_join) > 1:
            # 合并所有子对象
            bpy.context.view_layer.objects.active = objects_to_join[0]
            for obj in objects_to_join[1:]:
                obj.select_set(True)
            bpy.ops.object.join()

            merged_obj = bpy.context.active_object
            merged_obj.name = f"Building_{index}"
            return merged_obj
        else:
            return objects_to_join[0]
    else:
        # 单个 Polygon
        return _create_single_building(coords_data, height, materials, f"Building_{index}")


def _create_single_building(coords: List[Tuple[float, float]], height: float,
                           materials: Dict[str, bpy.types.Material],
                           name: str) -> bpy.types.Object:
    """创建单个建筑多边形"""
    # 创建曲线
    curve_data = bpy.data.curves.new(name=f"{name}_curve", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 2

    spline = curve_data.splines.new('POLY')
    spline.points.add(len(coords) - 1)

    for i, (x, y) in enumerate(coords):
        spline.points[i].co = (x, y, 0, 1)

    # 闭合曲线
    spline.use_cyclic_u = True

    # 创建对象
    curve_obj = bpy.data.objects.new(name=f"{name}_crv", object_data=curve_data)
    bpy.context.collection.objects.link(curve_obj)

    # 转换为网格
    bpy.context.view_layer.objects.active = curve_obj
    bpy.ops.object.select_all(action='DESELECT')
    curve_obj.select_set(True)
    bpy.ops.object.convert(target='MESH')

    # 挤出
    mesh_obj = bpy.context.active_object
    mesh_obj.name = name

    # 添加挤出修饰符
    modifier = mesh_obj.modifiers.new(name="Extrude", type='SOLIDIFY')
    modifier.thickness = height
    modifier.offset = 0

    # 应用修改器
    bpy.ops.object.modifier_apply(modifier="Extrude")

    # 添加材质
    if "building" in materials:
        mesh_obj.data.materials.append(materials["building"])

    return mesh_obj


def create_point_marker(location: Tuple[float, float, float],
                       marker_type: str,
                       materials: Dict[str, bpy.types.Material]) -> bpy.types.Object:
    """创建起终点标记"""
    if marker_type == "start":
        size = 15
        mat_name = "start"
        obj_name = "Start"
    else:  # goal
        size = 20
        mat_name = "goal"
        obj_name = "Goal"

    # 创建球体
    bpy.ops.mesh.primitive_uv_sphere_add(radius=size / 2, location=location)
    marker = bpy.context.active_object
    marker.name = obj_name

    # 添加材质
    if mat_name in materials:
        marker.data.materials.append(materials[mat_name])

    return marker


def create_path(path_coords: List[Tuple[float, float, float]],
               path_type: str,
               materials: Dict[str, bpy.types.Material],
               radius: float = 3.0) -> bpy.types.Object:
    """
    创建路径（使用曲线）

    Args:
        path_coords: 路径坐标列表
        path_type: "baseline" 或 "rl"
        materials: 材质字典
        radius: 路径粗细

    Returns:
        Blender 曲线对象
    """
    if len(path_coords) < 2:
        print(f"[Warning] Path has {len(path_coords)} points, skipping")
        return None

    # 创建曲线
    curve_data = bpy.data.curves.new(name=f"{path_type}_path_curve", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 12
    curve_data.bevel_depth = radius
    curve_data.bevel_resolution = 4

    spline = curve_data.splines.new('POLY')
    spline.points.add(len(path_coords) - 1)

    for i, (x, y, z) in enumerate(path_coords):
        spline.points[i].co = (x, y, z, 1)

    # 创建对象
    curve_obj = bpy.data.objects.new(
        name=f"{path_type.capitalize()}_Path",
        object_data=curve_data
    )
    bpy.context.collection.objects.link(curve_obj)

    # 添加材质
    mat_name = f"{path_type}_path"
    if mat_name in materials:
        curve_obj.data.materials.append(materials[mat_name])

    return curve_obj


def create_invalid_edges(invalid_edges: List[Dict[str, Any]],
                        edge_type: str,
                        materials: Dict[str, bpy.types.Material]) -> bpy.types.Object:
    """
    创建无效扩展边

    Args:
        invalid_edges: 无效边列表
        edge_type: "baseline" 或 "rl"
        materials: 材质字典

    Returns:
        Blender 对象（包含所有边的曲线）
    """
    if len(invalid_edges) == 0:
        print(f"[Info] No invalid edges for {edge_type}")
        return None

    # 创建曲线对象
    curve_data = bpy.data.curves.new(name=f"{edge_type}_invalid_curve", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 2
    curve_data.bevel_depth = 1.0
    curve_data.bevel_resolution = 2

    # 为每条边创建一个 spline
    for edge in invalid_edges:
        p_from = edge.get("from")
        p_to = edge.get("to")

        if p_from is None or p_to is None:
            continue

        spline = curve_data.splines.new('POLY')
        spline.points.add(1)

        spline.points[0].co = (p_from[0], p_from[1], p_from[2], 1)
        spline.points[1].co = (p_to[0], p_to[1], p_to[2], 1)

    # 创建对象
    curve_obj = bpy.data.objects.new(
        name=f"{edge_type.capitalize()}_Invalid_Edges",
        object_data=curve_data
    )
    bpy.context.collection.objects.link(curve_obj)

    # 添加材质
    mat_name = f"{edge_type}_invalid"
    if mat_name in materials:
        curve_obj.data.materials.append(materials[mat_name])

    return curve_obj


def setup_camera(bounds: Dict[str, float]):
    """设置相机位置"""
    # 删除默认相机
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj)

    # 创建新相机
    camera_data = bpy.data.cameras.new("Camera")
    camera_obj = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera_obj)

    # 设置相机位置
    x_center = (bounds["x_min"] + bounds["x_max"]) / 2
    y_center = (bounds["y_min"] + bounds["y_max"]) / 2
    z_max = bounds["z_max"]

    camera_obj.location = (x_center, y_center - (bounds["y_max"] - bounds["y_min"]) * 0.8, z_max * 1.5)
    camera_obj.rotation_euler = (math.radians(60), 0, 0)

    # 设置为活动相机
    bpy.context.scene.camera = camera_obj

    return camera_obj


def setup_lighting(bounds: Dict[str, float]):
    """设置太阳光"""
    # 删除默认灯光
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj)

    # 创建太阳光
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_data.energy = 5.0
    light_data.shadow_soft_size = 0.5

    light_obj = bpy.data.objects.new("Sun", light_data)
    bpy.context.collection.objects.link(light_obj)

    # 设置太阳位置和方向
    x_center = (bounds["x_min"] + bounds["x_max"]) / 2
    y_center = (bounds["y_min"] + bounds["y_max"]) / 2

    light_obj.location = (x_center, y_center, bounds["z_max"] * 2)
    light_obj.rotation_euler = (math.radians(45), 0, math.radians(-45))

    return light_obj


def setup_render_settings():
    """设置渲染参数"""
    # 使用 Eevee 渲染器（快速）
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'

    # 启用阴影
    bpy.context.scene.eevee.use_shadows = True

    # 环境光遮蔽
    bpy.context.scene.eevee.use_gtao = True
    bpy.context.scene.eevee.gtao_distance = 5.0
    bpy.context.scene.eevee.gtao_factor = 0.5

    # 运动模糊
    bpy.context.scene.eevee.use_motion_blur = False

    # 输出设置
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.resolution_percentage = 100


# =========================================================
# 主导入函数
# =========================================================
def import_from_json(json_path: str, clear: bool = True):
    """
    从 JSON 文件导入场景到 Blender

    Args:
        json_path: JSON 文件路径
        clear: 是否清空现有场景
    """
    print(f"[Info] Loading scene from: {json_path}")

    # 读取 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        scene_data = json.load(f)

    print(f"[Info] Scene metadata: {scene_data.get('metadata', {})}")

    # 可选：清空场景
    if clear:
        clear_scene()

    # 设置材质
    print("[Info] Setting up materials...")
    materials = setup_materials()

    # 创建地面
    print("[Info] Creating ground...")
    create_ground(scene_data["bounds"], materials)

    # 创建建筑
    print(f"[Info] Creating {len(scene_data['buildings'])} buildings...")
    for i, building_data in enumerate(scene_data["buildings"]):
        create_building(
            building_data["coordinates"],
            building_data["height"],
            materials,
            i
        )
    print(f"[Info] Created buildings")

    # 创建起点
    print("[Info] Creating start point...")
    start = tuple(scene_data["start"])
    create_point_marker(start, "start", materials)

    # 创建终点
    print("[Info] Creating goal point...")
    goal = tuple(scene_data["goal"])
    create_point_marker(goal, "goal", materials)

    # 创建 Baseline 路径
    print("[Info] Creating baseline path...")
    base_path = scene_data["paths"]["baseline"]["coordinates"]
    create_path(base_path, "baseline", materials)

    # 创建 RL 路径
    print("[Info] Creating RL path...")
    rl_path = scene_data["paths"]["rl"]["coordinates"]
    create_path(rl_path, "rl", materials)

    # 创建 Baseline 无效扩展（可选）
    base_invalid = scene_data["paths"]["baseline"]["invalid_edges"]
    if len(base_invalid) > 0:
        print(f"[Info] Creating {len(base_invalid)} baseline invalid edges...")
        create_invalid_edges(base_invalid, "baseline", materials)

    # 创建 RL 无效扩展（可选）
    rl_invalid = scene_data["paths"]["rl"]["invalid_edges"]
    if len(rl_invalid) > 0:
        print(f"[Info] Creating {len(rl_invalid)} RL invalid edges...")
        create_invalid_edges(rl_invalid, "rl", materials)

    # 设置相机
    print("[Info] Setting up camera...")
    setup_camera(scene_data["bounds"])

    # 设置灯光
    print("[Info] Setting up lighting...")
    setup_lighting(scene_data["bounds"])

    # 设置渲染参数
    print("[Info] Setting up render settings...")
    setup_render_settings()

    # 框选所有对象
    print("[Info] Framing all objects...")
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for region in area.regions:
                if region.type == 'WINDOW':
                    override = {'area': area, 'region': region}
                    bpy.ops.view3d.view_all(override)
                    break

    print(f"[Done] Scene imported successfully!")
    print(f"[Info] Start: {start}")
    print(f"[Info] Goal: {goal}")
    print(f"[Info] Baseline result: {scene_data['paths']['baseline']['result']}")
    print(f"[Info] RL result: {scene_data['paths']['rl']['result']}")
    print(f"[Info] Press F12 to render or use Viewport to navigate")


# =========================================================
# 命令行接口
# =========================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: blender -b -P import_scene_from_json.py -- <json_path>")
        print("Or in Blender: Scripting -> Run Script -> select this file")
        print("\nExample:")
        print("  blender -b -P import_scene_from_json.py -- /path/to/scene_blender.json")
        sys.exit(1)

    # 获取 JSON 路径（Blender 会添加额外参数）
    json_path = sys.argv[-1]
    if not json_path.endswith('.json'):
        print(f"[Error] Expected JSON file, got: {json_path}")
        sys.exit(1)

    import_from_json(json_path)
