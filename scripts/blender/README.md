# Blender 可视化工具

在 Blender 中查看 UAV 路径规划结果的 3D 可视化。

---

## 🚀 使用方法

### 第一步：在服务器上生成 JSON

```bash
cd /home/wuwenjun/uav_path_planning
python3 scripts/vis/plot_compare_real_3d.py --export-blender-json
```

生成的文件位置：`results/vis/*_blender.json`

### 第二步：下载 JSON 到你的电脑

**方法 1：使用 scp（从你本地电脑运行）**

```bash
scp wuwenjun@服务器IP:/home/wuwenjun/uav_path_planning/results/vis/*_blender.json ~/Downloads/
```

**方法 2：使用 FileZilla 等图形界面工具**

- 连接到服务器
- 找到文件：`/home/wuwenjun/uav_path_planning/results/vis/*_blender.json`
- 下载到本地

### 第三步：在 Blender 中导入

1. 打开 Blender
2. 切换到 **Scripting** 工作区
3. 点击 **Open** 按钮，选择 `import_scene_from_json.py`
4. 在脚本中修改 JSON 文件路径：
   ```python
   json_path = "/path/to/your/downloaded/file.json"
   ```
5. 点击 **Run Script**（或按 `Alt+P`）

---

## 📖 文件说明

| 文件 | 说明 |
|------|------|
| `import_scene_from_json.py` | Blender 导入脚本，在 Blender 中运行此脚本来导入场景 |

---

## 💡 场景内容

导入后你会看到：
- **地面**：灰色平面
- **建筑**：灰色挤出体
- **起点**：蓝色球体
- **终点**：红色发光球体
- **Baseline 路径**：深蓝色粗曲线
- **RL 路径**：橙色粗曲线

---

## ❓ 常见问题

**Q: 修改了代码，想重新生成 JSON？**

重新运行第一步的命令即可，会覆盖旧的 JSON 文件。

**Q: 想修改场景样式？**

在 Blender 中选中对象 → Properties → Material → 修改颜色/粗糙度等参数。

**Q: 找不到生成的 JSON 文件？**

检查输出路径：`results/vis/*_blender.json`

---

## 🎯 完整流程

```bash
# 服务器端：生成 JSON
python3 scripts/vis/plot_compare_real_3d.py --export-blender-json

# 本地端：下载 JSON
scp wuwenjun@服务器IP:/home/wuwenjun/uav_path_planning/results/vis/*_blender.json ~/Downloads/

# 本地端：在 Blender 中打开 import_scene_from_json.py 并运行
```
