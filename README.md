# ClassicalDetecting

使用传统视觉算法的目标追踪作业仓库。

以下是作业要求原文：
```
要求用自己编写的相关（Correlation）运算算法，利用模版（目标）对途中的物体进行匹配，最终实现对视频中的目标的稳定、连续跟踪，识别的准确率、识别的位置精度、漏检率等指标将作为评价的依据。要求不可以使用任何AI算法进行目标识别、判别或分类。
要求提交所编写的相关运算算法的源代码。
要求不单纯依赖相关算法进行目标的匹配，可以利用多个模版（不同视角下的目标图像）、或目标位置预测算法避免误检测、漏检测带来的目标位置的偏差。
要求将目标识别出后，在视频中实现对目标的框选，稳定跟踪，如案例视频所示（无人机目标识别跟踪案例.mp4）.
1. 动画表情视频：对图像中的动画表情人物进行识别并稳定跟踪，输出表情人物中心像素点位置，绘制出表情人物的运动轨迹（已提供三个不同的模版供选择）
2. 大疆无人机航拍视频：识别图像中的指定车辆目标，要求不能错误识别到其他车辆物体。车辆在“大疆无人机航拍视频目标”中已经用蓝色圆圈标识出，要求识别出车辆在图中的运动全过程直至目标消失，并画出轨迹。
3. 大疆无人机航拍骑车人：识别图像中的指定骑车人，要求不能错误识别到其他物体。车辆在“大疆无人机航拍骑车人目标”中已经用黄色圆圈标识出，要求识别出骑车人在图中的运动全过程直至其消失，并画出轨迹。
```
解法思路为先在ncc的基础上利用NCC与相位相关实现一组baseline；在baseline的基础上，利用SIFT与Kalman滤波，对任务二、任务三添加基于路面检测的运动约束，摆脱对硬编码参数的依赖。

由于task2的视频为夜间道路，针对其明显的像素亮度差异，对该问题的道路识别有两解，效果请参照下列示例输出。但需要注意，因ffmpeg在各平台的Otsu算法实现原理不同，检测结果会有偏差。若想完美复现，建议参考实验环境为Ubuntu 26.04 LTS，Python 3.14。

## 结构

```
src/
  baseline/       # 原始基线（纯 NCC + 相位相关）
  improved/       # 基于基线的改进
  classical/      # 共享库（NCC、Kalman、场景运动、边缘匹配）
scripts/          # 辅助工具
```

## 运行

项目使用uv进行依赖管理。克隆仓库后，使用`uv sync`即可创建该项目运行所需的虚拟环境。`uv`的使用须知见[官方wiki](https://docs.astral.sh/uv/)。

以运行改进后的task 3为例：

```bash
uv run python src/improved/task3_auto_road_tracker.py
```

输出到 `output/improved/<task_name>/`。

## 结果

### Task 2 — 夜间道路车辆追踪

| 版本 | 视频 |
|------|------|
| baseline | [example-outputs/baseline/task2/tracked_result.mp4](example-outputs/baseline/task2/tracked_result.mp4) |
| Otsu + 遮挡 | [example-outputs/improved/task2_otsu/tracked_result.mp4](example-outputs/improved/task2_otsu/tracked_result.mp4) |
| 路面约束 NCC | [example-outputs/improved/task2_constrained/tracked_result.mp4](example-outputs/improved/task2_constrained/tracked_result.mp4) |

### Task 3 — 白天骑车人追踪

| 版本 | 视频 |
|------|------|
| baseline | [example-outputs/baseline/task3/tracked_result.mp4](example-outputs/baseline/task3/tracked_result.mp4) |
| 自动路检 + SIFT | [example-outputs/improved/task3_auto_road/tracked_result.mp4](example-outputs/improved/task3_auto_road/tracked_result.mp4) |

### Task 1 — 动漫模板追踪

| 版本 | 视频 |
|------|------|
| baseline | [example-outputs/baseline/task1_result.mp4](example-outputs/baseline/task1_result.mp4) |
| 边缘 + Kalman | [example-outputs/improved/task1_result.mp4](example-outputs/improved/task1_result.mp4) |

## 方法

| 脚本 | 核心方法 |
|------|----------|
| `improved/task1_edge_kalman.py` | NCC + 边缘倒角距离 + Kalman 平滑 |
| `improved/task2_otsu_tracker.py` | Otsu 路检 + 道路坐标 + 遮挡状态机 |
| `improved/task2_constrained_tracker.py` | 路面蒙版约束 NCC + 统一 search_target() |
| `improved/task3_auto_road_tracker.py` | 车辆引导自动路检 + SIFT 蒙版跟踪 + 统一搜索 |

## 共享库

| 模块 | 功能 |
|------|------|
| `classical/ncc.py` | 手写快速 NCC（积分图 + filter2D） |
| `classical/motion.py` | Kalman 恒速滤波器 |
| `classical/scene_motion.py` | SIFT + 单应矩阵前景估计 |
| `classical/edge_match.py` | 边缘倒角距离匹配 |
