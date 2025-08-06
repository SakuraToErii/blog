---
title: Mujoco Playground G1 参数
published: 2025-08-06
description: '找到 MuJoCo Playground G1 机器人的核心参数配置'
image: ''
tags: ['RL', 'Mujoco']
category: '工作'
draft: false 
lang: 'zh-CN'
---

# Mujoco Playground G1 参数

## 3️⃣ G1 机器人常量配置说明 (G1 Constants)

以下为 `g1` 机器人的核心环境与控制常量定义，源自其仿真配置文件（`constants.py`），用于构建物理场景、传感器映射及关节约束。

---

### 🌍 场景 XML 文件路径

```python
FEET_ONLY_FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain.xml"
FEET_ONLY_ROUGH_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_feetonly_rough_terrain.xml"
# 这行代码的意思是：在基础路径 mjx_env.ROOT_PATH 下，找到名为 locomotion 的子目录，
# 再在该子目录中找到名为 g1 的子目录，最终将这个完整的路径赋值给 ROOT_PATH 变量。
# 这里的 ROOT_PATH 来自 _src 目录下 mjx_env.py 的 ROOT_PATH = epath.Path(__file__).parent，即 _src 目录。
```

- **功能**：定义两种地形下的 MuJoCo 场景描述文件。
- **用途**：
  - `flat_terrain`：平坦地面，用于基础步态训练。
  - `rough_terrain`：带不规则凸起的粗糙地形，用于泛化能力训练。

> 💡 通过 `task_to_xml(task_name)` 函数根据任务名称动态加载对应 XML。

---

### 🦶 足部与手部标记点（Sites）

```python
FEET_SITES = ["left_foot", "right_foot"]          # 足部接触检测点
HAND_SITES = ["left_palm", "right_palm"]          # 手掌接触检测点
```

- 用于在仿真中识别机器人与外界环境的**接触位置**。
- 在感知系统中，这些 sites 常被关联到传感器或 contact 状态判断逻辑。

---

### 🦵 足部几何体（Geoms）定义

```python
LEFT_FEET_GEOMS = ["left_foot"]
RIGHT_FEET_GEOMS = ["right_foot"]
FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS
```

- 指定实际参与碰撞的几何体名称。
- 用于：
  - 避免误检测（如躯干触地）
  - 实现“仅足部接触”行为约束
  - 在奖励函数中判断是否正常着地

---

### 🏗️ 根节点（Root Body）

```python
ROOT_BODY = "torso_link"
```

- 所有局部坐标系（如 `local_linvel`）、加速度、角度观测均以该节点为基准。
- 作为姿态估计与状态转换的核心参考帧。

---

### 📡 传感器名称映射（Sensor Names）

| 传感器类型 | 对应 MuJoCo Sensor 名称 |
|-----------|------------------------|
| 垂直向量传感器 | `upvector` (`GRAVITY_SENSOR`) |
| 全局线速度 | `global_linvel` (`GLOBAL_LINVEL_SENSOR`) |
| 全局角速度 | `global_angvel` (`GLOBAL_ANGVEL_SENSOR`) |
| 局部线速度 | `local_linvel` (`LOCAL_LINVEL_SENSOR`) |
| 加速度计 | `accelerometer` (`ACCELEROMETER_SENSOR`) |
| 陀螺仪 | `gyro` (`GYRO_SENSOR`) |

> ✳️ 所有传感器的输出数据可通过 `mjx` 接口直接读取，作为观测向量的一部分。

---

### 🔧 关节运动范围限制（Joint Limits）

```python
RESTRICTED_JOINT_RANGE = (
    # 左腿：12个关节（6 + 6）
    (-1.57, 1.57),  # hip_yaw_l
    (-0.5, 0.5),   # hip_roll_l
    (-0.7, 0.7),   # hip_pitch_l
    (0, 1.57),     # knee_pitch_l
    (-0.4, 0.4),   # ankle_pitch_l
    (-0.2, 0.2),   # ankle_roll_l

    # 右腿：12个关节
    (-1.57, 1.57),  # hip_yaw_r
    (-0.5, 0.5),   # hip_roll_r
    (-0.7, 0.7),   # hip_pitch_r
    (0, 1.57),     # knee_pitch_r
    (-0.4, 0.4),   # ankle_pitch_r
    (-0.2, 0.2),   # ankle_roll_r

    # 腰部：4个
    (-2.618, 2.618),  # waist_yaw
    (-0.52, 0.52),   # waist_roll
    (-0.52, 0.52),   # waist_pitch

    # 左肩：7个
    (-3.0892, 2.6704),   # shoulder_yaw_l
    (-1.5882, 2.2515),   # shoulder_roll_l
    (-2.618, 2.618),     # shoulder_pitch_l
    (-1.0472, 2.0944),   # elbow_pitch_l
    (-1.97222, 1.97222), # wrist_pitch_l
    (-1.61443, 1.61443), # wrist_roll_l
    (-1.61443, 1.61443), # wrist_yaw_l

    # 右肩：7个
    (-3.0892, 2.6704),   # shoulder_yaw_r
    (-2.2515, 1.5882),   # shoulder_roll_r
    (-2.618, 2.618),     # shoulder_pitch_r
    (-1.0472, 2.0944),   # elbow_pitch_r
    (-1.97222, 1.97222), # wrist_pitch_r
    (-1.61443, 1.61443), # wrist_roll_r
    (-1.61443, 1.61443), # wrist_yaw_r
)
```

- **共 29 个关节**，排列顺序与 `mj_model.jnt_qposadr` 对齐。
- 所有关节施加了**物理合理性与安全限制**，防止关节超限导致模型异常。
- 用于：
  - 动作空间裁剪（action clipping）
  - 奖励函数惩罚过界的关节约束
  - 数据归一化与训练稳定性

> 🔍 特别提醒：左肩与右肩的 `shoulder_roll` 范围不对称（左右不同），体现真实机械结构差异。

---

:::tip[小贴士]
在训练 RL 策略时，可将 `RESTRICTED_JOINT_RANGE` 用于构建动作空间约束；在部署时，也应根据此范围做安全回滚或平滑处理。
:::



## 2️⃣ 标准观测信息 (Standard Observations)
**103 维**


```python
state = jp.hstack([
    # 骨盆局部线性速度（含噪声）
    noisy_linvel,                                        # 3
    # 骨盆角速度（陀螺仪，含噪声）
    noisy_gyro,                                          # 3
    # 骨盆坐标系下的重力向量（含噪声）
    noisy_gravity,                                       # 3
    # 当前运动指令：目标控制指令
    info["command"],                                     # 3
    # 当前关节角度与默认姿态的偏差（含噪声）
    noisy_joint_angles - self._default_pose,             # 29
    # 当前关节角速度（含噪声）
    noisy_joint_vel,                                     # 29
    # 上一时间步的控制动作（用于平滑性奖励）
    info["last_act"],                                    # 29
    # 步态相位信息（cos⁡ϕL, cos⁡ϕR, sin⁡ϕL, sin⁡ϕR）
    phase,                                               # 4
])                                                       # 合计：103 维
```
---


## 3️⃣ 特权观测信息 (Privileged Observations)
**103 + 124 = 227 维**

:::warning[注意]
privileged_state（特权观测）是在仿真环境中可以获取，但在真实机器人上通常难以或无法直接测量的信息。这些信息在训练过程中（尤其是在使用 “teacher-student” 或非对称Actor-Critic架构时）可以帮助Critic网络更准确地评估状态价值，从而加速和稳定训练过程。
:::

```python
privileged_state = jp.hstack([
    # 标准观测状态（基础信息）
    state,                                              # 103    
    # 无噪声陀螺仪数据：骨盆角速度（真值）
    gyro,                                               # 3
    # 无噪声加速度计数据：骨盆加速度（真值）
    accelerometer,                                      # 3
    # 无噪声重力向量：在骨盆坐标系下的精确重力方向
    gravity,                                            # 3
    # 无噪声局部线性速度：骨盆坐标系下的真实速度
    linvel,                                             # 3
    # 全局角速度：世界坐标系下的骨盆旋转速率（真值）
    global_angvel,                                      # 3
    # 当前关节角度与默认姿态的偏差（无噪声）
    joint_angles - self._default_pose,                  # 29
    # 当前关节角速度（无噪声）
    joint_vel,                                          # 29
    # 根节点高度：躯干中心在世界坐标系下的 z 坐标
    root_height,                                        # 1
    # 各关节实际输出的电机力/力矩（真值）
    data.actuator_force,                                # 29
    # 足部接触状态：左右脚是否接触地面（布尔值）
    contact,                                            # 2
    # 足部关键点速度（2个点 × 3维，如脚跟/脚尖全局速度）
    feet_vel,                                           # 6
    # 左右脚离地时间（用于步态周期建模）
    info["feet_air_time"],                              # 2
])                                                      # 合计：216 维


# 其中
foot_linvel_sensor_adr = []
for site in consts.FEET_SITES:
    sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
    sensor_adr = self._mj_model.sensor_adr[sensor_id]
    sensor_dim = self._mj_model.sensor_dim[sensor_id]
    foot_linvel_sensor_adr.append(
        list(range(sensor_adr, sensor_adr + sensor_dim))
    )
self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
```

```xml
# 官方源码的 feet_vel, # 4*3 是错误的
<framelinvel objtype="site" objname="left_foot" name="left_foot_global_linvel"/>
<framelinvel objtype="site" objname="right_foot" name="right_foot_global_linvel"/>
```


