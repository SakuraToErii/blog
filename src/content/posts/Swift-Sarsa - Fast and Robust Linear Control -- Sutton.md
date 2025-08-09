---
title: Swift-Sarsa - Fast and Robust Linear Control -- Sutton
published: 2025-08-07
description: '还没想好'
image: ''
tags: ['RL']
category: '论文'
draft: false 
lang: 'zh-CN'
---

# Swift-Sarsa - Fast and Robust Linear Control -- Sutton

论文标题：Swift-Sarsa: Fast and Robust Linear Control 

论文链接：[https://arxiv.org/abs/2507.19539v1](https://arxiv.org/abs/2507.19539v1)

### 📄 PDF 预览   

:::pdf
Swift-Sarsa.pdf
:::    

 
**Swift-Sarsa** 是在 2024 年 Sutton 提出的 SwiftTD 的基础上，结合 True Online Sarsa (λ) 的策略学习框架所提出的强化学习算法。

## 1️⃣ 控制问题定义 (Control Problem Definition) 

本论文的控制问题由<b>观测（Observations）</b>与<b>动作（Actions）</b>构成。智能体在每一个时间步 $t$ 接收到一个观测向量 $x_t \in \mathbb{R}^n$，并输出一个动作向量 $a_t \in \mathbb{R}^d$。

### 观测向量结构 (Observation Vector Structure)

观测向量 $x_t$ 包含一个特殊的分量，即**奖励值** $r_t$。该奖励所在的分量索引在整个智能体生命周期中保持不变，用于在训练过程中提供即时反馈。

> ✅ **关键说明**：
> 虽然 $r_t$ 在观测向量中被编码，但其作用与标准的奖励信号一致 —— 它仅用于训练目标函数（如强化学习中的回报计算），并不参与策略网络的输入特征处理。
> 
> 该设计确保了奖励信号在训练过程中可被稳定访问，同时避免了因动态索引带来的系统复杂性。

:::note[建议]
在实现中，应将奖励分量的索引（如 `reward_idx`）作为常量定义，并在所有模块中统一使用，以避免运行时错误。
:::

**生命周期平均奖励（Lifetime Average Reward）**
$$
\text{Lifetime reward}(T) = \frac{1}{T}\sum_{t=1}^{T} r_t
$$ 
在控制问题中，智能体所选择的动作将决定其未来所能感知到的观测，因此智能体的目标是通过控制未来的观测序列来最大化其生命周期奖励。 

## 1️⃣ 核心思想

- 继承 True Online TD (λ) 的在线更新机制
- 引入 SwiftTD 的 **步长优化**、**有效学习率约束** 与 **步长衰减** 三大改进
- 通过对策略梯度的在线估计，实现对策略的直接优化

## 2️⃣ 算法步骤 (Pseudo-code)

```python
initialize θ ← 0          # value function parameters
initialize φ ← 0          # policy parameters
initialize α, β, $λ$       # step-size, decay factor, eligibility trace
for each episode: 
    initialize eligibility traces e ← 0
    observe state s
    choose action a ∼ π_φ(s)
    while not terminal:
        observe reward r, next state s'
        choose next action a' ∼ π_φ(s')
        δ ← r + γ * Q_θ(s', a') - Q_θ(s, a)
        e ← γ * λ * e + ∇_θ Q_θ(s, a)
        # SwiftTD style update for value
        θ ← θ + α * δ * e
        # Effective learning rate constraint
        α ← clamp(α, α_min, α_max)
        # Step-size decay
        α ← α * β
        # Policy update (Sarsa)
        φ ← φ + α * δ * ∇_φ log π_φ(a|s)
        s ← s'; a ← a'
```

## 3️⃣ 参数与调优

- **α**：初始步长，随后通过 **β** 进行指数衰减
- **α_min / α_max**：保证有效学习率不超出安全区间
- **λ**：控制资格迹衰减，取值 0–1
- **γ**：折扣因子，通常取 0.99

## 4️⃣ 性能与鲁棒性

- 在 Atari 预测任务中，Swift-Sarsa 统一优于 True Online Sarsa (λ) 与传统 Sarsa
- 对超参数 **α, $λ$** 的选择更具鲁棒性，减少了调参成本
- 步长衰减机制有效防止过早收敛，保持长期探索

## 5️⃣ 关键实现细节

- **Effective Learning Rate Constraint**：使用 `clamp` 函数限制 `α`，避免梯度爆炸
- **Step-Size Decay**：通过 `β` (0<β<1) 实现指数衰减，保持学习率随时间慢慢降低
- **Eligibility Traces**：与 True Online TD 保持一致，使用 `γ * λ` 递推

:::caution
在使用 Swift-Sarsa 时，需注意 **学习率衰减** 与 **政策梯度更新** 的步长同步，否则可能导致策略学习不稳定。
:::
