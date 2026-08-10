# 思路二：MetaAsk-GRPO / Information-Deficit-Aware Reasoning RL

## 背景与问题

现有脚手架方法通常把模型失败解释成“需要更多解题指导”，再由外部系统提供 Knowledge、Planning 或 Solution Hint。但失败不一定意味着缺少知识，也可能来自错误的局部假设、分支选择或中间判断。

因此存在 Assistance Mismatch：模型真正缺失的信息，与系统提供的帮助类型或信息量不匹配。模型可能只缺一个 Yes/No 判断，却收到很长的规划提示；也可能真的缺少某个定理，此时简单的二元反馈又不够。

## 核心动机

研究问题不再是“怎样给模型更好的 Hint”，而是：

> 模型失败时究竟缺什么，以及它能否只获取刚好足够的外部信息？

## 方法框架

把求助作为 reasoning policy 自己控制的动作：

```text
Question → Student reasoning → State s_t
                            ↓
        REASON / VERIFY / KNOWLEDGE / PLANNING / STEP
                            ↓
                 Minimal Oracle response
                            ↓
                 Continue reasoning → Answer
```

不同帮助具有不同成本：

```text
C(VERIFY) < C(KNOWLEDGE) < C(PLANNING) < C(STEP)
```

优化目标为：

```text
R = R_correct - λ · C(information)
```

模型需要同时学习：是否求助、询问什么、选择哪类帮助，以及帮助是否值得其信息成本。

## Active Epistemic Verification

最低成本的动作不是向教师索取解法，而是验证当前推理中的一个 proposition。例如模型询问“题目是否保证 x>0？”，Oracle 只回答 Yes/No，不提供定理、下一步或最终答案。这类反馈适合模型已有知识、但局部 belief 出错的情况。

纯二元验证不能替代所有提示。推荐采用逐步扩展的帮助空间：

```text
No Help → Binary Verification → Knowledge → Planning → Solution Step
```

与 SCAF-GRPO 的关键区别是：升级顺序不由外部系统固定执行，而由 student 根据当前 reasoning state、信息缺口和 Value of Information 主动选择。

## 最小实验

第一阶段不直接训练完整在线 Ask Policy，而是在同一批 all-zero 问题上比较：

1. No Help；
2. SCAF hierarchical hint；
3. Binary Verification ×1/×2/×3；
4. Random Verification；
5. Self-Asked Verification；
6. Knowledge、Planning 和 Solution Hint。

重点报告 Rescue Rate、无提示最终性能、外部信息 token、Oracle calls、单位信息成本成功率，以及 Performance–Information Pareto Frontier。

## 与思路一的区别

思路一仍由训练系统选择能力匹配的子问题和脚手架，重点是让提示变得可学习、可撤除。MetaAsk 则把“是否求助、缺什么、问什么”本身变成模型策略的一部分，重点是 learner-controlled information acquisition。

## 当前状态

MetaAsk 目前是独立研究方案和实验设计，尚未实现在线交互训练代码，也尚未产生实验结果。
