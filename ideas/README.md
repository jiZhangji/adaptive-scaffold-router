# Research ideas

本目录保存当前准备在服务器验证的两个独立研究方向：

1. [`capability_matched_scaffold_curriculum_zh.md`](capability_matched_scaffold_curriculum_zh.md)：图片中的统一方案，先训练可验证子问题，再激活能力匹配的脚手架，最后通过提示退火和 off-context 修正回到无提示推理。
2. [`metaask_grpo_zh.md`](metaask_grpo_zh.md)：让模型识别自身信息缺口，并在显式成本约束下主动选择最小充分外部信息。

两者不是同一个方法：第一个方向主要解决“脚手架何时对当前模型真正可学”，第二个方向主要解决“模型失败时究竟缺什么，以及是否值得主动求助”。
