<div align="center">

<b>VLMEvalKit: 一种多模态大模型评测工具 </b>

[English](https://github.com/EvolvingLMMs-Lab/NEO/blob/main/VLMEvalKit/README.md) | 简体中文  

<a href="https://rank.opencompass.org.cn/leaderboard-multimodal">🏆 OpenCompass 排行榜 </a> •
<a href="#%EF%B8%8F-quickstart">🏗️ 快速开始 </a> •
<a href="#-datasets-models-and-evaluation-results">📊 数据集和模型 </a> •
<a href="#%EF%B8%8F-development-guide">🛠️ 开发指南 </a> •
<a href="#-the-goal-of-vlmevalkit">🎯 我们的目标 </a> •
<a href="#%EF%B8%8F-citation">🖊️ 引用 </a>

<a href="https://huggingface.co/spaces/opencompass/open_vlm_leaderboard">🤗 HuggingFace 排行榜 (存档全部性能) </a> •
<a href="https://huggingface.co/datasets/VLMEval/OpenVLMRecords">🤗 原始评测记录</a> •
<a href="https://discord.gg/evDT4GZmxN">🔊 Discord</a> •
<a href="https://www.arxiv.org/abs/2407.11691">📝 技术报告 </a>
</div>

**VLMEvalKit** (python 包名为 **vlmeval**) 是一款专为大型视觉语言模型 (Large Vision-Language Models， LVLMs) 评测而设计的开源工具包。该工具支持在各种基准测试上对大型视觉语言模型进行**一键评估**，无需进行繁重的数据准备工作，让评估过程更加简便。在 VLMEvalKit 中，我们对所有大型视觉语言模型生成的结果进行评测，并提供基于**精确匹配**与基于 **LLM 的答案提取**两种评测结果。


## 🏗️ 快速开始 <a id="quickstart"></a>

请参阅[**快速开始**](https://github.com/EvolvingLMMs-Lab/NEO/blob/main/VLMEvalKit/docs/zh-CN/Quickstart.md)获取入门指南。

## 📊 测试演示 <a id="data-model-results"></a>

```python
from vlmeval.config import supported_VLM
model = supported_VLM['NEO-2B-SFT']()
# 前向单张图片
ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
print(ret)  # 这张图片上有一个带叶子的红苹果
# 前向多张图片
ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', 'How many apples are there in the provided images? '])
print(ret)  # 提供的图片中有两个苹果
```