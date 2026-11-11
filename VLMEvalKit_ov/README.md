<div align="center">

# 🔮 A Toolkit for Evaluating NEO-ov Models

<p>
  <b>Comprehensive Evaluation for NEO-ov across Images, Videos, and Spatial Intelligence</b>
</p>

<p>
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models-2B%20%7C%209B-4A90D9?style=for-the-badge" alt="Models"/>
  &nbsp;
  <img src="https://img.shields.io/badge/%F0%9F%93%8A%20Benchmarks-30-2EA44F?style=for-the-badge" alt="Benchmarks"/>
  &nbsp;
  <img src="https://img.shields.io/badge/%F0%9F%8F%B7%EF%B8%8F%20Categories-3-E8710A?style=for-the-badge" alt="Categories"/>
</p>

<p>
  <a href="#-getting-started">Getting Started</a> &bull;
  <a href="#-model-zoo">Model Zoo</a> &bull;
  <a href="#-benchmark-results">Results</a>
</p>

</div>

<br/>

## 🚀 Getting Started

### Dependencies

- 🔥 **PyTorch** `2.8.0`
- 🟢 **CUDA** `12.8`
- 🤗 **Transformers** `4.57.3`

### Installation

```bash
# 1. Create conda environment
conda create -n neo-ov python=3.10 -y
conda activate neo-ov

# 2. Install PyTorch (CUDA 12.8)
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# 3. Install transformers
pip install transformers==4.57.3

# 4. Install other dependencies
pip install -r requirements.txt
```

### Evaluation

```bash
bash eval_image.sh   # 🧠 Image Understanding
bash eval_video.sh   # 🎬 Video Understanding
bash eval_si.sh      # 🌐 Spatial Intelligence
```

> 💡 **Tip:** Edit the `DATASETS` array in each script to select specific benchmarks.

Built on top of [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). Refer to the original repository for detailed usage instructions.

<br/>

---

## 🤖 Model Zoo

We release 2B and 9B **NEO-ov** in Supervised Fine-Tuning (SFT).

<div style="overflow-x:auto;">
<table border="1" cellspacing="0" cellpadding="6" style="white-space:nowrap;">
  <tr>
    <th align="center">Model Name</th>
    <th align="center">Model Weight</th>
  </tr>
  <tr>
    <td>NEO-ov-2B-SFT</td>
    <td><a href="https://huggingface.co/Paranioar/NEO1_5-2B-SFT"><img src="assets/huggingface_logo.svg" width="16" height="16" /> NEO_ov-2B-SFT HF link</a></td>
  </tr>
  <tr>
    <td>NEO-ov-9B-SFT</td>
    <td><a href="https://huggingface.co/Paranioar/NEO1_5-9B-SFT"><img src="assets/huggingface_logo.svg" width="16" height="16" /> NEO_ov-9B-SFT HF link</a></td>
  </tr>
</table>
</div>

<br/>

## 📊 Benchmark Results

This project evaluates **NEO-ov** models across **3 major benchmark categories**:

| # | Category | Benchmarks | Description |
|:---:|:---|:---:|:---|
| 1 | 🧠 [Image Understanding](#-category-1-image-understanding) | 12 | Knowledge reasoning, general & OCR visual QA, hallucination detection |
| 2 | 🎬 [Video Understanding](#-category-2-video-understanding) | 6 | Video comprehension across varying lengths and complexity |
| 3 | 🌐 [Spatial Intelligence](#-category-3-spatial-intelligence) | 12 | Spatial reasoning on images, multi-images, and videos |


### 🧠 Category 1: Image Understanding

<div style="overflow-x:auto;">
<table border="1" cellspacing="0" cellpadding="6" style="white-space:nowrap;">
  <tr>
    <th rowspan="2" align="left">Model_NAME</th>
    <th rowspan="2" align="left">Base_LLM_NAME</th>
    <th colspan="1" align="center">Knowledge</th>
    <th colspan="4" align="center">General VQA</th>
    <th colspan="5" align="center">OCR VQA</th>
    <th colspan="2" align="center">Hallucination</th>
  </tr>
  <tr>
    <th align="center">MMMU-VAL</th>
    <th align="center">MMB-EN</th>
    <th align="center">RealWorldQA</th>
    <th align="center">MMStar</th>
    <th align="center">SEED-I</th>
    <th align="center">AI2D</th>
    <th align="center">DocVQA</th>
    <th align="center">ChartQA</th>
    <th align="center">TextVQA</th>
    <th align="center">OCRBench</th>
    <th align="center">POPE</th>
    <th align="center">HallusionBench</th>
  </tr>
  <tr>
  <!-- Native VLMs (2B) -->
  <tr><td colspan="14" align="left">🔻<b>Vision Language Models (Instruct-2B)</b></td></tr>
  <tr>
    <td>InternVL3.5</td><td>Qwen3-1.7B</td>
    <td align="center">53.0</td>
    <td align="center">78.2</td>
    <td align="center">62.0</td>
    <td align="center">62.7</td>
    <td align="center">75.3</td>
    <td align="center">78.8</td>
    <td align="center">89.4</td>
    <td align="center">80.7</td>
    <td align="center">76.5</td>
    <td align="center">83.6</td>
    <td align="center">87.2</td>
    <td align="center">48.6</td>
  </tr>
  <tr>
    <td>Qwen3-VL</td><td>Qwen3-1.7B</td>
    <td align="center">53.4</td>
    <td align="center">78.4</td>
    <td align="center">63.9</td>
    <td align="center">58.3</td>
    <td align="center">--</td>
    <td align="center">76.9</td>
    <td align="center">93.3</td>
    <td align="center">79.1</td>
    <td align="center">--</td>
    <td align="center">85.8</td>
    <td align="center">--</td>
    <td align="center">51.4</td>
  </tr>
  <tr>
    <td><b>NEO-ov</b></td><td>Qwen3-1.7B</td>
    <td align="center"><b>54.7</b></td>
    <td align="center"><b>80.0</b></td>
    <td align="center"><b>64.7</b></td>
    <td align="center"><b>58.7</b></td>
    <td align="center"><b>76.1</b></td>
    <td align="center"><b>81.7</b></td>
    <td align="center"><b>91.2</b></td>
    <td align="center"><b>83.7</b></td>
    <td align="center"><b>77.8</b></td>
    <td align="center"><b>81.2</b></td>
    <td align="center"><b>86.2</b></td>
    <td align="center"><b>54.6</b></td>
  </tr>
  <!-- Native VLMs (8B) -->
  <tr><td colspan="14" align="left">🔻<b>Vision Language Models (Instruct-8B)</b></td></tr>
  <tr>
  <tr>
    <td>InternVL3.5</td><td>Qwen3-8B</td>
    <td align="center">68.1</td>
    <td align="center">82.7</td>
    <td align="center">67.5</td>
    <td align="center">69.3</td>
    <td align="center">77.1</td>
    <td align="center">84.0</td>
    <td align="center">92.3</td>
    <td align="center">86.7</td>
    <td align="center">78.2</td>
    <td align="center">84</td>
    <td align="center">88.7</td>
    <td align="center">54.5</td>
  </tr>
  <tr>
    <td>Qwen3-VL</td><td>Qwen3-8B</td>
    <td align="center">69.6</td>
    <td align="center">84.5</td>
    <td align="center">71.5</td>
    <td align="center">70.9</td>
    <td align="center">--</td>
    <td align="center">85.7</td>
    <td align="center">96.1</td>
    <td align="center">89.6</td>
    <td align="center">--</td>
    <td align="center">89.6</td>
    <td align="center">--</td>
    <td align="center">61.1</td>
  </tr>
  <tr>
    <td><b>NEO-ov</b></td><td>Qwen3-8B</td>
    <td align="center"><b>68.1</b></td>
    <td align="center"><b>85.1</b></td>
    <td align="center"><b>68.8</b></td>
    <td align="center"><b>67.2</b></td>
    <td align="center"><b>76.5</b></td>
    <td align="center"><b>85.4</b></td>
    <td align="center"><b>91.8</b></td>
    <td align="center"><b>86.1</b></td>
    <td align="center"><b>78.6</b></td>
    <td align="center"><b>81.5</b></td>
    <td align="center"><b>89.0</b></td>
    <td align="center"><b>59.8</b></td>
  </tr>
</table>
</div>

<br/>

### 🎬 Category 2: Video Understanding

<div style="overflow-x:auto;">
<table border="1" cellspacing="0" cellpadding="6" style="white-space:nowrap;">
  <tr>
    <th align="left">Model_NAME</th>
    <th align="left">Base_LLM_NAME</th>
    <th align="center">Video-MME</th>
    <th align="center">MVBench</th>
    <th align="center">LVBench</th>
    <th align="center">MLVU (M-Avg)</th>
    <th align="center">LongVideoBench</th>
    <th align="center">Video-MMMU</th>
  </tr>
  <!-- Native VLMs (2B) -->
  <tr><td colspan="8" align="left">🔻<b>Vision Language Models (Instruct-2B)</b></td></tr>
  <tr>
  <tr>
    <td>InternVL3.5</td><td>Qwen3-1.7B</td>
    <td align="center">58.4</td>
    <td align="center">65.9</td>
    <td align="center">37.6</td>
    <td align="center">64.4</td>
    <td align="center">57.4</td>
    <td align="center">42.7</td>
  </tr>
  <tr>
    <td>Qwen3-VL</td><td>Qwen3-1.7B</td>
    <td align="center">61.9</td>
    <td align="center">61.7</td>
    <td align="center">47.4</td>
    <td align="center">68.3</td>
    <td align="center">55.6</td>
    <td align="center">41.9</td>
  </tr>
  <tr>
    <td><b>NEO-ov</b></td><td>Qwen3-1.7B</td>
    <td align="center"><b>60.4</b></td>
    <td align="center"><b>65.7</b></td>
    <td align="center"><b>43.3</b></td>
    <td align="center"><b>64.8</b></td>
    <td align="center"><b>56.8</b></td>
    <td align="center"><b>42.3</b></td>
  </tr>
  <!-- Native VLMs (8B) -->
  <tr><td colspan="8" align="left">🔻<b>Vision Language Models (Instruct-8B)</b></td></tr>
  <tr>
    <td>InternVL3.5</td><td>Qwen3-8B</td>
    <td align="center">66.0</td>
    <td align="center">72.1</td>
    <td align="center">45.9</td>
    <td align="center">70.2</td>
    <td align="center">62.1</td>
    <td align="center">54.9</td>
  </tr>
    <tr>
    <td>Qwen3-VL</td><td>Qwen3-8B</td>
    <td align="center">71.4</td>
    <td align="center">68.7</td>
    <td align="center">58.0</td>
    <td align="center">78.1</td>
    <td align="center">63.6</td>
    <td align="center">65.3</td>
  </tr>
  <tr>
    <td><b>NEO-ov</b></td><td>Qwen3-8B</td>
    <td align="center"><b>67.4</b></td>
    <td align="center"><b>70.7</b></td>
    <td align="center"><b>46.4</b></td>
    <td align="center"><b>69.3</b></td>
    <td align="center"><b>63.5</b></td>
    <td align="center"><b>51.6</b></td>
  </tr>
</table>
</div>

<br/>

### 🌐 Category 3: Spatial Intelligence

<div style="overflow-x:auto;">
<table border="1" cellspacing="0" cellpadding="6" style="white-space:nowrap;">
  <tr>
    <th align="left">Model_NAME</th>
    <th align="left">Base_LLM_NAME</th>
    <th align="center">VSI-Bench</th>
    <th align="center">MMSI</th>
    <th align="center">Mindcube-tiny</th>
    <th align="center">ViewSpatial</th>
    <th align="center">SITE</th>
    <th align="center">3DSR</th>
    <th align="center">EmbSpatial</th>
    <th align="center">SPAR</th>
    <th align="center">MMSI-video</th>
    <th align="center">Omni_Manual cot</th>
    <th align="center">BLINK</th>
    <th align="center">MUIRBENCH</th>
  </tr>
  <!-- Native VLMs (2B) -->
  <tr><td colspan="14" align="left">🔻<b>Vision Language Models (Instruct-2B)</b></td></tr>
  <tr>
    <td>InternVL3.5</td><td>Qwen3-1.7B</td>
    <td align="center">53.8</td>
    <td align="center">25.6</td>
    <td align="center">42.1</td>
    <td align="center">37.9</td>
    <td align="center">34.8</td>
    <td align="center">31.4</td>
    <td align="center">61.5</td>
    <td align="center">32.4</td>
    <td align="center">25.9</td>
    <td align="center">44.4</td>
    <td align="center">51.3</td>
    <td align="center">44</td>
  </tr>

  <tr>
    <td>Qwen3-VL</td><td>Qwen3-1.7B</td>
    <td align="center">53.9</td>
    <td align="center">27.8</td>
    <td align="center">34.2</td>
    <td align="center">36.7</td>
    <td align="center">35.8</td>
    <td align="center">47.6</td>
    <td align="center">69.2</td>
    <td align="center">34.1</td>
    <td align="center">25.6</td>
    <td align="center">36.3</td>
    <td align="center">53.8</td>
    <td align="center">47.4</td>
  </tr>
  <tr>
    <td><b>NEO-ov</b></td><td>Qwen3-1.7B</td>
    <td align="center"><b>58.4</b></td>
    <td align="center"><b>33.6</b></td>
    <td align="center"><b>77.2</b></td>
    <td align="center"><b>52.8</b></td>
    <td align="center"><b>38.4</b></td>
    <td align="center"><b>52.9</b></td>
    <td align="center"><b>63.8</b></td>
    <td align="center"><b>41.2</b></td>
    <td align="center"><b>23.1</b></td>
    <td align="center"><b>43.1</b></td>
    <td align="center"><b>53.9</b></td>
    <td align="center"><b>56.8</b></td>
  </tr>
  <!-- Native VLMs (8B) -->
  <tr><td colspan="14" align="left">🔻<b>Vision Language Models (Instruct-8B)</b></td></tr>
  <tr>
    <td>InternVL3.5</td><td>Qwen3-8B</td>
    <td align="center">56.3</td>
    <td align="center">29.1</td>
    <td align="center">40.4</td>
    <td align="center">40</td>
    <td align="center">54.4</td>
    <td align="center">35.3</td>
    <td align="center">75.7</td>
    <td align="center">38.2</td>
    <td align="center">28</td>
    <td align="center">47.8</td>
    <td align="center">59.5</td>
    <td align="center">55.8</td>
  </tr>
  <tr>
    <td>Qwen3-VL</td><td>Qwen3-8B</td>
    <td align="center">59.4</td>
    <td align="center">31.2</td>
    <td align="center">29.6</td>
    <td align="center">41.9</td>
    <td align="center">45.4</td>
    <td align="center">52.9</td>
    <td align="center">77.8</td>
    <td align="center">40.3</td>
    <td align="center">28.4</td>
    <td align="center">47</td>
    <td align="center">69.1</td>
    <td align="center">64.4</td>
  </tr>
  <tr>
    <td><b>NEO-ov</b></td><td>Qwen3-8B</td>
    <td align="center"><b>64.8</b></td>
    <td align="center"><b>41.3</b></td>
    <td align="center"><b>90</b></td>
    <td align="center"><b>55.2</b></td>
    <td align="center"><b>54.3</b></td>
    <td align="center"><b>61.7</b></td>
    <td align="center"><b>78.8</b></td>
    <td align="center"><b>48.8</b></td>
    <td align="center"><b>28.7</b></td>
    <td align="center"><b>45</b></td>
    <td align="center"><b>62.8</b></td>
    <td align="center"><b>58.2</b></td>
  </tr>
</table>
</div>

<br/>


## 📊 Demonstration

```python
# Demo
from vlmeval.config import supported_VLM
model = supported_VLM['NEOov-2B-image']()
# Forward Single Image
ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
print(ret)  # The image features a red apple with a leaf on it.
# Forward Multiple Images
ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', 'How many apples are there in the provided images? '])
print(ret)  # There are two apples in the provided images.
```

