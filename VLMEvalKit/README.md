<div align="center">

# 📐 A Toolkit for Evaluating NEO Models

<p>
  <b>Comprehensive Evaluation for NEO across Knowledge, Hallucination, General and OCR VQA</b>
</p>

<p>
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models-2B%20%7C%209B-4A90D9?style=for-the-badge" alt="Models"/>
  &nbsp;
  <img src="https://img.shields.io/badge/%F0%9F%93%8A%20Benchmarks-13-2EA44F?style=for-the-badge" alt="Benchmarks"/>
</p>

<p>
  <a href="#%EF%B8%8F-quickstart">Usage & Scripts</a> &bull;
  <a href="#-model-zoo">Model Zoo</a> &bull;
  <a href="#-benchmark-results">Results</a>
</p>

</div>

<br/>

## 🏗️ QuickStart

See [[QuickStart](./docs/en/Quickstart.md) | [快速开始](./docs/zh-CN/Quickstart.md)] for a quick start guide.

## 🤖 Model Zoo

We release 2B and 9B **NEO** in Pre-Training (PT), Mid-Training (MT), and Supervised Fine-Tuning (SFT). 

<div style="overflow-x:auto;">
<table border="1" cellspacing="0" cellpadding="6" style="white-space:nowrap;">
  <tr>
    <th align="center">Model Name</th>
    <th align="center">Model Weight</th>
  </tr>
  <tr>
    <td>NEO-2B-PT</td>
    <td><a href="https://huggingface.co/Paranioar/NEO1_0-2B-PT">🤗 NEO-2B-PT HF link</a></td>
  </tr>
  <tr>
    <td>NEO-2B-MT</td>
    <td><a href="https://huggingface.co/Paranioar/NEO1_0-2B-MT">🤗 NEO-2B-MT HF link</a></td>
  </tr>
  <tr>
    <td>NEO-2B-SFT</td>
    <td><a href="https://huggingface.co/Paranioar/NEO1_0-2B-SFT">🤗 NEO-2B-SFT HF link</a></td>
  </tr>
  <tr>
    <td>NEO-9B-PT</td>
    <td><a href="https://huggingface.co/Paranioar/NEO1_0-9B-PT">🤗 NEO-9B-PT HF link</a></td>
  </tr>
  <tr>
    <td>NEO-9B-MT</td>
    <td><a href="https://huggingface.co/Paranioar/NEO1_0-9B-MT">🤗 NEO-9B-MT HF link</a></td>
  </tr>
  <tr>
    <td>NEO-9B-SFT</td>
    <td><a href="https://huggingface.co/Paranioar/NEO1_0-9B-SFT">🤗 NEO-9B-SFT HF link</a></td>
  </tr>
</table>
</div>

## 📊 Benchmark Results

> **TABLE NOTE:**  
> - “# Data” = data scale for pre-training / mid-training / supervised fine-tuning.  
> - “†“ = vision-language models using Reinforcement Learning (RL).  
> - “Any Res.” = any resolution; “Tile-wise” = image split into tiles; <br> “Any Rat.” = any aspect ratio; “Fix Res.” = fixed resolution.  
> - “MoE“ = Mixture-of-Experts; “DaC“ = Divide-and-Conquer.  
> - **Bold** = best score in each column. <br><br>

<div style="overflow-x:auto;">
<table border="1" cellspacing="0" cellpadding="6" style="white-space:nowrap;">
  <tr>
    <th rowspan="2" align="left">Model_NAME</th>
    <th rowspan="2" align="left">Base_LLM_NAME</th>
    <th rowspan="2" align="left">#Data_PT·MT·SFT</th>
    <th rowspan="2" align="left">Input_TYPE</th>
    <th rowspan="2" align="left">RoPE_TYPE</th>
    <th colspan="1" align="center">Knowledge</th>
    <th colspan="4" align="center">General VQA</th>
    <th colspan="6" align="center">OCR VQA</th>
    <th colspan="2" align="center">Hallucination</th>
  </tr>
  <tr>
    <th align="center">MMMU</th>
    <th align="center">MMB</th>
    <th align="center">MMVet</th>
    <th align="center">MMStar</th>
    <th align="center">SEED_I</th>
    <th align="center">AI2D</th>
    <th align="center">DocVQA</th>
    <th align="center">ChartQA</th>
    <th align="center">InfoVQA</th>
    <th align="center">TextVQA</th>
    <th align="center">OCRBench</th>
    <th align="center">POPE</th>
    <th align="center">HallB</th>
  </tr>
  <!-- Modular VLMs (2B) -->
  <tr><td colspan="18" align="left">🔻<b>Modular Vision Language Models (Instruct-2B)</b></td></tr>
  <tr>
    <td>Qwen2-VL</td><td>Qwen2-1.5B</td><td>--·--·--</td><td>Any Res.</td><td>M-RoPE</td>
    <td align="center">41.1</td><td align="center">74.9</td><td align="center">49.5</td><td align="center">48.0</td><td align="center">--</td>
    <td align="center">74.7</td><td align="center"><b>90.1</b></td><td align="center">73.5</td><td align="center">65.5</td><td align="center"><b>79.7</b></td><td align="center">80.9</td>
    <td align="center">--</td><td align="center">41.7</td>
  </tr>
  <tr>
    <td>InternVL2.5</td><td>InternLM2.5-1.8B</td><td>&gt;6B·100M·16M</td><td>Tile-wise</td><td>1D-RoPE</td>
    <td align="center">43.6</td><td align="center">74.7</td><td align="center">60.8</td><td align="center">53.7</td><td align="center">--</td>
    <td align="center">74.9</td><td align="center">88.7</td><td align="center">79.2</td><td align="center">60.9</td><td align="center">74.3</td><td align="center">80.4</td>
    <td align="center"><b>90.6</b></td><td align="center">42.6</td>
  </tr>
  <tr>
    <td>InternVL3†</td><td>Qwen2.5-1.5B</td><td>&gt;6B·100M·22M</td><td>Tile-wise</td><td>1D-RoPE</td>
    <td align="center"><b>48.6</b></td><td align="center"><b>81.1</b></td><td align="center"><b>62.2</b></td><td align="center"><b>60.7</b></td><td align="center">--</td>
    <td align="center"><b>78.7</b></td><td align="center">88.3</td><td align="center"><b>80.2</b></td><td align="center"><b>66.1</b></td><td align="center">77.0</td><td align="center"><b>83.5</b></td>
    <td align="center">89.6</td><td align="center">42.5</td>
  </tr>
  <tr>
    <td><i>Qwen2.5-VL†</i></td><td><i><b>Qwen2.5-3B</b></i></td><td><i>--·--·--</i></td><td><i>Any Res.</i></td><td><i>M-RoPE</i></td>
    <td align="center"><i>51.2</i></td><td align="center"><i>79.1</i></td><td align="center"><i>61.8</i></td><td align="center"><i>55.9</i></td><td align="center">--</td>
    <td align="center"><i>81.6</i></td><td align="center"><i>93.9</i></td><td align="center"><i>84.0</i></td><td align="center"><i>77.1</i></td><td align="center"><i>79.3</i></td><td align="center"><i>79.7</i></td>
    <td align="center">--</td><td align="center"><i>46.3</i></td>
  </tr>
  <tr>
    <td>Encoder_Based</td><td>Qwen3-1.7B</td><td>&gt;6B·40M·4M</td><td>Tile-wise</td><td>1D-RoPE</td>
    <td align="center">47.1</td><td align="center">75.8</td><td align="center">37.4</td><td align="center">52.7</td><td align="center"><b>73.6</b></td>
    <td align="center">77.4</td><td align="center">89.9</td><td align="center">78.4</td><td align="center">65.9</td><td align="center">73.3</td><td align="center"><b>83.5</b></td>
    <td align="center">87.0</td><td align="center"><b>44.4</b></td>
  </tr>
  <!-- Native VLMs (2B) -->
  <tr><td colspan="18" align="left">🔻<b>Native Vision Language Models (Instruct-2B)</b></td></tr>
  <tr>
    <td>Mono-InternVL</td><td>InternLM2-1.8B</td><td>1.2B·143M·7M</td><td>Tile-wise</td><td>1D-RoPE</td>
    <td align="center">33.7</td><td align="center">65.5</td><td align="center">40.1</td><td align="center">--</td><td align="center">67.4</td>
    <td align="center">68.6</td><td align="center">80.0</td><td align="center">73.7</td><td align="center">43.0</td><td align="center">72.6</td><td align="center">76.7</td>
    <td align="center">--</td><td align="center">34.8</td>
  </tr>
  <tr>
    <td>Mono-InternVL-1.5</td><td>InternLM2-1.8B</td><td>400M·150M·7M</td><td>Tile-wise</td><td>1D-RoPE</td>
    <td align="center">39.1</td><td align="center">64.0</td><td align="center"><b>54.0</b></td><td align="center">--</td><td align="center">66.9</td>
    <td align="center">67.4</td><td align="center">81.7</td><td align="center">72.2</td><td align="center">47.9</td><td align="center">73.7</td><td align="center"><b>80.1</b></td>
    <td align="center">--</td><td align="center">32.5</td>
  </tr>
  <tr>
    <td>HoVLE</td><td>InternLM2-1.8B</td><td>550M·50M·7M</td><td>Tile-wise</td><td>1D-RoPE</td>
    <td align="center">32.2</td><td align="center">73.3</td><td align="center">43.8</td><td align="center">--</td><td align="center">70.9</td>
    <td align="center">73.0</td><td align="center">86.1</td><td align="center">78.6</td><td align="center">55.7</td><td align="center">70.9</td><td align="center">74.0</td>
    <td align="center">87.4</td><td align="center">38.4</td>
  </tr>
  <tr>
    <td>OneCAT</td><td>Qwen2.5-1.5B</td><td>436M·70M·13M</td><td>Any Res.</td><td>M-RoPE</td>
    <td align="center">39.0</td><td align="center">72.4</td><td align="center">42.4</td><td align="center">--</td><td align="center">70.9</td>
    <td align="center">72.4</td><td align="center">87.1</td><td align="center">76.2</td><td align="center">56.3</td><td align="center">67.0</td><td align="center">--</td>
    <td align="center">--</td><td align="center">--</td>
  </tr>
  <tr>
    <td><b>NEO</b></td><td>Qwen3-1.7B</td><td>345M·40M·4M</td><td>Any Res.</td><td>Native_RoPE</td>
    <td align="center"><b>48.6</b></td><td align="center"><b>76.0</b></td><td align="center">49.6</td><td align="center"><b>54.2</b></td><td align="center"><b>74.2</b></td>
    <td align="center"><b>80.1</b></td><td align="center"><b>89.9</b></td><td align="center"><b>81.2</b></td><td align="center"><b>63.2</b></td><td align="center"><b>74.0</b></td><td align="center">77.1</td>
    <td align="center"><b>87.5</b></td><td align="center"><b>43.1</b></td>
  </tr>
</table>
</div>

<br/>

<div style="overflow-x:auto;">
<table border="1" cellspacing="0" cellpadding="6" style="white-space:nowrap;">
  <tr>
    <th rowspan="2" align="left">Model_NAME</th>
    <th rowspan="2" align="left">Base_LLM_NAME</th>
    <th rowspan="2" align="left">#Data_PT·MT·SFT</th>
    <th rowspan="2" align="left">Input_TYPE</th>
    <th rowspan="2" align="left">RoPE_TYPE</th>
    <th colspan="1" align="center">📚 Knowledge</th>
    <th colspan="4" align="center">💬 General VQA</th>
    <th colspan="6" align="center">🔍 OCR VQA</th>
    <th colspan="2" align="center">👻 Hallucination</th>
  </tr>
  <tr>
    <th align="center">MMMU</th>
    <th align="center">MMB</th>
    <th align="center">MMVet</th>
    <th align="center">MMStar</th>
    <th align="center">SEED_I</th>
    <th align="center">AI2D</th>
    <th align="center">DocVQA</th>
    <th align="center">ChartQA</th>
    <th align="center">InfoVQA</th>
    <th align="center">TextVQA</th>
    <th align="center">OCRBench</th>
    <th align="center">POPE</th>
    <th align="center">HallB</th>
  </tr>
  <!-- Modular VLMs (8B) -->
  <tr><td colspan="18" align="left">🔻<b>Modular Vision Language Models (Instruct-8B)</b></td></tr>
  <tr>
    <td>Qwen2-VL</td><td>Qwen2-7B</td><td>--·--·--</td><td>Any Res.</td><td>M-RoPE</td>
    <td align="center">54.1</td><td align="center">83.0</td><td align="center">62.0</td><td align="center">60.7</td><td align="center">--</td>
    <td align="center">83.0</td><td align="center">94.5</td><td align="center">83.0</td><td align="center">76.5</td><td align="center">84.3</td><td align="center">86.6</td>
    <td align="center">88.1</td><td align="center">50.6</td>
  </tr>
  <tr>
    <td>InternVL2.5</td><td>InternLM2.5-7B</td><td>&gt;6B·50M·4M</td><td>Tile-wise</td><td>1D-RoPE</td>
    <td align="center">56.0</td><td align="center"><b>84.6</b></td><td align="center">62.8</td><td align="center">64.4</td><td align="center">--</td>
    <td align="center">84.5</td><td align="center">93.0</td><td align="center">84.8</td><td align="center">77.6</td><td align="center">79.1</td><td align="center">82.2</td>
    <td align="center">90.6</td><td align="center">50.1</td>
  </tr>
  <tr>
    <td>Qwen2.5-VL†</td><td>Qwen2.5-7B</td><td>--·--·--</td><td>Any Res.</td><td>M-RoPE</td>
    <td align="center">55.0</td><td align="center">83.5</td><td align="center">67.1</td><td align="center">63.9</td><td align="center">--</td>
    <td align="center">83.9</td><td align="center"><b>95.7</b></td><td align="center"><b>87.3</b></td><td align="center"><b>82.6</b></td><td align="center"><b>84.9</b></td><td align="center">86.4</td>
    <td align="center">86.4</td><td align="center"><b>52.9</b></td>
  </tr>
  <tr>
    <td>InternVL3†</td><td>Qwen2.5-7B</td><td>&gt;6B·100M·22M</td><td>Tile-wise</td><td>1D-RoPE</td>
    <td align="center"><b>62.7</b></td><td align="center">83.4</td><td align="center"><b>81.3</b></td><td align="center"><b>68.2</b></td><td align="center">--</td>
    <td align="center"><b>85.2</b></td><td align="center">92.7</td><td align="center">86.6</td><td align="center">76.8</td><td align="center">80.2</td><td align="center"><b>88.0</b></td>
    <td align="center"><b>91.1</b></td><td align="center">49.9</td>
  </tr>
  <tr>
    <td>Encoder-Based</td><td>Qwen3-8B</td><td>&gt;6B·40M·4M</td><td>Tile-wise</td><td>1D-RoPE</td>
    <td align="center">54.1</td><td align="center">84.0</td><td align="center">60.0</td><td align="center">63.5</td><td align="center"><b>76.2</b></td>
    <td align="center">82.9</td><td align="center">92.1</td><td align="center">83.5</td><td align="center">75.0</td><td align="center">77.1</td><td align="center">85.3</td>
    <td align="center">87.8</td><td align="center">51.4</td>
  </tr>
  <!-- Native VLMs (8B) -->
  <tr><td colspan="18" align="left">🔻<b>Native Vision Language Models (Instruct-8B)</b></td></tr>
  <tr>
    <td>Fuyu</td><td>Persimmon-8B</td><td>--·--·--</td><td>Any Res.</td><td>1D-RoPE</td>
    <td align="center">27.9</td><td align="center">10.7</td><td align="center">21.4</td><td align="center">--</td><td align="center">59.3</td>
    <td align="center">64.5</td><td align="center">--</td><td align="center">--</td><td align="center">--</td><td align="center">--</td><td align="center">36.6</td>
    <td align="center">84.0</td><td align="center">--</td>
  </tr>
  <tr>
    <td>Chameleon</td><td>from scratch</td><td>1.4B·0M·1.8M</td><td>Fix Res.</td><td>1D-RoPE</td>
    <td align="center">25.4</td><td align="center">31.1</td><td align="center">8.3</td><td align="center">--</td><td align="center">30.6</td>
    <td align="center">46.0</td><td align="center">1.5</td><td align="center">2.9</td><td align="center">5.0</td><td align="center">4.8</td><td align="center">0.7</td>
    <td align="center">19.4</td><td align="center">17.1</td>
  </tr>
  <tr>
    <td>EVE</td><td>Vicuna-7B</td><td>33M·0M·1.8M</td><td>Any Rat.</td><td>1D-RoPE</td>
    <td align="center">32.6</td><td align="center">52.3</td><td align="center">25.7</td><td align="center">--</td><td align="center">64.6</td>
    <td align="center">61.0</td><td align="center">53.0</td><td align="center">59.1</td><td align="center">25.0</td><td align="center">56.8</td><td align="center">39.8</td>
    <td align="center">85.0</td><td align="center">26.4</td>
  </tr>
  <tr>
    <td>SOLO</td><td>Mistral-7B</td><td>44M·0M·2M</td><td>Any Res.</td><td>1D-RoPE</td>
    <td align="center">--</td><td align="center">67.7</td><td align="center">30.4</td><td align="center">--</td><td align="center">64.4</td>
    <td align="center">61.4</td><td align="center">--</td><td align="center">--</td><td align="center">--</td><td align="center">--</td><td align="center">12.6</td>
    <td align="center">78.6</td><td align="center">--</td>
  </tr>
  <tr>
    <td>Emu3</td><td>from scratch</td><td>--·--·--</td><td>Fix Res.</td><td>1D-RoPE</td>
    <td align="center">31.6</td><td align="center">58.5</td><td align="center">37.2</td><td align="center">--</td><td align="center">68.2</td>
    <td align="center">70.0</td><td align="center">76.3</td><td align="center">68.6</td><td align="center">43.8</td><td align="center">64.7</td><td align="center">68.7</td>
    <td align="center">85.2</td><td align="center">--</td>
  </tr>
  <tr>
    <td>EVEv2</td><td>Qwen2.5-7B</td><td>77M·15M·7M</td><td>Any Rat.</td><td>1D-RoPE</td>
    <td align="center">39.3</td><td align="center">66.3</td><td align="center">45.0</td><td align="center">--</td><td align="center">71.4</td>
    <td align="center">74.8</td><td align="center">--</td><td align="center">73.9</td><td align="center">--</td><td align="center">71.1</td><td align="center">70.2</td>
    <td align="center">87.6</td><td align="center">--</td>
  </tr>
  <tr>
    <td>BREEN</td><td>Qwen2.5-7B</td><td>13M·0M·4M</td><td>Any Res.</td><td>1D-RoPE</td>
    <td align="center">42.7</td><td align="center">71.4</td><td align="center">38.9</td><td align="center">51.2</td><td align="center">--</td>
    <td align="center">76.4</td><td align="center">--</td><td align="center">--</td><td align="center">--</td><td align="center">65.7</td><td align="center">--</td>
    <td align="center">--</td><td align="center">37.0</td>
  </tr>
  <tr>
    <td>VoRA</td><td>Qwen2.5-7B</td><td>30M·0M·0.6M</td><td>Any Res.</td><td>1D-RoPE</td>
    <td align="center">32.0</td><td align="center">61.3</td><td align="center">33.7</td><td align="center">--</td><td align="center">68.9</td>
    <td align="center">61.1</td><td align="center">--</td><td align="center">--</td><td align="center">--</td><td align="center">58.7</td><td align="center">--</td>
    <td align="center">85.5</td><td align="center">--</td>
  </tr>
  <tr>
    <td>SAIL</td><td>Mistral-7B</td><td>512M·86M·6M</td><td>Any Res.</td><td>M-RoPE</td>
    <td align="center">--</td><td align="center">70.1</td><td align="center">46.3</td><td align="center">53.1</td><td align="center">72.9</td>
    <td align="center">76.7</td><td align="center">--</td><td align="center">--</td><td align="center">--</td><td align="center"><b>77.1</b></td><td align="center"><b>78.3</b></td>
    <td align="center">85.8</td><td align="center"><b>54.2</b></td>
  </tr>
  <tr>
    <td><b>NEO</b></td><td>Qwen3-8B</td><td>345M·40M·4M</td><td>Any Res.</td><td>Native_RoPE</td>
    <td align="center"><b>54.6</b></td><td align="center"><b>82.1</b></td><td align="center"><b>53.6</b></td><td align="center"><b>62.4</b></td><td align="center"><b>76.3</b></td>
    <td align="center"><b>83.1</b></td><td align="center"><b>88.6</b></td><td align="center"><b>82.1</b></td><td align="center"><b>60.9</b></td><td align="center">75.0</td><td align="center">77.7</td>
    <td align="center"><b>88.4</b></td><td align="center">46.4</td>
  </tr>
</table>
</div>



## 📊 Demonstration

```python
# Demo
from vlmeval.config import supported_VLM
model = supported_VLM['NEO-2B-SFT']()
# Forward Single Image
ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
print(ret)  # The image features a red apple with a leaf on it.
# Forward Multiple Images
ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', 'How many apples are there in the provided images? '])
print(ret)  # There are two apples in the provided images.
```