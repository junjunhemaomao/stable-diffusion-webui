# Diffusers txt2img

基于 HuggingFace Diffusers 的文生图脚本，支持 SD1.5 / SDXL，完全离线。

## 快速开始

```bash
# SDXL（默认，1024×1024, 30步）
venv/Scripts/python diffusers-scripts/txt2img.py

# SDXL 自定义参数
venv/Scripts/python diffusers-scripts/txt2img.py --prompt "a cat" --steps 30 --W 1024 --H 1024

# SD1.5
venv/Scripts/python diffusers-scripts/txt2img.py --model models/Stable-diffusion/v1-5-pruned-emaonly.safetensors --type sd15

# 50 系显卡 NaN 兜底（FP32 + 关 upcast）
venv/Scripts/python diffusers-scripts/txt2img.py --fp32 --no-upcast
```

也可以直接在 Python 里调用，不需要命令行：

```python
import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_single_file(
    "models/Stable-diffusion/sd_xl_base_1.0_0.9vae.safetensors",
    torch_dtype=torch.float16,
)
pipe.to("cuda")

# 修复 50 系显卡 FP16 加载导致的 position_ids 类型错误
for comp in [pipe.text_encoder, pipe.text_encoder_2]:
    emb = comp.text_model.embeddings
    if emb.position_ids.dtype != torch.long:
        emb.position_ids = emb.position_ids.to(torch.long)

pipe.vae.config.force_upcast = True   # 修复 NaN
pipe.enable_attention_slicing()       # 节省显存

image = pipe(
    "a cute cat",
    num_inference_steps=30,
    width=1024,
    height=1024,
).images[0]
image.save("output.png")
```

## 模型文件

| 文件 | 格式 | 大小 | 加载方式 |
|---|---|---|---|
| `v1-5-pruned-emaonly.safetensors` | 单文件 ckpt | 4.0 GB | `from_single_file` |
| `sd_xl_base_1.0_0.9vae.safetensors` | 单文件 ckpt (0.9VAE) | 6.5 GB | `from_single_file` |
| `sd_xl_base_1.0.safetensors` | 单文件 ckpt | 6.5 GB | `from_single_file` |

所有模型通过 `from_single_file` 加载，不需要 diffusers 分体格式目录，也不要额外的 ~10GB 权重副本。

## 换电脑 / 公司离线环境

`~/.cache/huggingface/` 不在 git 仓库里。首次运行 `from_single_file` 需要下载 ~2MB 配置文件，如果公司网络不通就会卡住。

**从家里拷贝缓存（~5MB）：**

把家里 `C:\Users\<用户名>\.cache\huggingface\hub\` 下这两个目录拷贝到公司同路径：
- `models--runwayml--stable-diffusion-v1-5\`
- `models--stabilityai--stable-diffusion-xl-base-1.0\`

拷完就是 100% 离线。

## 加载方式：from_single_file（100% 离线）

```python
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

# SD1.5
pipe = StableDiffusionPipeline.from_single_file(
    "models/Stable-diffusion/v1-5-pruned-emaonly.safetensors",
    torch_dtype=torch.float16,
)

# SDXL（用 0.9 VAE 版，修复 NaN）
pipe = StableDiffusionXLPipeline.from_single_file(
    "models/Stable-diffusion/sd_xl_base_1.0_0.9vae.safetensors",
    torch_dtype=torch.float16,
)
pipe.to("cuda")
```

首次运行会自动从 `hf-mirror.com` 下载 ~2MB 配置文件。缓存后完全离线。

## 参数默认值

| 参数 | SD1.5 | SDXL |
|---|---|---|
| 分辨率 | 512×512 | 1024×1024 |
| 步数 | 25 | 30 |
| CFG | 7.5 | 7.0 |

SDXL 原生训练分辨率 1024，低于此尺寸会导致画面扭曲/模糊。

## RTX 50 系列 NaN 修复

50 系显卡 FP16 精度 bug 会在 SDXL 上触发 NaN。脚本默认启用了：
- `vae.config.force_upcast = True`（等价 webui "Upcast cross attention to float32"）
- `position_ids` 修复（`from_single_file` FP16 加载 bug）
- 如仍报 NaN，加 `--fp32` 强制全精度

## 文件结构

```
stable-diffusion-webui/
├── diffusers-scripts/
│   ├── txt2img.py              ← 文生图主脚本
│   ├── README.md
│   └── output/                 ← 生成图片保存目录
│
└── models/Stable-diffusion/
    ├── v1-5-pruned-emaonly.safetensors     ← SD1.5
    ├── sd_xl_base_1.0.safetensors          ← SDXL（内置 VAE）
    └── sd_xl_base_1.0_0.9vae.safetensors   ← SDXL（0.9 VAE，推荐）
```

## diffusers 与 webui 的关系

- **共用**：PyTorch、CUDA、venv 环境
- **独立**：各自的 SD 推理实现、模型格式、使用方式
- **互不依赖**：webui 不需要 diffusers，diffusers 也不需要 webui
