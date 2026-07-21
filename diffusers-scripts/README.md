# Diffusers SD1.5 / SDXL 环境说明

## 当前状态

`diffusers-scripts/txt2img.py` **完全离线可用**。两种模型均通过 `from_single_file` 从单文件 safetensors 加载，配置文件首次运行后已缓存到本地 `~/.cache/huggingface/`。

```bash
# SDXL（默认，1024×1024, 30步）
venv/Scripts/python diffusers-scripts/txt2img.py
venv/Scripts/python diffusers-scripts/txt2img.py --prompt "a cat" --steps 30

# SD1.5
venv/Scripts/python diffusers-scripts/txt2img.py --model models/Stable-diffusion/v1-5-pruned-emaonly.safetensors --type sd15

# 50 系显卡 NaN 兜底（FP32 + upcast）
venv/Scripts/python diffusers-scripts/txt2img.py --fp32 --no-upcast
```

## 模型文件

| 文件 | 格式 | 大小 | 加载方式 |
|---|---|---|---|
| `v1-5-pruned-emaonly.safetensors` | 单文件 ckpt | 4.0 GB | `from_single_file` |
| `sd_xl_base_1.0_0.9vae.safetensors` | 单文件 ckpt (0.9VAE) | 6.5 GB | `from_single_file` |
| `sd_xl_base_1.0.safetensors` | 单文件 ckpt | 6.5 GB | `from_single_file` |

**不再需要** diffusers 分体格式目录（`sd-v1-5-diffusers/`、`sd-xl-diffusers/`）。`from_single_file` 首次运行下载 ~2MB 配置文件后即完全离线，不需要额外的 ~10GB 权重副本。

## 换电脑 / 公司离线环境

`~/.cache/huggingface/` 不在 git 仓库里，pull 代码不会带上。首次运行 `from_single_file` 需要下载 ~2MB 配置文件，如果公司网络不通就会卡住。

**方案 A（推荐）：从家里拷贝缓存**

把家里 `C:\Users\<用户名>\.cache\huggingface\hub\` 下这两个目录拷贝到公司同路径：
- `models--runwayml--stable-diffusion-v1-5\`
- `models--stabilityai--stable-diffusion-xl-base-1.0\`

总共不到 5MB，拷完就是 100% 离线。

**方案 B：用 setup 脚本本地生成（无需网络）**

```bash
# SD1.5 — 配置硬编码在脚本里，直接跑
venv/Scripts/python diffusers-scripts/setup_sd15_offline.py
```

SD1.5 的配置文件全部硬编码在脚本里，完全不需要网络。SDXL 的脚本 (`setup_sdxl_offline.py`) 目前还在调试，建议先用方案 A。

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
│   ├── setup_sd15_offline.py   ← SD1.5 离线转换（仅首次配置用）
│   ├── setup_sdxl_offline.py   ← SDXL 离线转换（仅首次配置用）
│   ├── try_single_file.py      ← from_single_file 试验脚本
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
