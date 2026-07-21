# Diffusers SD1.5 环境说明

## 当前状态

`diffusers-scripts/txt2img.py` **已可以正常运行**，在公司网络下使用手动下载的完整 diffusers-format 模型，100% 离线。

```bash
# 运行方式（在项目根目录）
venv/Scripts/python diffusers-scripts/txt2img.py
venv/Scripts/python diffusers-scripts/txt2img.py --prompt "..." --steps 30
```

## 为什么下载了两份模型

| 文件 | 格式 | 大小 | 谁在用 |
|---|---|---|---|
| `models/Stable-diffusion/v1-5-pruned-emaonly.safetensors` | 单文件 ckpt | 4.2 GB | webui |
| `models/Stable-diffusion/sd-v1-5-diffusers/` | diffusers 分体 | ~5 GB | diffusers 脚本 |

核心原因是 **公司网络从 Python 进程访问 HuggingFace Hub 被墙**。

## 两种加载方式

### 方式 A：from_single_file（理想，但需要网络一次）

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_single_file(
    "models/Stable-diffusion/v1-5-pruned-emaonly.safetensors",
    torch_dtype=torch.float16,
)
pipe.to("cuda")
```

**优点**：只需要一个 safetensors 文件，不占额外空间。

**问题**：`from_single_file` 首次运行时会去 HuggingFace Hub 下载约 2MB 的配置文件（model_index.json、unet/config.json 等）。一旦配置文件缓存到本地 `~/.cache/huggingface/`，之后就完全离线了。

**在家试试**：网络正常的话，只保留 `v1-5-pruned-emaonly.safetensors`，删掉 `sd-v1-5-diffusers/`，运行这句代码，配置文件自动下载一次就够了。

### 方式 B：from_pretrained（当前方案，100% 离线）

```python
pipe = StableDiffusionPipeline.from_pretrained(
    "models/Stable-diffusion/sd-v1-5-diffusers",
    torch_dtype=torch.float16,
    local_files_only=True,
)
pipe.to("cuda")
```

**优点**：完全不依赖网络，所有文件本地齐全。

**缺点**：多占 ~5GB 空间，和 safetensors 内的权重是重复的。

## 文件结构

```
stable-diffusion-webui/
├── venv/                                  ← 共用 Python 环境
│   └── Lib/site-packages/
│       ├── diffusers/          (0.31.0)
│       ├── torch/              (2.1.2+cu121)
│       └── transformers/       (4.30.2)
│
├── diffusers-scripts/                     ← 新增：你的 Python 驱动脚本
│   ├── txt2img.py              ← 文生图脚本
│   ├── setup_sd15_offline.py   ← 转换脚本（未使用，可删）
│   └── output/                 ← 生成图片保存目录
│
└── models/Stable-diffusion/
    ├── v1-5-pruned-emaonly.safetensors    ← webui 用
    ├── sd_xl_base_1.0.safetensors         ← webui 用
    └── sd-v1-5-diffusers/                 ← diffusers 用
        ├── model_index.json
        ├── unet/
        │   ├── config.json
        │   └── diffusion_pytorch_model.safetensors  (3.3 GB)
        ├── vae/
        │   ├── config.json
        │   └── diffusion_pytorch_model.safetensors  (320 MB)
        ├── text_encoder/
        │   ├── config.json
        │   └── model.safetensors                     (470 MB)
        ├── tokenizer/
        │   ├── vocab.json
        │   ├── merges.txt
        │   ├── tokenizer_config.json
        │   └── special_tokens_map.json
        ├── scheduler/scheduler_config.json
        └── feature_extractor/preprocessor_config.json
```

## 在家中网络环境的建议操作

1. 确认网络正常后，试方式 A：
```bash
python -c "
from diffusers import StableDiffusionPipeline
import torch
pipe = StableDiffusionPipeline.from_single_file(
    'models/Stable-diffusion/v1-5-pruned-emaonly.safetensors',
    torch_dtype=torch.float16
)
pipe.to('cuda')
print('OK')
"
```

2. 如果方式 A 成功，`sd-v1-5-diffusers/` 目录可以删掉，节省 5GB。

3. 更新 `diffusers-scripts/txt2img.py` 中的默认模型路径，从方式 B 切换到方式 A。

## diffusers 与 webui 的关系

- **共用**：PyTorch、CUDA、venv 环境
- **独立**：各自的 SD 推理实现、模型格式、使用方式
- **互不依赖**：webui 不需要 diffusers，diffusers 也不需要 webui
