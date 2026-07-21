"""
txt2img — 使用 HuggingFace Diffusers 文生图
用法:
    venv/Scripts/python diffusers-scripts/txt2img.py
    venv/Scripts/python diffusers-scripts/txt2img.py --prompt "..." --steps 30 --cfg 7.5
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

# ---------------------------------------------------------------------------
# HuggingFace 镜像（国内加速下载）
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"

# 本地已有的模型
LOCAL_DIFFUSERS = PROJECT_DIR / "models" / "Stable-diffusion" / "sd-v1-5-diffusers"
LOCAL_SD15_CKPT = PROJECT_DIR / "models" / "Stable-diffusion" / "v1-5-pruned-emaonly.safetensors"
LOCAL_SDXL_CKPT = PROJECT_DIR / "models" / "Stable-diffusion" / "sd_xl_base_1.0.safetensors"

# 自动检测默认模型（优先 diffusers 格式 > safetensors 单文件 > 在线）
if LOCAL_DIFFUSERS.exists():
    DEFAULT_MODEL = str(LOCAL_DIFFUSERS)
    MODEL_TYPE = "sd15"
elif LOCAL_SD15_CKPT.exists():
    DEFAULT_MODEL = str(LOCAL_SD15_CKPT)
    MODEL_TYPE = "sd15"
elif LOCAL_SDXL_CKPT.exists():
    DEFAULT_MODEL = str(LOCAL_SDXL_CKPT)
    MODEL_TYPE = "sdxl"
else:
    DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"
    MODEL_TYPE = "sd15"

# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Diffusers 文生图")
parser.add_argument("--model", default=DEFAULT_MODEL, help="模型 ID、本地 safetensors 路径、或 HuggingFace repo")
parser.add_argument("--type", default=MODEL_TYPE, choices=["sd15", "sdxl"], help="模型类型")
parser.add_argument("--prompt", default="a photo of an astronaut riding a horse on mars, highly detailed", help="正向提示词")
parser.add_argument("--negative", default="ugly, blurry, low quality, distorted, deformed", help="反向提示词")
parser.add_argument("--steps", type=int, default=25, help="推理步数 (默认 25)")
parser.add_argument("--cfg", type=float, default=7.5, help="CFG scale (默认 7.5)")
parser.add_argument("--seed", type=int, default=-1, help="随机种子，-1 为随机")
parser.add_argument("--W", type=int, default=512, help="图像宽度")
parser.add_argument("--H", type=int, default=512, help="图像高度")
parser.add_argument("--batch", type=int, default=1, help="一次生成几张")
parser.add_argument("--fp16", action="store_true", default=True, help="使用 FP16 节省显存")
parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="使用 FP32")
parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
args = parser.parse_args()

model_path = args.model

# ---------------------------------------------------------------------------
# 设备 & 精度
# ---------------------------------------------------------------------------
if args.cpu:
    device = "cpu"
    dtype = torch.float32
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (args.fp16 and device == "cuda") else torch.float32

print(f"设备: {device}  |  精度: {'FP16' if dtype == torch.float16 else 'FP32'}")
if device == "cuda":
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"显存: {mem:.1f} GB")

# ---------------------------------------------------------------------------
# 加载模型
# ---------------------------------------------------------------------------
model_is_file = model_path.endswith((".safetensors", ".ckpt"))

print(f"\n加载模型: {model_path}")
print(f"类型: {'单文件 safetensors/ckpt' if model_is_file else 'HuggingFace diffusers 格式'}")
t0 = time.time()

if args.type == "sdxl":
    Pipeline = StableDiffusionXLPipeline
else:
    Pipeline = StableDiffusionPipeline

if model_is_file:
    # 从单文件加载（本地 .safetensors / .ckpt）
    pipe = Pipeline.from_single_file(
        model_path,
        torch_dtype=dtype,
    )
else:
    # 从 HuggingFace Hub 或本地 diffusers 目录加载
    # （设置了 HF_ENDPOINT=https://hf-mirror.com 国内也能下）
    pipe = Pipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        safety_checker=None,
    )

pipe.to(device)

# 内存优化
if device == "cuda":
    pipe.enable_attention_slicing()

print(f"加载完成，耗时 {time.time() - t0:.1f}s")

# ---------------------------------------------------------------------------
# 种子
# ---------------------------------------------------------------------------
seed = args.seed if args.seed != -1 else torch.seed()
generator = torch.Generator(device=device).manual_seed(seed)
print(f"种子: {seed}")

# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------
print(f"\n提示词: {args.prompt}")
print(f"反向词: {args.negative}")
print(f"步数: {args.steps}  |  CFG: {args.cfg}  |  尺寸: {args.W}x{args.H}")

t0 = time.time()
with torch.autocast(device_type=device if device == "cuda" else "cpu"):
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        width=args.W,
        height=args.H,
        generator=generator,
        num_images_per_prompt=args.batch,
    )
elapsed = time.time() - t0
print(f"生成完成，耗时 {elapsed:.1f}s  ({elapsed / max(args.batch, 1):.1f}s/张)")

# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------
OUTPUT_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for i, img in enumerate(result.images):
    name = f"{timestamp}_seed{seed}_{i:02d}.png"
    path = OUTPUT_DIR / name
    img.save(path)
    print(f"已保存: {path}")

print("\n完成!")
