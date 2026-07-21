"""
txt2img — 使用 HuggingFace Diffusers 文生图
用法:
    venv/Scripts/python diffusers-scripts/txt2img.py
    venv/Scripts/python diffusers-scripts/txt2img.py --prompt "..." --steps 30 --W 1024 --H 1024
    venv/Scripts/python diffusers-scripts/txt2img.py --type sdxl --fp32  # 50系显卡NaN兜底
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
MODELS_DIR = PROJECT_DIR / "models" / "Stable-diffusion"

# 本地已有的模型
LOCAL_DIFFUSERS   = MODELS_DIR / "sd-v1-5-diffusers"
LOCAL_SDXL_DIFFUSERS = MODELS_DIR / "sd-xl-diffusers"
LOCAL_SD15_CKPT   = MODELS_DIR / "v1-5-pruned-emaonly.safetensors"
LOCAL_SDXL_CKPT   = MODELS_DIR / "sd_xl_base_1.0.safetensors"
LOCAL_SDXL_VAE9   = MODELS_DIR / "sd_xl_base_1.0_0.9vae.safetensors"  # 修复 NaN 的 0.9 VAE 版本

# 自动检测默认模型
# 优先级：SDXL 0.9VAE > SD1.5 diffusers > SDXL diffusers > SD1.5 ckpt > SDXL ckpt > 在线
if LOCAL_SDXL_VAE9.exists():
    DEFAULT_MODEL = str(LOCAL_SDXL_VAE9)
    MODEL_TYPE = "sdxl"
elif LOCAL_DIFFUSERS.exists():
    DEFAULT_MODEL = str(LOCAL_DIFFUSERS)
    MODEL_TYPE = "sd15"
elif LOCAL_SDXL_DIFFUSERS.exists():
    DEFAULT_MODEL = str(LOCAL_SDXL_DIFFUSERS)
    MODEL_TYPE = "sdxl"
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
# 命令行参数（SD1.5 / SDXL 各自有不同的合理默认值；留 None 等类型确定后再设）
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Diffusers 文生图")
parser.add_argument("--model", default=DEFAULT_MODEL,
                    help="模型 ID、本地 safetensors 路径、或 HuggingFace repo")
parser.add_argument("--type", default=MODEL_TYPE, choices=["sd15", "sdxl"],
                    help="模型类型（自动检测）")
parser.add_argument("--prompt", default="a photo of an astronaut riding a horse on mars, highly detailed",
                    help="正向提示词")
parser.add_argument("--negative", default="ugly, blurry, low quality, distorted, deformed",
                    help="反向提示词")
parser.add_argument("--steps", type=int, default=None,
                    help="推理步数（SD1.5 默认 25，SDXL 默认 30）")
parser.add_argument("--cfg", type=float, default=None,
                    help="CFG scale（SD1.5 默认 7.5，SDXL 默认 7）")
parser.add_argument("--seed", type=int, default=-1, help="随机种子，-1 为随机")
parser.add_argument("--W", type=int, default=None, help="图像宽度（SD1.5 默认 512，SDXL 默认 1024）")
parser.add_argument("--H", type=int, default=None, help="图像高度（SD1.5 默认 512，SDXL 默认 1024）")
parser.add_argument("--batch", type=int, default=1, help="一次生成几张")
parser.add_argument("--fp16", action="store_true", default=True, help="使用 FP16（默认）")
parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="使用 FP32")
parser.add_argument("--fp32", action="store_true",
                    help="强制 FP32 全精度（等价 webui 的 --no-half，50系显卡 NaN 兜底方案）")
parser.add_argument("--upcast", action="store_true", default=True,
                    help="上采样 VAE 到 FP32 + 强制 attention 精度（默认开，修复 50 系 NaN）")
parser.add_argument("--no-upcast", dest="upcast", action="store_false",
                    help="关闭 VAE / attention 上采样")
parser.add_argument("--xformers", action="store_true",
                    help="启用 xformers 内存优化注意力")
parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
args = parser.parse_args()

# ---- 按模型类型补齐默认值 --------------------------------------------------
if args.steps is None:
    args.steps = 30 if args.type == "sdxl" else 25
if args.cfg is None:
    args.cfg = 7.0 if args.type == "sdxl" else 7.5
if args.W is None:
    args.W = 1024 if args.type == "sdxl" else 512
if args.H is None:
    args.H = 1024 if args.type == "sdxl" else 512

model_path = args.model

# ---------------------------------------------------------------------------
# 设备 & 精度
# ---------------------------------------------------------------------------
if args.cpu:
    device = "cpu"
    dtype = torch.float32
elif args.fp32:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    print("[FP32] 强制全精度，50 系显卡 NaN 兜底方案")
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
print(f"模型架构: {'SDXL' if args.type == 'sdxl' else 'SD1.5'}")
t0 = time.time()

if args.type == "sdxl":
    Pipeline = StableDiffusionXLPipeline
else:
    Pipeline = StableDiffusionPipeline

if model_is_file:
    pipe = Pipeline.from_single_file(
        model_path,
        torch_dtype=dtype,
    )
else:
    pipe = Pipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        safety_checker=None,
    )

pipe.to(device)

# ---------------------------------------------------------------------------
# 修复 FP16 加载导致 position_ids 变 Half 的 bug（SDXL from_single_file 专属）
# ---------------------------------------------------------------------------
for _component in ["text_encoder", "text_encoder_2"]:
    if hasattr(pipe, _component) and getattr(pipe, _component) is not None:
        model = getattr(pipe, _component)
        if hasattr(model, "text_model"):
            emb = model.text_model.embeddings
            if hasattr(emb, "position_ids") and emb.position_ids.dtype != torch.long:
                emb.position_ids = emb.position_ids.to(torch.long)
                print(f"[FIX] {_component} position_ids 已从 {emb.position_ids.dtype} 修复为 long")

# ---------------------------------------------------------------------------
# 50 系显卡 NaN 修复（等价 webui 的 "Upcast cross attention layer to float32"）
# ---------------------------------------------------------------------------
if device == "cuda" and args.upcast:
    # ① 强制 VAE 内部 upcast — 修复 SDXL NaN（不改变 VAE 整体 dtype，避免类型不匹配）
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "config"):
        try:
            pipe.vae.config.force_upcast = True
            print("[OK] VAE force_upcast=True（修复 NaN）")
        except Exception:
            pass

    # ② UNet attention slicing — 节省显存
    if dtype == torch.float16:
        try:
            pipe.enable_attention_slicing()
            print("[OK] attention slicing 已启用")
        except Exception:
            pass

    # ③ VAE slicing — 大图时防止显存溢出
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

# ---------------------------------------------------------------------------
# xformers
# ---------------------------------------------------------------------------
if args.xformers and device == "cuda":
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[OK] xformers 已启用")
    except Exception as e:
        print(f"[WARN] xformers 启用失败: {e}")

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
