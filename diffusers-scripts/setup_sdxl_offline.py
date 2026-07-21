"""
将 SDXL safetensors 转为 diffusers 格式，完全离线。
原理：在本地创建 SDXL 所需的全部配置文件，骗过 from_single_file 的网络检查，
然后从 safetensors 加载权重并保存为完整的 diffusers 格式。

用法：venv/Scripts/python diffusers-scripts/setup_sdxl_offline.py
"""
import json
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CKPT_PATH = PROJECT_DIR / "models" / "Stable-diffusion" / "sd_xl_base_1.0.safetensors"
OUTPUT_DIR = PROJECT_DIR / "models" / "Stable-diffusion" / "sd-xl-diffusers"

FAKE_CACHE = SCRIPT_DIR / "_hf_cache"
FAKE_HASH = "abcdef1234567890abcdef1234567890abcdef12"

# ---------------------------------------------------------------------------
# SDXL 标准配置文件内容
# ---------------------------------------------------------------------------
CONFIGS = {
    "model_index.json": {
        "_class_name": "StableDiffusionXLPipeline",
        "_diffusers_version": "0.31.0",
        "feature_extractor": ["transformers", "CLIPImageProcessor"],
        "scheduler": ["diffusers", "EulerDiscreteScheduler"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "tokenizer_2": ["transformers", "CLIPTokenizer"],
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    },
    # SDXL UNet — 比 SD1.5 大得多的架构
    "unet/config.json": {
        "_class_name": "UNet2DConditionModel",
        "_diffusers_version": "0.31.0",
        "act_fn": "silu",
        "addition_embed_type": "text_time",
        "addition_embed_type_num_heads": 64,
        "addition_time_embed_dim": 256,
        "attention_head_dim": [5, 10, 20],
        "block_out_channels": [320, 640, 1280],
        "center_input_sample": False,
        "class_embed_type": None,
        "cross_attention_dim": 2048,
        "down_block_types": [
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ],
        "downsample_padding": 1,
        "dual_cross_attention": False,
        "encoder_hid_dim": None,
        "encoder_hid_dim_type": None,
        "flip_sin_to_cos": True,
        "freq_shift": 0,
        "in_channels": 4,
        "layers_per_block": 2,
        "mid_block_scale_factor": 1,
        "mid_block_type": None,
        "norm_eps": 1e-05,
        "norm_num_groups": 32,
        "num_class_embeds": None,
        "only_cross_attention": False,
        "out_channels": 4,
        "projection_class_embeddings_input_dim": 2816,
        "resnet_out_scale_factor": 1.0,
        "resnet_skip_time_act": False,
        "resnet_time_scale_shift": "default",
        "sample_size": 128,
        "time_cond_proj_dim": None,
        "time_embedding_dim": None,
        "time_embedding_type": "positional",
        "transformer_layers_per_block": [1, 2, 10],
        "up_block_types": [
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
        ],
        "upcast_attention": False,
        "use_linear_projection": True,
    },
    # VAE — 和 SD1.5 相同
    "vae/config.json": {
        "_class_name": "AutoencoderKL",
        "_diffusers_version": "0.31.0",
        "act_fn": "silu",
        "block_out_channels": [128, 256, 512, 512],
        "down_block_types": [
            "DownEncoderBlock2D", "DownEncoderBlock2D",
            "DownEncoderBlock2D", "DownEncoderBlock2D",
        ],
        "force_upcast": True,
        "in_channels": 3,
        "latent_channels": 4,
        "layers_per_block": 2,
        "norm_num_groups": 32,
        "out_channels": 3,
        "sample_size": 1024,
        "scaling_factor": 0.13025,
        "up_block_types": [
            "UpDecoderBlock2D", "UpDecoderBlock2D",
            "UpDecoderBlock2D", "UpDecoderBlock2D",
        ],
    },
    # text_encoder = CLIP-L (openai/clip-vit-large-patch14)，和 SD1.5 一样
    "text_encoder/config.json": {
        "_class_name": "CLIPTextModel",
        "_diffusers_version": "0.31.0",
        "_name_or_path": "openai/clip-vit-large-patch14",
        "architectures": ["CLIPTextModel"],
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "dropout": 0.0,
        "eos_token_id": 2,
        "hidden_act": "quick_gelu",
        "hidden_size": 768,
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 77,
        "model_type": "clip_text_model",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 1,
        "projection_dim": 768,
        "torch_dtype": "float32",
        "transformers_version": "4.30.2",
        "vocab_size": 49408,
    },
    # text_encoder_2 = OpenCLIP-G/14
    "text_encoder_2/config.json": {
        "_class_name": "CLIPTextModelWithProjection",
        "_diffusers_version": "0.31.0",
        "_name_or_path": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        "architectures": ["CLIPTextModelWithProjection"],
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "dropout": 0.0,
        "eos_token_id": 2,
        "hidden_act": "gelu",
        "hidden_size": 1280,
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 5120,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 77,
        "model_type": "clip_text_model",
        "num_attention_heads": 20,
        "num_hidden_layers": 32,
        "pad_token_id": 1,
        "projection_dim": 1280,
        "torch_dtype": "float32",
        "transformers_version": "4.30.2",
        "vocab_size": 49408,
    },
    # scheduler = EulerDiscrete（SDXL 默认）
    "scheduler/scheduler_config.json": {
        "_class_name": "EulerDiscreteScheduler",
        "_diffusers_version": "0.31.0",
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "clip_sample": False,
        "interpolation_type": "linear",
        "num_train_timesteps": 1000,
        "prediction_type": "epsilon",
        "set_alpha_to_one": False,
        "sigma_max": None,
        "sigma_min": None,
        "skip_prk_steps": True,
        "steps_offset": 1,
        "timestep_spacing": "leading",
        "trained_betas": None,
        "use_karras_sigmas": False,
    },
    "feature_extractor/preprocessor_config.json": {
        "crop_size": 224,
        "do_center_crop": True,
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_resize": True,
        "feature_extractor_type": "CLIPFeatureExtractor",
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "resample": 3,
        "size": 224,
    },
}


def get_tokenizer_files(repo_id: str) -> tuple:
    """从本地 transformers 缓存获取 tokenizer 文件"""
    from transformers import CLIPTokenizer

    tmp_dir = SCRIPT_DIR / "_tmp_tok"
    tmp_dir.mkdir(exist_ok=True)
    tok = CLIPTokenizer.from_pretrained(repo_id, local_files_only=True)
    tok.save_pretrained(str(tmp_dir))

    vocab = (tmp_dir / "vocab.json").read_text()
    merges = (tmp_dir / "merges.txt").read_text()
    tok_config = json.loads((tmp_dir / "tokenizer_config.json").read_text())
    special_map = json.loads((tmp_dir / "special_tokens_map.json").read_text())

    shutil.rmtree(tmp_dir)
    return vocab, merges, tok_config, special_map


def setup_fake_cache():
    """在假缓存目录中创建 SDXL 完整的配置文件结构"""
    # 用 stabilityai 的 repo 路径（SDXL 的正确 repo）
    snapshot = FAKE_CACHE / "models--stabilityai--stable-diffusion-xl-base-1.0" / "snapshots" / FAKE_HASH
    refs_dir = FAKE_CACHE / "models--stabilityai--stable-diffusion-xl-base-1.0" / "refs"

    snapshot.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text(FAKE_HASH)

    # 写入所有 config JSON
    for rel_path, config in CONFIGS.items():
        file_path = snapshot / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(config, f, indent=2)

    # tokenizer（CLIP-L）— 和 SD1.5 一样
    vocab, merges, tok_config, special_map = get_tokenizer_files("openai/clip-vit-large-patch14")
    for subdir in ["tokenizer", "tokenizer_2"]:
        tok_dir = snapshot / subdir
        tok_dir.mkdir(parents=True, exist_ok=True)
        (tok_dir / "vocab.json").write_text(vocab)
        (tok_dir / "merges.txt").write_text(merges)
        with open(tok_dir / "tokenizer_config.json", "w") as f:
            json.dump(tok_config, f, indent=2)
        with open(tok_dir / "special_tokens_map.json", "w") as f:
            json.dump(special_map, f, indent=2)

    print(f"  假缓存已创建: {snapshot}")


def main():
    print(f"原始文件: {CKPT_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"找不到模型: {CKPT_PATH}")

    # 1. 创建假 HF 缓存
    print("\n[1/2] 创建本地配置文件缓存 ...")
    setup_fake_cache()

    # 2. 用假缓存加载 SDXL，然后保存为完整 diffusers 格式
    print("\n[2/2] 从 safetensors 加载并保存为 diffusers 格式 ...")
    import os
    os.environ["HF_HOME"] = str(FAKE_CACHE.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"

    from diffusers import StableDiffusionXLPipeline
    import torch

    pipe = StableDiffusionXLPipeline.from_single_file(
        str(CKPT_PATH),
        torch_dtype=torch.float16,
        cache_dir=str(FAKE_CACHE),
        local_files_only=True,
    )

    pipe.save_pretrained(str(OUTPUT_DIR))
    print(f"\n完成! 模型已保存到: {OUTPUT_DIR}")

    # 清理
    shutil.rmtree(FAKE_CACHE, ignore_errors=True)


if __name__ == "__main__":
    main()
