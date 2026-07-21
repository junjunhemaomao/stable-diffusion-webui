"""
试验：只用手动下载的配置文件（~2MB）+ 已有的 safetensors，能否让 from_single_file 工作。
原理：在 HF 缓存中手动创建 run wayml/stable-diffusion-v1-5 的配置目录，
      让 from_single_file 发现"已有本地缓存"而跳过网络下载。
"""
import json
import os
import shutil
from pathlib import Path

import torch

# ---------- 路径 ----------
PROJECT_DIR = Path(__file__).resolve().parent.parent
CKPT = PROJECT_DIR / "models" / "Stable-diffusion" / "v1-5-pruned-emaonly.safetensors"
DIFFUSERS_DIR = PROJECT_DIR / "models" / "Stable-diffusion" / "sd-v1-5-diffusers"

# 假缓存目录（模拟 HF 缓存结构）
FAKE_HUB = Path(__file__).resolve().parent / "_test_hub"

# ---------- 获取正确的哈希 ----------
# 从 hf-mirror 页面或 model_index.json 可能能拿到。这里从已有的 diffusers 文件推测。
# hf-mirror.com/runwayml/stable-diffusion-v1-5 主页显示的 commit hash。
# 如果不知道，可以随意用一个，只要 refs/main 和 snapshots/<hash> 对应上。
#
# 关键：需要知道正确的 commit hash。可以在镜像站页面找到（通常在文件列表上方）。
# 临时用一个假的试试。

REPO_ID = "runwayml/stable-diffusion-v1-5"
CACHE_REPO = FAKE_HUB / "models--runwayml--stable-diffusion-v1-5"

# 你需要从镜像站页面获取真实的 commit hash，填在这里
# 在 https://hf-mirror.com/runwayml/stable-diffusion-v1-5 页面顶部找
SNAPSHOT_HASH = "???"  # <--- 替换为真实哈希


def setup_cache(hash_value):
    """在假缓存中创建配置文件的副本"""
    snapshot_dir = CACHE_REPO / "snapshots" / hash_value
    refs_dir = CACHE_REPO / "refs"

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)

    (refs_dir / "main").write_text(hash_value)

    # 从已有的 diffusers 目录复制所有配置文件（跳过权重 safetensors）
    for f in DIFFUSERS_DIR.rglob("*"):
        if f.is_file() and f.suffix in (".json", ".txt"):
            rel = f.relative_to(DIFFUSERS_DIR)
            dest = snapshot_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)

    print(f"缓存已创建: {snapshot_dir}")
    print("包含文件:")
    for f in sorted(snapshot_dir.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(snapshot_dir)}")


def main():
    # 如果没有真实哈希，先展示需要的步骤
    if SNAPSHOT_HASH == "???":
        print("=" * 60)
        print("首先需要获取仓库的 commit hash")
        print("浏览器打开:")
        print("  https://hf-mirror.com/runwayml/stable-diffusion-v1-5")
        print("在页面顶部找到 commit hash（例如 1d0c4eb 或完整 SHA），")
        print("填入本脚本的 SNAPSHOT_HASH 变量，然后重新运行。")
        print("=" * 60)
        return

    # 1. 设置假缓存
    print("\n[1/3] 设置本地配置文件缓存 ...")
    setup_cache(SNAPSHOT_HASH)

    # 2. 设置环境（强制离线，使用假缓存）
    os.environ["HF_HOME"] = str(FAKE_HUB.resolve())
    os.environ["HF_HUB_CACHE"] = str(FAKE_HUB.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"

    # 3. 尝试 from_single_file
    print("\n[2/3] 尝试 from_single_file（local_files_only=True）...")
    from diffusers import StableDiffusionPipeline

    try:
        pipe = StableDiffusionPipeline.from_single_file(
            str(CKPT),
            torch_dtype=torch.float16,
            cache_dir=str(FAKE_HUB),
            local_files_only=True,
        )
        pipe.to("cuda")
        print("成功！")

        # 快速测试
        print("\n[3/3] 测试生成 ...")
        img = pipe("a cat", num_inference_steps=5).images[0]
        out = Path(__file__).resolve().parent / "output" / "_test_single_file.png"
        out.parent.mkdir(exist_ok=True)
        img.save(out)
        print(f"已保存: {out}")

    except Exception as e:
        print(f"失败: {e}")
        print("\n如果错误是 'Cannot find an appropriate cached snapshot folder'")
        print("说明哈希值或目录结构不对，检查 SNAPSHOT_HASH 是否和镜像站一致。")

    finally:
        # 清理
        shutil.rmtree(FAKE_HUB, ignore_errors=True)


if __name__ == "__main__":
    main()
