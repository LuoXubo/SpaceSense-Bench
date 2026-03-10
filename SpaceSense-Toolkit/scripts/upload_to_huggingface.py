#!/usr/bin/env python3
"""
将原始数据按卫星打包并上传到 HuggingFace

两种使用方式:
  1. 仅打包（网络不好时先本地打包，然后网页手动上传）:
     python upload_to_huggingface.py --raw-data /path/to/raw_data --pack-only --pack-dir ./packed

  2. 打包 + 自动上传:
     python upload_to_huggingface.py --raw-data /path/to/raw_data --repo-id your-username/SpaceSense-136

依赖: pip install huggingface_hub  (仅自动上传时需要)
"""
import os
import sys
import tarfile
import argparse
from pathlib import Path
from tqdm import tqdm


def pack_satellite(satellite_dir, output_dir):
    """将单颗卫星文件夹打包为 tar.gz"""
    sat_name = satellite_dir.name
    parts = sat_name.split('_', 1)
    clean_name = parts[1] if len(parts) > 1 else sat_name

    output_path = output_dir / f"{clean_name}.tar.gz"
    if output_path.exists():
        print(f"  跳过 {clean_name} (已存在)")
        return output_path, True

    print(f"  打包 {clean_name} ...")
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(str(satellite_dir), arcname=clean_name)

    size_gb = output_path.stat().st_size / (1024**3)
    print(f"  完成: {output_path.name} ({size_gb:.2f} GB)")
    return output_path, False


def upload_to_hf(packed_dir, repo_id, repo_type="dataset"):
    """将打包好的文件上传到 HuggingFace"""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("需要安装 huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()

    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)

    tar_files = sorted(packed_dir.glob("*.tar.gz"))
    print(f"\n准备上传 {len(tar_files)} 个文件到 {repo_id}")

    for tar_file in tqdm(tar_files, desc="上传中"):
        path_in_repo = f"raw/{tar_file.name}"
        print(f"  上传 {tar_file.name} -> {path_in_repo}")
        api.upload_file(
            path_or_fileobj=str(tar_file),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
        )

    print(f"\n全部上传完成! https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="按卫星打包原始数据并上传到HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  仅打包:   python upload_to_huggingface.py --raw-data ./raw_data --pack-only --pack-dir ./packed
  打包+上传: python upload_to_huggingface.py --raw-data ./raw_data --repo-id user/SpaceSense-136
        """
    )
    parser.add_argument("--raw-data", type=str, required=True,
                       help="原始数据根目录 (包含 <timestamp>_<satellite>/ 子文件夹)")
    parser.add_argument("--pack-dir", type=str, default="./packed",
                       help="打包输出目录 (默认: ./packed)")
    parser.add_argument("--pack-only", action="store_true",
                       help="仅打包不上传 (可之后网页手动上传)")
    parser.add_argument("--repo-id", type=str, default=None,
                       help="HuggingFace 仓库ID (如 username/SpaceSense-136)")

    args = parser.parse_args()

    if not args.pack_only and not args.repo_id:
        print("错误: 请指定 --repo-id 或使用 --pack-only 仅打包")
        sys.exit(1)

    raw_data_root = Path(args.raw_data)
    pack_dir = Path(args.pack_dir)

    if not raw_data_root.exists():
        print(f"错误: 目录不存在 {raw_data_root}")
        sys.exit(1)

    satellite_dirs = sorted([
        d for d in raw_data_root.iterdir()
        if d.is_dir() and not d.name.startswith('trajectory')
    ])

    print(f"找到 {len(satellite_dirs)} 个卫星文件夹")
    pack_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    for sat_dir in satellite_dirs:
        _, was_skipped = pack_satellite(sat_dir, pack_dir)
        if was_skipped:
            skipped += 1

    total = len(satellite_dirs)
    print(f"\n打包完成: {total - skipped} 个新打包, {skipped} 个跳过")
    print(f"输出目录: {pack_dir}")

    if args.pack_only:
        print("\n已完成打包。可通过以下方式上传:")
        print("  方式1: 在 HuggingFace 网页拖拽上传 .tar.gz 文件")
        print("  方式2: huggingface-cli upload <repo-id> packed/ raw/ --repo-type dataset")
        return

    upload_to_hf(pack_dir, args.repo_id)


if __name__ == "__main__":
    main()
