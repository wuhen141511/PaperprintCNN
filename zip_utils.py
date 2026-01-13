import os
import argparse
import zipfile
from pathlib import Path
from typing import Union


def split_zip(input_zip: str, output_dir: str = None, chunk_size: str = "100M") -> None:
    """将zip压缩包分割成若干小份
    
    Args:
        input_zip: 输入zip文件路径
        output_dir: 输出目录路径，默认为输入文件同目录下的split文件夹
        chunk_size: 每个小份的大小，支持K/M/G后缀，默认100M
    """
    if not os.path.exists(input_zip):
        raise FileNotFoundError(f"输入文件不存在: {input_zip}")
    
    if not input_zip.lower().endswith('.zip'):
        raise ValueError("输入文件必须是.zip格式")
    
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(input_zip), "split")
    
    os.makedirs(output_dir, exist_ok=True)
    
    size_bytes = parse_size(chunk_size)
    
    base_name = os.path.basename(input_zip)
    output_base = os.path.splitext(base_name)[0]
    
    with open(input_zip, 'rb') as f:
        chunk_num = 1
        while True:
            chunk_data = f.read(size_bytes)
            if not chunk_data:
                break
            
            chunk_filename = f"{output_base}.zip.{chunk_num:03d}"
            chunk_path = os.path.join(output_dir, chunk_filename)
            
            with open(chunk_path, 'wb') as chunk_file:
                chunk_file.write(chunk_data)
            
            print(f"已创建: {chunk_filename} ({len(chunk_data)} bytes)")
            chunk_num += 1
    
    print(f"\n分割完成! 共分割成 {chunk_num - 1} 个文件")
    print(f"输出目录: {output_dir}")


def merge_zip(input_pattern: str, output_zip: str = None) -> None:
    """将同名的小份zip压缩包合并成完整的zip压缩包
    
    Args:
        input_pattern: 输入文件模式，支持通配符，如 "output_dir/split/*.zip.001"
                        或 "output_dir/split/filename.zip.*"
        output_zip: 输出zip文件路径，默认为第一个输入文件去掉序号后的名称
    """
    import glob
    
    if '*' in input_pattern:
        files = sorted(glob.glob(input_pattern))
    else:
        if os.path.isdir(input_pattern):
            files = sorted(glob.glob(os.path.join(input_pattern, "*.zip.*")))
        else:
            files = [input_pattern]
    
    if not files:
        raise FileNotFoundError(f"未找到匹配的文件: {input_pattern}")
    
    if not output_zip:
        base_name = os.path.basename(files[0])
        if '.zip.' in base_name:
            output_zip = base_name.split('.zip.')[0] + '.zip'
            output_zip = os.path.join(os.path.dirname(files[0]), output_zip)
        else:
            output_zip = os.path.join(os.path.dirname(files[0]), "merged.zip")
    
    with open(output_zip, 'wb') as outfile:
        for i, chunk_file in enumerate(files, 1):
            with open(chunk_file, 'rb') as infile:
                outfile.write(infile.read())
            print(f"已合并: {os.path.basename(chunk_file)}")
    
    print(f"\n合并完成! 共合并 {len(files)} 个文件")
    print(f"输出文件: {output_zip}")
    
    if zipfile.is_zipfile(output_zip):
        print("✓ 合并后的文件是有效的zip格式")
    else:
        print("⚠ 警告: 合并后的文件可能不是有效的zip格式")


def extract_zip(input_zip: str, output_dir: str = None, password: str = None) -> None:
    """解压zip压缩包
    
    Args:
        input_zip: 输入zip文件路径
        output_dir: 输出目录路径，默认为输入文件同目录下的extract文件夹
        password: zip密码（如果有加密）
    """
    if not os.path.exists(input_zip):
        raise FileNotFoundError(f"输入文件不存在: {input_zip}")
    
    if not zipfile.is_zipfile(input_zip):
        raise ValueError("输入文件不是有效的zip格式")
    
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(input_zip), "extract")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(input_zip, 'r') as zip_ref:
            if password:
                zip_ref.setpassword(password.encode('utf-8'))
            
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            for i, file in enumerate(file_list, 1):
                try:
                    zip_ref.extract(file, output_dir)
                    print(f"[{i}/{total_files}] 已解压: {file}")
                except Exception as e:
                    print(f"[{i}/{total_files}] 解压失败: {file} - {e}")
        
        print(f"\n解压完成! 共解压 {total_files} 个文件")
        print(f"输出目录: {output_dir}")
        
    except RuntimeError as e:
        if "Bad password" in str(e):
            raise ValueError("密码错误或zip文件未加密")
        raise


def parse_size(size_str: str) -> int:
    """解析大小字符串为字节数
    
    Args:
        size_str: 大小字符串，如 "100M", "1G", "500K"
    
    Returns:
        字节数
    """
    size_str = size_str.upper().strip()
    
    if size_str.isdigit():
        return int(size_str)
    
    multipliers = {
        'K': 1024,
        'M': 1024 ** 2,
        'G': 1024 ** 3,
        'T': 1024 ** 4,
    }
    
    for suffix, multiplier in multipliers.items():
        if size_str.endswith(suffix):
            num = size_str[:-1]
            if num.isdigit():
                return int(num) * multiplier
    
    raise ValueError(f"无效的大小格式: {size_str}。支持的格式: 100M, 1G, 500K 等")


def main():
    parser = argparse.ArgumentParser(description="ZIP文件工具：分割、合并、解压")
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    split_parser = subparsers.add_parser('split', help='分割zip文件')
    split_parser.add_argument('input_zip', help='输入zip文件路径')
    split_parser.add_argument('--output', '-o', help='输出目录路径')
    split_parser.add_argument('--size', '-s', default='200M', 
                              help='每个分块的大小，支持K/M/G后缀，默认100M')
    
    merge_parser = subparsers.add_parser('merge', help='合并zip分块文件')
    merge_parser.add_argument('input_pattern', help='输入文件模式，支持通配符或目录')
    merge_parser.add_argument('--output', '-o', help='输出zip文件路径')
    
    extract_parser = subparsers.add_parser('extract', help='解压zip文件')
    extract_parser.add_argument('input_zip', help='输入zip文件路径')
    extract_parser.add_argument('--output', '-o', help='输出目录路径')
    extract_parser.add_argument('--password', '-p', help='zip密码（如果有加密）')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'split':
            split_zip(args.input_zip, args.output, args.size)
        elif args.command == 'merge':
            merge_zip(args.input_pattern, args.output)
        elif args.command == 'extract':
            extract_zip(args.input_zip, args.output, args.password)
    except Exception as e:
        print(f"错误: {e}")
        exit(1)


if __name__ == "__main__":
    main()
