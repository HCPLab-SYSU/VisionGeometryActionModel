"""
功能，比较两次下载的libero文件是否完全相同
"""

import os
import json
import hashlib
import tensorflow as tf

# Set these variables when comparing two local dataset copies.
PATH1 = os.environ.get("TFRECORD_PATH1", "data/modified_libero_rlds")
PATH2 = os.environ.get("TFRECORD_PATH2", "data/modified_libero_rlds_copy")

def get_file_hash(filepath):
    """计算文件的 MD5 校验和，用于快速跳过相同文件"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def compare_json(p1, p2):
    """逻辑比对 JSON 内容（忽略格式和排序差异）"""
    try:
        with open(p1, 'r', encoding='utf-8') as f1, open(p2, 'r', encoding='utf-8') as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)
            return data1 == data2
    except Exception as e:
        print(f"  [!] JSON 解析错误: {e}")
        return False

def compare_tfrecord(p1, p2):
    """逻辑比对 TFRecord 内容（逐条 Record 对比）"""
    try:
        ds1 = tf.data.TFRecordDataset(p1)
        ds2 = tf.data.TFRecordDataset(p2)
        
        # 逐条对比二进制记录
        for r1, r2 in zip(ds1, ds2):
            if r1.numpy() != r2.numpy():
                return False
        
        # 检查记录总数是否一致
        count1 = sum(1 for _ in ds1)
        count2 = sum(1 for _ in ds2)
        return count1 == count2
    except Exception as e:
        print(f"  [!] TFRecord 解析错误: {e}")
        return False

def main():
    print(f"🚀 开始对比目录...\n路径A: {PATH1}\n路径B: {PATH2}\n")

    # 1. 扫描所有文件
    def scan_files(base_path):
        file_map = {}
        for root, _, files in os.walk(base_path):
            for f in files:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, base_path)
                file_map[rel_path] = full_path
        return file_map

    files1 = scan_files(PATH1)
    files2 = scan_files(PATH2)

    all_rel_paths = set(files1.keys()) | set(files2.keys())
    
    mismatched = []
    only_in_1 = []
    only_in_2 = []

    print(f"🔍 扫描完成，共发现 {len(all_rel_paths)} 个唯一路径。开始逐一比对...\n")

    for rel_path in sorted(all_rel_paths):
        p1 = files1.get(rel_path)
        p2 = files2.get(rel_path)

        # 情况 A: 只有一边有文件
        if not p1:
            only_in_2.append(rel_path)
            print(f"[缺失] 仅在路径B存在: {rel_path}")
            continue
        if not p2:
            only_in_1.append(rel_path)
            print(f"[缺失] 仅在路径A存在: {rel_path}")
            continue

        # 情况 B: 两边都有，先过二进制哈希（提速）
        if get_file_hash(p1) == get_file_hash(p2):
            # 二进制一致，逻辑肯定一致，直接跳过
            continue

        # 情况 C: 二进制不一致，进行逻辑比对
        is_same = False
        if rel_path.endswith('.json'):
            is_same = compare_json(p1, p2)
        elif rel_path.endswith('.tfrecord'):
            is_same = compare_tfrecord(p1, p2)
        else:
            # 其他非特定格式文件，既然 MD5 不同，就认为不同
            is_same = False

        if not is_same:
            mismatched.append(rel_path)
            print(f"[不同] 逻辑内容不一致: {rel_path}")

    # 最终报告
    print("\n" + "="*40)
    print("对比报告摘要:")
    print(f"✅ 完全一致的文件数: {len(all_rel_paths) - len(mismatched) - len(only_in_1) - len(only_in_2)}")
    print(f"❌ 内容逻辑不一致: {len(mismatched)}")
    print(f"📁 仅在 A 中存在: {len(only_in_1)}")
    print(f"📁 仅在 B 中存在: {len(only_in_2)}")
    print("="*40)

if __name__ == "__main__":
    main()
