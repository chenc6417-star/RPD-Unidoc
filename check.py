import json
import random

def merge_and_shuffle_jsonl(file1_path, file2_path, output_path, sample_size=4000):
    # 1. 从第一个文件中随机抽取
    with open(file1_path, 'r', encoding='utf-8') as f1:
        lines1 = f1.readlines()
    
    sampled_from_f1 = random.sample(lines1, min(len(lines1), sample_size))
    
    # 2. 读取第二个文件的全部内容
    with open(file2_path, 'r', encoding='utf-8') as f2:
        lines2 = f2.readlines()
    
    # 3. 合并两个列表
    combined_lines = sampled_from_f1 + lines2
    
    # 4. 全局随机打乱
    random.shuffle(combined_lines)
    
    # 5. 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for line in combined_lines:
            # 确保每行末尾有且只有一个换行符
            outfile.write(line.strip() + '\n')

    print(f"处理完成！总计 {len(combined_lines)} 条数据已打乱并保存至: {output_path}")

# 执行
merge_and_shuffle_jsonl('/home/hk-project-pai00054/wm9835/qwen35/qwen_training/dataset/qwen35_grounding.jsonl', '/home/hk-project-pai00054/wm9835/qwen35/qwen_training/dataset/qwen35_unified_full_v2.jsonl', 'merged_output.jsonl')
