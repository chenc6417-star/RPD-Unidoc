import torch
from safetensors.torch import load_file

# 指向你训好的 safetensors 文件
path = "/home/hk-project-pai00054/wm9835/qwen35/output_qwen35_expert/v10-20260410-184031/checkpoint-1600/model.safetensors"
state_dict = load_file(path)

# 提取我们自己的残差模块权重
enhancer_weights = {k: v for k, v in state_dict.items() if 'spatial_enhancer' in k}

# 保存为最终的插件权重
torch.save(enhancer_weights, "spatial_enhancer_final.pth")
print("✅ 空间增强插件提取完成！")