import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualSpatialEnhancer(nn.Module):
    def __init__(self, visual_dim, merged_dim, num_layers=4):
        super().__init__()
        # 1. 动态层级权重
        self.layer_weights = nn.Parameter(torch.empty(num_layers))
        nn.init.constant_(self.layer_weights, 1.0) # 均等初始化
        
        # 2. 空间特征投射器
        self.spatial_projector = nn.Sequential(
            nn.Linear(visual_dim, merged_dim),
            nn.LayerNorm(merged_dim),
            nn.GELU(),
            nn.Linear(merged_dim, merged_dim)
        )
        
        # 零初始化 (Zero-Init)
        nn.init.zeros_(self.spatial_projector[-1].weight)
        nn.init.zeros_(self.spatial_projector[-1].bias)

    def forward(self, list_of_layer_features, original_merged_tokens):
        # ==========================================
        # 修复 1：形状自适应的加权融合 (支持 2D 和 3D)
        # ==========================================
        # list_of_layer_features 里面的元素可能是 [3432, 1152] (2D)
        # 堆叠后变成 [4, 3432, 1152]
        stacked_feats = torch.stack(list_of_layer_features, dim=0) 
        
        weights = F.softmax(self.layer_weights, dim=0)
        # 动态生成广播维度，例如 [4, 1, 1] 匹配 [4, L, D]
        weight_shape = [len(list_of_layer_features)] + [1] * (stacked_feats.dim() - 1)
        weights = weights.view(*weight_shape) 
        
        # 加权融合 -> [3432, 1152]
        fused_vis_feature = (stacked_feats * weights).sum(dim=0) 
        
        # 维度对齐 -> [3432, 1024] (假设 Merged Dim 是 1024)
        spatial_guidance = self.spatial_projector(fused_vis_feature) 
        
        # ==========================================
        # 修复 2：兼容 2D 的特征对齐 (Sequence Length Alignment)
        # ==========================================
        is_2d = spatial_guidance.dim() == 2
        
        if is_2d:
            # 如果是 2D [L, D]，强行增加一维变成 3D [1, L, D] 以配合插值
            spatial_guidance = spatial_guidance.unsqueeze(0)
            target_seq_len = original_merged_tokens.shape[0] # [L_merged, D]
        else:
            target_seq_len = original_merged_tokens.shape[1] # [B, L_merged, D]

        current_seq_len = spatial_guidance.shape[1]
        
        if current_seq_len != target_seq_len:
            # 转换形状: [B, Seq, Dim] -> [B, Dim, Seq]
            spatial_guidance = spatial_guidance.transpose(1, 2)
            
            # 🚀 核心修复：使用线性插值 (Linear Interpolation) 替换自适应池化
            # 解决超长序列在反向传播时超出 CUDA 共享内存 (Shared Memory) 的 Bug
            spatial_guidance = F.interpolate(
                spatial_guidance, 
                size=target_seq_len, 
                mode='linear', 
                align_corners=False
            )
            
            # 转换回来: [B, Dim, Seq] -> [B, Seq, Dim]
            spatial_guidance = spatial_guidance.transpose(1, 2)
            
        if is_2d:
            # 如果一开始是 2D，用完之后把假 Batch 维度脱掉 -> [L_merged, D]
            spatial_guidance = spatial_guidance.squeeze(0) 
            
        # 残差注入！
        enhanced_merged_tokens = original_merged_tokens + spatial_guidance
        
        return enhanced_merged_tokens

def patch_qwen_spatial_residual(model, weight_path=None):
    """
    无损挂载视觉空间残差增强模块
    """
    try:
        visual_dim = model.model.visual.blocks[0].mlp.linear_fc2.out_features
        merged_dim = model.model.visual.merger.linear_fc2.out_features
    except AttributeError:
        visual_dim = 1024
        merged_dim = 1024
        
    target_layers = [2, 5, 8, 11] 
    
    model.spatial_enhancer = VisualSpatialEnhancer(
        visual_dim=visual_dim, 
        merged_dim=merged_dim, 
        num_layers=len(target_layers)
    ).to(model.device).to(model.dtype)
    
    if weight_path and os.path.exists(weight_path):
        print(f"✅ 正在加载空间增强权重: {weight_path}")
        state_dict = torch.load(weight_path, map_location=model.device)
        
        # ✂️ 把名字里的 'spatial_enhancer.' 前缀剪掉
        clean_state_dict = {k.replace('spatial_enhancer.', ''): v for k, v in state_dict.items()}
        
        model.spatial_enhancer.load_state_dict(clean_state_dict)
        print("✅ 权重参数精确匹配并加载成功！")
    else:
        print("⚠️ 未加载权重，当前增强模块使用零初始化（输出等同于原生模型）。")

    model.hooked_features = {} 

    # 利用闭包为每一层定制一个专属钩子
    def get_hook_fn(layer_idx):
        def hook_fn(module, input, output):
            feature = output[0] if isinstance(output, tuple) else output
            # 无论反向传播重算多少次，强制覆盖这个 key，防内存泄漏
            model.hooked_features[layer_idx] = feature 
        return hook_fn

    for idx in target_layers:
        model.model.visual.blocks[idx].register_forward_hook(get_hook_fn(idx))

    original_merger_forward = model.model.visual.merger.forward

    def new_merger_forward(*args, **kwargs):
        original_merged_tokens = original_merger_forward(*args, **kwargs)
        
        # 只要收集齐了我们需要的几层特征
        if len(model.hooked_features) == len(target_layers):
            # 严格按照层级顺序提取出来
            features_list = [model.hooked_features[i] for i in target_layers]
            
            enhanced_tokens = model.spatial_enhancer(
                features_list, 
                original_merged_tokens
            )
            # 处理完后清空字典
            model.hooked_features.clear() 
            return enhanced_tokens
            
        return original_merged_tokens

    model.model.visual.merger.forward = new_merger_forward
    print("✅ 隐式空间增强补丁打入成功！")
    return model