import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualSpatialEnhancer(nn.Module):
    def __init__(self, visual_dim, merged_dim, num_layers=4):
        super().__init__()
        concat_dim = visual_dim * num_layers
        self.spatial_projector = nn.Sequential(
            nn.Linear(concat_dim, merged_dim * 2),
            nn.LayerNorm(merged_dim * 2),
            nn.GELU(),
            nn.Linear(merged_dim * 2, merged_dim)
        )
        
        self.gate = nn.Parameter(torch.tensor([1.0]))

    def forward(self, list_of_layer_features, original_merged_tokens, grid_thw=None):
        concat_feats = torch.cat(list_of_layer_features, dim=-1)
        spatial_guidance = self.spatial_projector(concat_feats)
        
        is_2d = spatial_guidance.dim() == 2
        if is_2d:
            spatial_guidance = spatial_guidance.unsqueeze(0)
            target_seq_len = original_merged_tokens.shape[0] 
        else:
            target_seq_len = original_merged_tokens.shape[1] 

        current_seq_len = spatial_guidance.shape[1]
        
        if current_seq_len != target_seq_len:
            success_2d = False
            
            if grid_thw is not None and grid_thw.dim() == 2 and grid_thw.shape[1] == 3:
                pooled_chunks = []
                start_idx = 0
                success_2d = True
                
                for i in range(grid_thw.shape[0]):
                    t = int(grid_thw[i, 0])
                    h = int(grid_thw[i, 1])
                    w = int(grid_thw[i, 2])
                    
                    chunk_len_in = t * h * w
                    h_out, w_out = h // 2, w // 2
                    chunk_len_out = t * h_out * w_out
                    
                    if start_idx + chunk_len_in > current_seq_len:
                        success_2d = False
                        break
                        
                    chunk = spatial_guidance[:, start_idx : start_idx + chunk_len_in, :]
                    B, _, Dim = chunk.shape
                    
                    try:
                        chunk_2d = chunk.view(B * t, h, w, Dim).permute(0, 3, 1, 2).contiguous()
                        chunk_pooled = F.adaptive_avg_pool2d(chunk_2d, (h_out, w_out))
                        chunk_out = chunk_pooled.permute(0, 2, 3, 1).contiguous().view(B, chunk_len_out, Dim)
                        pooled_chunks.append(chunk_out)
                    except Exception as e:
                        success_2d = False
                        break
                        
                    start_idx += chunk_len_in
                if success_2d and start_idx == current_seq_len:
                    spatial_guidance_aligned = torch.cat(pooled_chunks, dim=1)
                    if spatial_guidance_aligned.shape[1] == target_seq_len:
                        spatial_guidance = spatial_guidance_aligned
                    else:
                        success_2d = False
                else:
                    success_2d = False
            
            if not success_2d:
                return original_merged_tokens

        if is_2d:
            spatial_guidance = spatial_guidance.squeeze(0)
        
        enhanced_merged_tokens = original_merged_tokens + self.gate * spatial_guidance
        return enhanced_merged_tokens


def patch_qwen_spatial_residual(model, weight_path=None):
    visual_module = None
    queue = [model]
    while queue:
        curr = queue.pop(0)
        if hasattr(curr, 'blocks') and hasattr(curr, 'merger'):
            visual_module = curr
            break
        for name, child in curr.named_children():
            queue.append(child)

    if visual_module is None:
        raise RuntimeError("error: no vision blocks  merger patch ")

    try:
        visual_dim = visual_module.blocks[0].mlp.linear_fc2.out_features
        merged_dim = visual_module.merger.linear_fc2.out_features
    except AttributeError:
        visual_dim = 768
        merged_dim = 1024
        
    target_layers = [2, 5, 8, 11] 
    print(f"target layers: {target_layers}")
    
    model.spatial_enhancer = VisualSpatialEnhancer(
        visual_dim=visual_dim, 
        merged_dim=merged_dim, 
        num_layers=len(target_layers)
    ).to(model.device).to(model.dtype)
    
    if weight_path and os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location=model.device)
        clean_state_dict = {k.replace('spatial_enhancer.', ''): v for k, v in state_dict.items()}
        model.spatial_enhancer.load_state_dict(clean_state_dict, strict=False)
    else:
        model.hooked_features = {} 

    original_visual_forward = visual_module.forward
    def new_visual_forward(*args, **kwargs):
        grid_thw = kwargs.get('grid_thw', None)
        if grid_thw is None and len(args) >= 2:
            grid_thw = args[1]
        
        model._hooked_grid_thw = grid_thw
        return original_visual_forward(*args, **kwargs)
        
    visual_module.forward = new_visual_forward

    def get_hook_fn(layer_idx):
        def hook_fn(module, input, output):
            feature = output[0] if isinstance(output, tuple) else output
            model.hooked_features[layer_idx] = feature 
        return hook_fn

    for idx in target_layers:
        visual_module.blocks[idx].register_forward_hook(get_hook_fn(idx))

    original_merger_forward = visual_module.merger.forward

    def new_merger_forward(*args, **kwargs):
        original_merged_tokens = original_merger_forward(*args, **kwargs)
        grid_thw = getattr(model, '_hooked_grid_thw', None)

        if len(model.hooked_features) == len(target_layers):
            features_list = [model.hooked_features[i] for i in target_layers]
            
            enhanced_tokens = model.spatial_enhancer(
                features_list, 
                original_merged_tokens,
                grid_thw=grid_thw  
            )
            model.hooked_features.clear() 
            return enhanced_tokens
            
        return original_merged_tokens

    visual_module.merger.forward = new_merger_forward
    print("successfully merge the patcher into the model")
    
    for param in model.spatial_enhancer.parameters():
        param.requires_grad = True
            
    return model
