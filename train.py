import os
import sys
import torch
import transformers
from model_patcher import patch_qwen_spatial_residual

print("============== 🧠 正在初始化全参微调引擎 (安全断点续训版) ==============")

try:
    from swift.llm import sft_main
except ImportError:
    import runpy
    sft_main = lambda: runpy.run_module('swift.cli.sft', run_name='__main__')

original_train = transformers.Trainer.train

def custom_train(self, *args, **kwargs):
    print("\n⏳ [拦截阶段 1] 正在动态注入空间残差增强器...")
    
    if not hasattr(self.model, 'spatial_enhancer'):
        # ⚠️ 核心修改 1：传入 None，防止旧的 .pth 覆盖 Checkpoint！
        self.model = patch_qwen_spatial_residual(
            self.model, 
            weight_path=None 
        )
        
        # ⚠️ 核心修改 2：手动从 checkpoint 捞回 1300 步的增强器权重
        model_arg_idx = sys.argv.index("--model")
        checkpoint_dir = sys.argv[model_arg_idx + 1]
        print(f"🔍 正在尝试从 {checkpoint_dir} 捞回已训练的增强器权重...")
        try:
            from safetensors.torch import load_file
            ckpt_path = os.path.join(checkpoint_dir, "model.safetensors")
            if os.path.exists(ckpt_path):
                state_dict = load_file(ckpt_path)
                enhancer_dict = {
                    k.split('spatial_enhancer.')[-1]: v 
                    for k, v in state_dict.items() 
                    if 'spatial_enhancer' in k
                }
                if enhancer_dict:
                    self.model.spatial_enhancer.load_state_dict(enhancer_dict)
                    print(f"✅ 成功恢复 enhancer 权重！共 {len(enhancer_dict)} 个张量。")
                else:
                    print("⚠️ 未在 checkpoint 中找到 enhancer 权重。")
            else:
                print(f"⚠️ 未找到单体 model.safetensors 文件。")
        except Exception as e:
            print(f"❌ 恢复 enhancer 权重失败: {e}")
            
    print("\n🛑 [拦截阶段 2] 执行精确冻结：只锁死 Vision Encoder！")
    trained_params, frozen_params = 0, 0
    
    for name, param in self.model.named_parameters():
        if 'visual' in name and 'spatial_enhancer' not in name:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            param.requires_grad = True
            trained_params += param.numel()
            
    print(f"❄️  冻结的视觉参数: {frozen_params / 1e6:.2f} M")
    print(f"🔥 全参训练的参数: {trained_params / 1e6:.2f} M\n")
    
    if trained_params == 0:
        sys.exit(1)

    def custom_create_optimizer():
        print("🔧 [拦截阶段 3] 正在构建差异化优化器...")
        enhancer_params = []
        llm_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'spatial_enhancer' in name:
                enhancer_params.append(param)
            else:
                llm_params.append(param)
                
        optimizer_grouped_parameters = [
            {"params": enhancer_params, "lr": 1e-5}, 
            {"params": llm_params, "lr": 1e-5}       
        ]
        
        from torch.optim import AdamW
        self.optimizer = AdamW(optimizer_grouped_parameters, weight_decay=0.01)
        return self.optimizer

    self.create_optimizer = custom_create_optimizer

    print("🚀 拦截结束，正式开始断点续训...")
    return original_train(self, *args, **kwargs)

transformers.Trainer.train = custom_train

def run_training():
    # 统一定义 checkpoint 路径
    checkpoint_path = "/home/hk-project-pai00054/wm9835/qwen35/output_qwen35_expert/v12-20260412-134855/checkpoint-1300"
    
    sys.argv = [
        "swift_sft",
        "--model", checkpoint_path, # 👈 1. 这里依然填 checkpoint 路径，用于加载权重
        "--dataset", "/home/hk-project-pai00054/wm9835/qwen35/qwen35_vqa_short_format_shuffled.jsonl",
        "--train_type", "full",
        "--torch_dtype", "bfloat16",
        "--num_train_epochs", "1",               
        "--per_device_train_batch_size", "3",
        "--gradient_accumulation_steps", "16",
        "--gradient_checkpointing", "true",
        "--learning_rate", "1e-6",               
        "--max_length", "2048",
        "--save_steps", "60",
        "--save_total_limit", "10",
        "--logging_steps", "10",
        "--output_dir", "./output_qwen35_expert",
        "--system", "You are a document analysis expert. Answer questions based on the provided document."
    ]
    sft_main()

if __name__ == "__main__":
    run_training()