import sys
import torch
import transformers
from model_patcher import patch_qwen_spatial_residual
import os

print("============== 🧠 正在初始化训练引擎 (终极战役：联合微调 Joint Training) ==============")

# ---------------------------------------------------------
# 模块 1：解决导包兼容性问题
# ---------------------------------------------------------
try:
    from swift.llm import sft_main
except ImportError:
    import runpy
    sft_main = lambda: runpy.run_module('swift.cli.sft', run_name='__main__')

# ---------------------------------------------------------
# 模块 2：终极拦截器 - 联合微调的参数守护者
# ---------------------------------------------------------
original_train = transformers.Trainer.train

def custom_train(self, *args, **kwargs):
    print("\n⏳ [拦截阶段 1] 正在动态注入空间残差增强器 (从零初始化，避免语义污染)...")
    
    if not hasattr(self.model, 'spatial_enhancer'):
        self.model = patch_qwen_spatial_residual(self.model, weight_path=None)

    print("\n🛑 [拦截阶段 2] 训练即将开始，执行【联合微调】参数分配策略！")
    trained_params, frozen_params = 0, 0
    spatial_params, lora_params = 0, 0
    
    for name, param in self.model.named_parameters():
        # 1. 强制放开我们自己的空间增强器
        if 'spatial_enhancer' in name:
            param.requires_grad = True  
            spatial_params += param.numel()
            trained_params += param.numel()
        # 2. 顺应 Swift 框架，保持 LoRA 参数的开启状态
        elif param.requires_grad:
            lora_params += param.numel()
            trained_params += param.numel()
        # 3. 冻结所有底座模型
        else:
            frozen_params += param.numel()
            
    print(f"❄️ 冻结的原生底座参数: {frozen_params / 1e6:.2f} M")
    print(f"🔥 训练的空间增强参数 (眼睛): {spatial_params / 1e6:.2f} M")
    print(f"🔥 训练的语言模型 LoRA (大脑): {lora_params / 1e6:.2f} M")
    print(f"📈 总计可训练参数: {trained_params / 1e6:.2f} M\n")
    
    if trained_params == 0:
        print("❌ 致命错误：没有找到可训练的参数！")
        sys.exit(1)
        
    print("🚀 拦截结束，把控制权交还 SWIFT，正式开始训练...")
    return original_train(self, *args, **kwargs)

transformers.Trainer.train = custom_train

# ---------------------------------------------------------
# 模块 2.5：强行存盘拦截器 (你之前缺失的核心护航代码)
# ---------------------------------------------------------
original_save_checkpoint = transformers.Trainer._save_checkpoint

def custom_save_checkpoint(self, *args, **kwargs):
    # 1. 万能透传：不管底层有几个参数，原封不动地传给原版保存函数
    original_save_checkpoint(self, *args, **kwargs)
    
    # 2. 找到刚刚生成的 checkpoint 文件夹路径
    checkpoint_folder = f"checkpoint-{self.state.global_step}"
    output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
    
    # 3. 🔪 强行把我们的测绘仪权重塞进这个文件夹里！
    if os.path.exists(output_dir) and hasattr(self.model, 'spatial_enhancer'):
        try:
            enhancer_state = self.model.spatial_enhancer.state_dict()
            save_path = os.path.join(output_dir, "spatial_enhancer.pth")
            torch.save(enhancer_state, save_path)
            print(f"\n💾 [强行存盘成功] 测绘仪权重已安全备份至: {save_path}！\n")
        except Exception as e:
            print(f"\n❌ [强行存盘失败] 增强器保存报错: {e}\n")

# 替换框架的底层保存机制
transformers.Trainer._save_checkpoint = custom_save_checkpoint

# ---------------------------------------------------------
# 模块 3：配置联合微调参数并启动
# ---------------------------------------------------------
def run_training():
    sys.argv = [
        "swift_sft",
        "--model", "/home/hk-project-pai00054/wm9835/qwen35/Qwen3.5-2B", 
        "--dataset", "/home/hk-project-pai00054/wm9835/qwen35/qwen_training/dataset/merged_shuffled_dataset.jsonl",
        
        "--train_type", "lora", 
        "--lora_rank", "32",            
        "--lora_alpha", "64",
        
        "--torch_dtype", "bfloat16",
        "--num_train_epochs", "1",           
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "4",
        "--gradient_checkpointing", "true",
        
        "--learning_rate", "1e-4",      
        "--lr_scheduler_type", "cosine",
        "--warmup_ratio", "0.05",       
        
        "--max_length", "4096",
        "--save_steps", "200",
        "--save_total_limit", "10",
        "--logging_steps", "10",
        "--enable_thinking", "false",             
        "--add_non_thinking_prefix", "false",
        "--output_dir", "./output_qwen35_2b_joint_training", 
        "--system", "You are a document analysis expert. You can read text, analyze layouts, and locate elements using precise bounding boxes [ymin, xmin, ymax, xmax]. Answer the user's question directly based on the provided image.",
    ]
    sft_main()

if __name__ == "__main__":
    run_training()