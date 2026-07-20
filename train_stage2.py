import sys
import torch
import transformers
from model_patcher import patch_qwen_spatial_residual

print("============== 🧠 [Stage 2] 正在初始化 LoRA 联合训练引擎 ==============")

# ---------------------------------------------------------
# 模块 1：解决导包兼容性问题
# ---------------------------------------------------------
try:
    from swift.llm import sft_main
except ImportError:
    import runpy
    sft_main = lambda: runpy.run_module('swift.cli.sft', run_name='__main__')

# ---------------------------------------------------------
# 模块 2：终极拦截器 - 挂载 Stage 1 权重并全盘冻结，只放行 LoRA
# ---------------------------------------------------------
original_train = transformers.Trainer.train

def custom_train(self, *args, **kwargs):
    print("\n⏳ [拦截阶段 1] 正在注入已训练的空间残差增强器 (Stage 1 结晶)...")
    
    if getattr(self, "optimizer", None) is not None:
        raise RuntimeError(
            "❌ optimizer 已经创建，请把 patch 提前到 Trainer 创建之前。"
        )

    if not hasattr(self.model, 'spatial_enhancer'):
        self.model = patch_qwen_spatial_residual(
            self.model, 
            weight_path="/home/hk-project-pai00054/wm9835/qwen35/best_4layer/spatial_enhancer_final.pth",
        )
        
    print("\n🛑 [拦截阶段 2] 启动 Stage 2 冻结策略：冻结底座与视觉，仅激活 LoRA！")
    trained_params, frozen_params = 0, 0
    
    # 🌟 核心修改 2：现在的白名单里只有 LoRA，其余全部冻结！
    def should_train(name: str) -> bool:
        # 1. 必须是 LoRA 参数
        is_lora = 'lora' in name.lower()
        
        # 2. 绝对黑名单：视觉塔、对齐层、你的增强器，绝对不能碰！
        # 即使 SWIFT 给它们强加了 LoRA，我们也坚决不给它求导的权力！
        in_blacklisted_modules = any(
            forbidden_word in name.lower() 
            for forbidden_word in ['visual', 'merger', 'spatial_enhancer']
        )
        
        # 只有是 LoRA 且不在黑名单里的（也就是只有 LLM 部分的 LoRA），才放行
        return is_lora and not in_blacklisted_modules

    # 遍历并执行冻结/解冻
    for name, param in self.model.named_parameters():
        if should_train(name):
            param.requires_grad = True
            trained_params += param.numel()
        else:
            param.requires_grad = False
            frozen_params += param.numel()
            
    print(f"❄️ 冻结参数 (LLM + ViT + Merger + Enhancer): {frozen_params / 1e6:.2f} M")
    print(f"🔥 训练参数 (仅限 LoRA 插件): {trained_params / 1e6:.2f} M\n")
    
    if trained_params == 0:
        print("❌ 致命错误：没有找到可训练的 LoRA 参数！请检查 SWIFT 配置。")
        sys.exit(1)
        
    print("🚀 拦截结束，把控制权交还 SWIFT，正式开始指令微调...")
    return original_train(self, *args, **kwargs)

transformers.Trainer.train = custom_train

# ---------------------------------------------------------
# 模块 3：配置参数并启动训练 (专为 OCR/Layout 优化)
# ---------------------------------------------------------
def run_training():
    sys.argv = [
        "swift_sft",
        # 🌟 基础模型：如果你 Stage 1 训练了原生 Merger，请指向 Stage 1 的 Checkpoint 目录
        # 如果 Stage 1 纯靠 Enhancer 输出，这里填最初的 Qwen3.5-2B 路径即可
        "--model", "/home/hk-project-pai00054/wm9835/qwen35/best_4layer/checkpoint-4600",
        "--dataset", "/home/hk-project-pai00054/wm9835/qwen35/merged_output.jsonl", # 换成你包含 OCR/Layout 的数据集
        
        # 🌟 核心切换：通知 SWIFT 使用 LoRA 模式
        "--train_type", "lora",  
        
        # 🌟 LoRA OCR/Layout 专属大火力配置
        "--lora_rank", "32",                 # 提高秩，应对复杂的版面逻辑
        "--lora_alpha", "64",                # 维持 Alpha / Rank = 2 的经典比例
        "--lora_dropout", "0.05",

        "--torch_dtype", "bfloat16",
        "--num_train_epochs", "2",               
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "6",
        "--gradient_checkpointing", "true",
        
        # 🌟 LoRA 需要更大的学习率唤醒
        "--learning_rate", "1e-4",               
        
        # 🌟 关键修复：防止因 OCR 文本过长导致截断崩溃
        "--max_length", "8192",              
        "--save_steps", "100",
        "--save_total_limit", "10",
        "--logging_steps", "10",
        "--enable_thinking", "false",             
        "--add_non_thinking_prefix", "false",
        
        # 🌟 更换输出目录，保护 Stage 1 的心血
        "--output_dir", "./qwen35_2b_expert_stage2_lora_6layer",
        
        # 🌟 针对 OCR/Layout 修改 System Prompt
        "--system", "You are a document analysis expert. You can read text, analyze layouts, and locate elements using precise bounding boxes [ymin, xmin, ymax, xmax]. Answer the user's question directly based on the provided image.",
    ]
    sft_main()

if __name__ == "__main__":
    run_training()