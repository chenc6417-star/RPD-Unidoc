import sys
import torch
import transformers
from model_patcher import patch_qwen_spatial_residual

print("============== 🧠 正在初始化训练引擎 ==============")

# ---------------------------------------------------------
# 模块 1：解决导包兼容性问题
# ---------------------------------------------------------
try:
    from swift.llm import sft_main
except ImportError:
    import runpy
    sft_main = lambda: runpy.run_module('swift.cli.sft', run_name='__main__')

# ---------------------------------------------------------
# 模块 2：终极拦截器 - 在开跑前 1 秒钟强行注入并冻结
# ---------------------------------------------------------
original_train = transformers.Trainer.train

def custom_train(self, *args, **kwargs):
    print("\n⏳ [拦截阶段 1] 正在动态注入空间残差增强器...")
    
    # 防止优化器被提前创建导致新挂载的参数没被收录
    if getattr(self, "optimizer", None) is not None:
        raise RuntimeError(
            "❌ optimizer 已经创建，不能在此时再注入 spatial_enhancer。"
            "请把 patch 提前到 Trainer 创建之前，或强制重建 optimizer。"
        )

    if not hasattr(self.model, 'spatial_enhancer'):
        # 🌟 核心修改：注入你写的残差模块
        self.model = patch_qwen_spatial_residual(
            self.model, 
            weight_path=None,
        )
        
    print("\n🛑 [拦截阶段 2] 训练即将开始，执行精确参数冻结策略！")
    trained_params, frozen_params = 0, 0
    
    # 🌟 核心整合：你的智能筛选函数
    def should_train(name: str) -> bool:
        train_keywords = [
            "spatial_enhancer",      # 你自己外挂的空间模块
            "model.visual.merger",   # Qwen 原生的视觉语言对齐层
        ]
        return any(k in name for k in train_keywords)

    # 遍历并执行冻结/解冻
    for name, param in self.model.named_parameters():
        if should_train(name):
            param.requires_grad = True
            trained_params += param.numel()
        else:
            param.requires_grad = False
            frozen_params += param.numel()
            
    print(f"❄️ 冻结的大模型参数 (LLM + ViT): {frozen_params / 1e6:.2f} M")
    print(f"🔥 联合训练的参数 (Enhancer + Merger): {trained_params / 1e6:.2f} M\n")
    
    # 终极安全锁
    if trained_params == 0:
        print("❌ 致命错误：可训练参数为 0！请检查模块名称。")
        sys.exit(1)
        
    print("🚀 拦截结束，把控制权交还 SWIFT，正式开始训练...")
    return original_train(self, *args, **kwargs)

transformers.Trainer.train = custom_train

# ---------------------------------------------------------
# 模块 3：配置参数并启动训练
# ---------------------------------------------------------
def run_training():
    sys.argv = [
        "swift_sft",
        "--model", "/home/hk-project-pai00054/wm9835/Qwen3.5-2B",
        "--dataset", "/home/hk-project-pai00054/wm9835/qwen35/qwen35_grounding.jsonl",
        "--train_type", "full",  # 保持 full 即可，拦截器会搞定一切
        "--torch_dtype", "bfloat16",
        "--num_train_epochs", "1",               
        "--per_device_train_batch_size", "3",
        "--gradient_accumulation_steps", "8",
        "--gradient_checkpointing", "true",
        "--learning_rate", "1e-4",               
        "--max_length", "1024",
        "--save_steps", "200",
        "--save_total_limit", "10",
        "--logging_steps", "10",
        "--enable_thinking", "false",             
        "--add_non_thinking_prefix", "false",
        "--output_dir", "./output_qwen35_2b_4layer",
        "--system", "You are a document analysis expert. You can read text, analyze layouts, and locate elements using precise bounding boxes [ymin, xmin, ymax, xmax]. Answer the user's question directly based on the provided image.",
    ]
    sft_main()

if __name__ == "__main__":
    run_training()