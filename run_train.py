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
    # 此时 SWIFT 已经把模型加载好了。我们直接在这里做手术！
    if not hasattr(self.model, 'spatial_enhancer'):
        # 🌟 核心修改：让它带着之前练好的记忆出生！
        self.model = patch_qwen_spatial_residual(
            self.model, 
            weight_path="/home/hk-project-pai00054/wm9835/qwen35/output_qwen35_expert/spatial_enhancer_final.pth"
        )
        
        
    print("\n🛑 [拦截阶段 2] 训练即将开始，执行精确参数冻结策略！")
    trained_params, frozen_params = 0, 0
    
    # 打印看一眼我们新挂上去的模块参数名
    all_names = [n for n, _ in self.model.named_parameters()]
    enhancer_names = [n for n in all_names if 'spatial_enhancer' in n]
    print(f"DEBUG: 成功挂载的增强器参数: {enhancer_names[:2]}...")

    for name, param in self.model.named_parameters():
        if 'spatial_enhancer' in name:
            param.requires_grad = True
            trained_params += param.numel()
        else:
            param.requires_grad = False
            frozen_params += param.numel()
            
    print(f"❄️  冻结的大模型参数: {frozen_params / 1e6:.2f} M")
    print(f"🔥 训练的增强模块参数: {trained_params / 1e6:.2f} M\n")
    
    # 终极安全锁：如果还是 0，直接报错中止，防止白跑
    if trained_params == 0:
        print("❌ 致命错误：空间增强器注入失败！请检查 model_patcher.py。")
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
        "--model", "/home/hk-project-pai00054/wm9835/qwen35/output_qwen35_expert/v5-20260305-144841_nothinking_version/checkpoint-2086",
        "--dataset", "./qwen35_unified_full_v2.jsonl",
        "--train_type", "full",
        "--torch_dtype", "bfloat16",
        "--num_train_epochs", "5",               
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "16",
        "--gradient_checkpointing", "true",
        "--learning_rate", "1e-4",               
        "--max_length", "4096",
        "--save_steps", "200",
        "--save_total_limit", "10",
        "--logging_steps", "10",
        "--output_dir", "./output_qwen35_expert",
        "--system", "You are a document analysis expert. For each page, output the correct containing bounding boxes [ymin, xmin, ymax, xmax] and corresponding labels, ocr text and alt text for figures or tables in this page."
    ]
    sft_main()

if __name__ == "__main__":
    run_training()