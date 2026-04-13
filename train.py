import sys
import torch
import transformers
from model_patcher import patch_qwen_spatial_residual

print("============== 🧠 正在初始化全参微调引擎 (Frozen Vision) ==============")

# ---------------------------------------------------------
# 模块 1：解决导包兼容性问题
# ---------------------------------------------------------
try:
    from swift.llm import sft_main
except ImportError:
    import runpy
    sft_main = lambda: runpy.run_module('swift.cli.sft', run_name='__main__')

# ---------------------------------------------------------
# 模块 2：终极拦截器 - 注入补丁、劫持优化器、精确冻结
# ---------------------------------------------------------
original_train = transformers.Trainer.train

def custom_train(self, *args, **kwargs):
    print("\n⏳ [拦截阶段 1] 正在动态注入空间残差增强器...")
    if not hasattr(self.model, 'spatial_enhancer'):
        self.model = patch_qwen_spatial_residual(
            self.model, 
            weight_path="/home/hk-project-pai00054/wm9835/qwen35/spatial_enhancer_final.pth"
        )
        
    print("\n🛑 [拦截阶段 2] 执行精确冻结：只锁死 Vision Encoder！")
    trained_params, frozen_params = 0, 0
    
    # 遍历所有参数，执行“反向冻结”逻辑
    for name, param in self.model.named_parameters():
        # Qwen-VL 的视觉特征通常包含 'visual' 字段。
        # 注意：排除掉我们自己挂载的 spatial_enhancer，防止误伤
        if 'visual' in name and 'spatial_enhancer' not in name:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            # LLM 的所有层、lm_head、embed_tokens 以及 spatial_enhancer 全部解冻参与训练
            param.requires_grad = True
            trained_params += param.numel()
            
    print(f"❄️  冻结的视觉参数: {frozen_params / 1e6:.2f} M")
    print(f"🔥 全参训练的 LLM + 增强器参数: {trained_params / 1e6:.2f} M\n")
    
    if trained_params == 0:
        print("❌ 致命错误：没有可训练的参数！")
        sys.exit(1)

    # 🌟 核心新增：劫持优化器，实现差异化学习率保护 LLM
    def custom_create_optimizer():
        print("🔧 [拦截阶段 3] 正在构建差异化优化器 (LLM 维稳，Adapter 激进)...")
        enhancer_params = []
        llm_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'spatial_enhancer' in name:
                enhancer_params.append(param)
            else:
                llm_params.append(param)
                
        # 分组设置学习率
        optimizer_grouped_parameters = [
            {"params": enhancer_params, "lr": 1e-5}, # Adapter：高学习率暴力破局
            {"params": llm_params, "lr": 1e-5}       # LLM 全参：低学习率温柔微调，保护常识
        ]
        
        from torch.optim import AdamW
        self.optimizer = AdamW(optimizer_grouped_parameters, weight_decay=0.01)
        return self.optimizer

    # 替换原本的优化器创建方法
    self.create_optimizer = custom_create_optimizer

    print("🚀 拦截结束，把控制权交还 SWIFT，正式开始全参训练...")
    return original_train(self, *args, **kwargs)

transformers.Trainer.train = custom_train

# ---------------------------------------------------------
# 模块 3：配置参数并启动训练
# ---------------------------------------------------------
def run_training():
    sys.argv = [
        "swift_sft",
        "--model", "/home/hk-project-pai00054/wm9835/qwen35/model_weights/Qwen3.5-0.8B",
        "--dataset", "./qwen35_vqa_short_format.jsonl",
        "--train_type", "full",
        "--torch_dtype", "bfloat16",
        "--num_train_epochs", "1",               
        "--per_device_train_batch_size", "3",
        "--gradient_accumulation_steps", "16",
        "--gradient_checkpointing", "true",
        # 这里的主学习率调低到 1e-5，作为 Scheduler 的全局参考基准
        "--learning_rate", "1e-5",               
        "--max_length", "2048",
        "--save_steps", "100",
        "--save_total_limit", "10",
        "--logging_steps", "10",
        "--output_dir", "./output_qwen35_expert",
        "--system", "You are a document analysis expert. Answer questions based on the provided document."]
    sft_main()

if __name__ == "__main__":
    run_training()