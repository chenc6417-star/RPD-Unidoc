import sys
import torch
import transformers
from model_patcher import patch_qwen_spatial_residual

print(" Qwen3.5-with-Spatial-Enhancer-Adapter Stage 2 Training Script")

try:
    from swift.llm import sft_main
except ImportError:
    import runpy
    sft_main = lambda: runpy.run_module('swift.cli.sft', run_name='__main__')

original_train = transformers.Trainer.train

# stage 2 training: freeze the base model and visual tower, only train LoRA parameters

def custom_train(self, *args, **kwargs):
    
    if getattr(self, "optimizer", None) is not None:
        raise RuntimeError(
            "❌ optimizer already exists. Please ensure that you are not calling Trainer.train() multiple times without reinitializing the Trainer instance."
        )

    if not hasattr(self.model, 'spatial_enhancer'):
        self.model = patch_qwen_spatial_residual(
            self.model, 
            weight_path="/home/hk-project-pai00054/wm9835/qwen35/best_4layer/spatial_enhancer_final.pth",
        )
        
    print("stage2 training: freezing base model and visual tower, only training LoRA parameters...")
    trained_params, frozen_params = 0, 0
    
    # only lora parameter is free！
    def should_train(name: str) -> bool: 
        is_lora = 'lora' in name.lower()
        in_blacklisted_modules = any(
            forbidden_word in name.lower() 
            for forbidden_word in ['visual', 'merger', 'spatial_enhancer']
        )
        
        return is_lora and not in_blacklisted_modules
    for name, param in self.model.named_parameters():
        if should_train(name):
            param.requires_grad = True
            trained_params += param.numel()
        else:
            param.requires_grad = False
            frozen_params += param.numel()
            
    print(f"❄️ freeze (LLM + ViT + Merger + Enhancer): {frozen_params / 1e6:.2f} M")
    print(f"🔥 training (only LoRA parameter): {trained_params / 1e6:.2f} M\n")
    
    if trained_params == 0:
        print("❌ No trainable parameters found. Please check your model and LoRA configuration.")
        sys.exit(1)
        
    print("✅ Model is ready for stage 2 training. Starting training...")
    return original_train(self, *args, **kwargs)

transformers.Trainer.train = custom_train

def run_training():
    sys.argv = [
        "swift_sft",
        "--model", "/home/hk-project-pai00054/wm9835/qwen35/best_4layer/checkpoint-4600",
        "--dataset", "/home/hk-project-pai00054/wm9835/qwen35/merged_output.jsonl", 
        "--train_type", "lora",  
        "--lora_rank", "32",                 
        "--lora_alpha", "64",                
        "--lora_dropout", "0.05",
        "--torch_dtype", "bfloat16",
        "--num_train_epochs", "2",               
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "6",
        "--gradient_checkpointing", "true",
        "--learning_rate", "1e-4",               
        "--max_length", "8192",              
        "--save_steps", "100",
        "--save_total_limit", "10",
        "--logging_steps", "10",
        "--enable_thinking", "false",             
        "--add_non_thinking_prefix", "false",
        "--output_dir", "./qwen35_2b_expert_stage2_lora_6layer",
        "--system", "You are a document analysis expert. You can read text, analyze layouts, and locate elements using precise bounding boxes [ymin, xmin, ymax, xmax]. Answer the user's question directly based on the provided image.",
    ]
    sft_main()

if __name__ == "__main__":
    run_training()