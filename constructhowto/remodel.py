from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import torch
import os

# ================= 路径配置 =================
# 1. 原版模型路径
base_model_path = r"F:\LLM\pythonProject\model"
# 2. 你的 LoRA 路径
lora_path = r"F:\LLM\pythonProject\chatglm2_ielts_lora"
# 3. 输出的新模型路径 (程序会自动创建)
output_path = r"F:\LLM\pythonProject\chatglm2_ielts_merged"
# ===========================================

print("🚀 正在加载原版模型 (CPU模式)...")
# 关键：都在 CPU 上加载，避免显存和量化冲突
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="cpu")

print(f"🔗 正在加载 LoRA: {lora_path} ...")
model = PeftModel.from_pretrained(model, lora_path, device_map="cpu")

print("♻️ 正在将 LoRA 融合进主模型 (Merge)...")
# 这一步把 LoRA 的权重永久加到了原模型里
model = model.merge_and_unload()

print(f"💾 正在保存融合后的新模型到: {output_path} ...")
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print("✅ 恭喜！合并完成！")
print(f"以后请直接加载这个新文件夹：{output_path}")