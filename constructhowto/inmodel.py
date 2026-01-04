from transformers import AutoTokenizer, AutoModel
import torch
import sys
import os

os.environ['CUDA_PATH'] = r"F:\Conda\envs\glm_env"
os.environ['PATH'] = r"F:\Conda\envs\glm_env\Library\bin;" + os.environ['PATH']


# 1. 设置为你本地权重的绝对路径
model_path = r"F:\LLM\pythonProject\model"

print("正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("正在加载模型权重到显卡 (这可能需要 1-2 分钟)...")
# 根据你的显存情况选择加载方式：
# 如果显存 > 14GB，用 .half().cuda()
# 如果显存 < 14GB，用 .quantize(4).half().cuda()
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).quantize(4).half().cuda()

# 2. 开启评估模式
model = model.eval()

# 3. 进行第一次对话测试
response, history = model.chat(tokenizer, "你好，检测到你已经成功加载了本地权重！", history=[])
print(f"模型回复：{response}")
