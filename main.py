import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# ================= 配置区域 =================
# 这里填你本地模型文件夹的路径
# 如果你的模型在 model_weights 文件夹下，就不用改
LOCAL_MODEL_PATH = "./model_weights"


def load_local_model():
    """
    加载本地模型和分词器
    """
    print(f"正在从 {LOCAL_MODEL_PATH} 加载模型...")

    # 检测设备：优先使用 NVIDIA 显卡 (cuda) 或 Mac 显卡 (mps)，否则用 CPU
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备: {device.upper()}")

    try:
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)

        # 加载模型
        # device_map="auto" 会自动把模型分配到显卡或内存中最合适的位置
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            torch_dtype="auto",
            device_map="auto" if device != "cpu" else None,
            local_files_only=True
        )

        if device == "cpu":
            model = model.to("cpu")  # 显式转到 CPU

        print("模型加载成功！")
        return model, tokenizer, device

    except Exception as e:
        print(f"加载失败，请检查路径是否正确。\n错误信息: {e}")
        return None, None, None


def chat(model, tokenizer, device):
    """
    简单的命令行对话循环
    """
    print("\n--- AI 助手已启动 (输入 'quit' 退出) ---")

    while True:
        user_input = input("\n你: ")
        if user_input.lower() in ["quit", "exit", "退出"]:
            break

        # 简单的 Prompt 模板 (根据模型不同可能需要调整，这里用最通用的)
        messages = [
            {"role": "user", "content": user_input}
        ]

        # 应用聊天模板 (如果模型支持 chat template)
        try:
            text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            # 如果不支持模板，直接用原始文本
            text_input = user_input

        # 编码输入
        model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

        # 生成回复
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,  # 最大生成长度
            do_sample=True,  # 随机采样，让回答更灵动
            temperature=0.7  # 创造性 (0-1)
        )

        # 解码输出 (只取新生成的部分)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"AI: {response}")


if __name__ == "__main__":
    # 1. 检查路径是否存在
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"错误：找不到路径 '{LOCAL_MODEL_PATH}'")
        print("请确保你已经下载了模型，并将其解压到了项目文件夹中。")
    else:
        # 2. 加载模型
        model, tokenizer, device = load_local_model()

        # 3. 开始对话
        if model:
            chat(model, tokenizer, device)