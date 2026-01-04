import json
import docx
import os


def convert_docx_to_finetune_jsonl(input_file, output_file):
    """
    读取指定格式的docx文件 (高分句 \\n --- 低分句)，
    转换为大模型微调所需的 JSONL 格式。
    """

    # 1. 读取 Word 文档
    doc = docx.Document(input_file)

    # 用于存储提取的句子对
    pairs = []

    current_high = None
    current_low_buffer = []

    # 2. 遍历文档中的每一段
    for para in doc.paragraphs:
        text = para.text.strip()

        # 跳过空行
        if not text:
            continue

        # 识别低分句（以 --- 开头）
        if text.startswith('---'):
            # 去掉开头的 "--- " 并清理空白
            cleaned_low = text.replace('---', '', 1).strip()
            if cleaned_low:
                current_low_buffer.append(cleaned_low)

        # 识别高分句（不以 --- 开头，且不是空行）
        else:
            # 如果之前已经暂存了一组 (High, Low)，先保存上一组
            if current_high and current_low_buffer:
                # 有些低分句可能跨行，将它们合并
                full_low_text = " ".join(current_low_buffer)
                pairs.append({
                    "high": current_high,
                    "low": full_low_text
                })
                # 重置低分句缓存
                current_low_buffer = []

            # 更新当前的高分句
            current_high = text

    # 处理文件末尾的最后一组数据
    if current_high and current_low_buffer:
        full_low_text = " ".join(current_low_buffer)
        pairs.append({
            "high": current_high,
            "low": full_low_text
        })

    # 3. 构建微调数据格式 (Chat Format)
    # System Prompt 定义了模型的角色
    system_prompt = (
        "You are a professional IELTS writing coach. "
        "Your task is to rewrite the user's Band 5 (low score) sentence "
        "into a Band 7+ (high score) version, improving vocabulary, "
        "grammar, and sentence structure while maintaining the original meaning."
    )

    jsonl_data = []
    for p in pairs:
        entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": p['low']},  # 输入：低分句
                {"role": "assistant", "content": p['high']}  # 输出：高分句
            ]
        }
        jsonl_data.append(entry)

    # 4. 写入 JSONL 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in jsonl_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    print(f"转换完成！共处理 {len(jsonl_data)} 组数据。")
    print(f"文件已保存为: {output_file}")


# --- 使用示例 ---
# 请将文件名替换为你实际的文件路径
input_docx = r"C:\Users\14561\Desktop\优秀作文.docx"
output_jsonl = "ielts_finetune_data.jsonl"

# 检查文件是否存在再运行
if os.path.exists(input_docx):
    convert_docx_to_finetune_jsonl(input_docx, output_jsonl)
else:
    print(f"未找到文件: {input_docx}，请确保文件在当前目录下或提供绝对路径。")