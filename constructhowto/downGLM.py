from huggingface_hub import snapshot_download

# 指定模型 ID
repo_id = "zai-org/chatglm2-6b"
# 指定本地保存路径（建议用绝对路径或相对于代码仓库的路径）
local_dir = "./model"

print("正在开始下载模型权重，请保持网络连接...")
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True  # 支持断点续传，如果断了重新运行即可
)
print("下载完成！")