from huggingface_hub import snapshot_download

snapshot_download(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", local_dir="TinyLlama_Model", resume_download= True)

print("Download Completed!!!")