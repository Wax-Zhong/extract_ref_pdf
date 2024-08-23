#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('mirror013/mxbai-embed-large-v1')
print(model_dir)