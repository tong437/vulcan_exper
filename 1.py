import os
# 设置镜像源（中国大陆用户必选）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

# 加载数据集
dataset = load_dataset("flaviagiammarino/vqa-rad")

# 查看数据集结构
print(dataset)
print(dataset['train'].features)

# 保存到本地
dataset.save_to_disk("/root/autodl-pub-RTX4090-hdd-1/datasets/vqa-rad")
print("数据集已保存到本地！")
