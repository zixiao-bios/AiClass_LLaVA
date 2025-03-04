import json
from torch.utils.data import Dataset
from PIL import Image
import requests
from io import BytesIO

class MyImageInstructionDataset(Dataset):
    def __init__(self, json_path, transform=None):
        """
        Args:
            json_path (str): 数据集 JSON 文件的路径，文件中每个样本包含 question, ans, url 三个字段。
            transform: 可选，对图像的预处理操作。
        """
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        # 去除异常数据
        self.data = [sample for sample in self.data if sample and sample["question"] and sample["ans"] and sample["url"]]

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        question = sample["question"]
        answer = sample["ans"]
        url = sample["url"]

        # 下载图像
        image = Image.open(requests.get(url, stream=True).raw)
        
        if self.transform:
            image = self.transform(image)
        
        return {"image": image, "question": question, "answer": answer}
