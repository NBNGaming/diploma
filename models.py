import os
import timm
import torch
from PIL import Image
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
import numpy as np


def color_clustering(img):
    img = img.copy()
    img.thumbnail((640, 640))
    img = np.asarray(img)

    img = img.reshape(img.shape[0] * img.shape[1], 3)
    model = KMeans(n_clusters=5)
    model.fit(img)

    counts = np.bincount(model.labels_)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_centroids = model.cluster_centers_[sorted_indices]

    return sorted_centroids


# work in progress
def color_hist_3d(img):
    # Чтобы было везде одинаковое число пикселей
    img = img.resize((640, 640))
    img = np.asarray(img)
    pixels = img.reshape(-1, 3)
    hist, _ = np.histogramdd(pixels, bins=(10, 10, 10))
    hist /= pixels.shape[0]  # нормализуем
    return hist.flatten()


def get_device() -> str:
    """Автоопределения устройства, на котором будут запускаться нейросети (CUDA или CPU)"""
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


class ImageDataset(Dataset):
    def __init__(self, root, transform1, transform2):
        self.root = root
        self.transform1 = transform1
        self.transform2 = transform2
        self.image_files = os.listdir(root)[:10000]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.image_files[idx])
        image = Image.open(path).convert('RGB')
        tensor1 = self.transform1(image)
        tensor2 = self.transform2(image)

        # colors = color_clustering(image)
        # colors /= 255  # нормализуем

        return tensor1, tensor2, path


class MyModel:
    def __init__(self, model_name):
        self.device = get_device()
        # Скачиваем модель
        model = timm.create_model(
            model_name,
            pretrained=True,  # используем предварительно обученную модель
            num_classes=0,  # отключаем классификатор
        )
        model = model.eval()
        self.model = model.to(self.device)  # переносим модель на устройство
        data_config = timm.data.resolve_model_data_config(model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)

    def run(self, data):
        tensor = data.to(self.device)
        # Отключаем вычисление градиентов
        with torch.no_grad():
            output = self.model(tensor)
        output_np = output.detach().cpu().numpy()
        return output_np
