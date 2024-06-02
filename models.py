import cv2
import numpy as np
import timm
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import *


def preprocess_image(img):
    img = img.copy()
    img.thumbnail((640, 640))
    return np.asarray(img)


def color_hist_3d(img_arr):
    pixels = img_arr.reshape(-1, 3)
    hist, _ = np.histogramdd(pixels, bins=10, range=((0, 256), (0, 256), (0, 256)))
    hist /= pixels.shape[0]  # нормализуем
    return hist.flatten()


def get_sift_desc(img_arr):
    img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    _, des = sift.detectAndCompute(img, None)
    return des


def get_visual_words(model, img_arr):
    des = get_sift_desc(img_arr)
    if des is not None:
        classes = model.predict(des)
        hist, _ = np.histogram(classes, SIFT_CLUSTERS, density=True)
    else:
        hist = np.zeros(SIFT_CLUSTERS, dtype='float64')
    return hist


def get_device() -> str:
    """Автоопределения устройства, на котором будут запускаться нейросети (CUDA или CPU)"""
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


class ImageDataset(Dataset):
    def __init__(self, img_files, transform1, transform2):
        self.img_files = img_files
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        image = Image.open(self.img_files[idx]).convert('RGB')
        tensor1 = self.transform1(image)
        tensor2 = self.transform2(image)
        return tensor1, tensor2


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
