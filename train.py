import os
import pickle
from multiprocessing import Pool

import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from models import MyModel, ImageDataset, preprocess_image, color_hist_3d, get_sift_desc, get_visual_words

kmeans = None


def _img_hist(path):
    image = Image.open(path).convert('RGB')
    image = preprocess_image(image)
    colors = color_hist_3d(image)
    return colors


def calc_all_colors(paths):
    with Pool(4) as p:
        all_colors = tuple(tqdm(
            p.imap(_img_hist, paths),
            total=len(paths),
            desc='Вычисление цветовых гистограмм'
        ))
    all_colors = np.asarray(all_colors)
    np.save('db/colors', all_colors)


def _extract_sift(path):
    image = Image.open(path).convert('RGB')
    image = preprocess_image(image)
    des = get_sift_desc(image)
    return des


def calc_kmeans(paths):
    model = MiniBatchKMeans(n_clusters=SIFT_CLUSTERS, n_init='auto')
    chunk_size = 1000
    chunks = [paths[i:i + chunk_size] for i in range(0, len(paths), chunk_size)]
    for chunk in tqdm(chunks, desc='Кластеризация дескрипторов'):
        with Pool(4) as p:
            result = p.map(_extract_sift, chunk)
        des_list = np.vstack([des for des in result if des is not None])
        model.partial_fit(des_list)
    with open('db/sift_model.pickle', 'wb') as f:
        pickle.dump(model, f, protocol=5)
    return model


def _model_init(model):
    global kmeans
    kmeans = model


def _img_visual_words(path):
    image = Image.open(path).convert('RGB')
    image = preprocess_image(image)
    hist = get_visual_words(kmeans, image)
    return hist


def calc_visual_words(model, paths):
    with Pool(4, initializer=_model_init, initargs=(model,)) as p:
        all_sift = tuple(tqdm(
            p.imap(_img_visual_words, paths),
            total=len(paths),
            desc='Построение BOVW'
        ))
    all_sift = np.asarray(all_sift)
    np.save('db/sift', all_sift)


def calc_all_embeddings(paths):
    print('Загрузка моделей...')
    eff_model = MyModel('tf_efficientnetv2_xl.in21k')
    vit_model = MyModel('vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k')
    dataset = ImageDataset(paths, eff_model.transform, vit_model.transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_embeddings_eff = []
    all_embeddings_vit = []

    for images_eff, images_vit in tqdm(dataloader, desc='Обработка нейросетями'):
        embeddings_eff = eff_model.run(images_eff)
        all_embeddings_eff.append(embeddings_eff)
        embeddings_vit = vit_model.run(images_vit)
        all_embeddings_vit.append(embeddings_vit)

    all_embeddings_eff = np.vstack(all_embeddings_eff)
    all_embeddings_vit = np.vstack(all_embeddings_vit)
    np.save('db/embeddings_eff', all_embeddings_eff)
    np.save('db/embeddings_vit', all_embeddings_vit)


if __name__ == '__main__':
    if not os.path.exists('db'):
        os.makedirs('db')
    root = 'dataset'
    path_list = [os.path.join(root, file) for file in os.listdir(root)]
    with open('db/paths.txt', 'w') as f:
        f.write('\n'.join(path_list))

    calc_all_embeddings(path_list)
    # calc_all_colors(path_list)
    sift_model = calc_kmeans(path_list)
    calc_visual_words(sift_model, path_list)

    print('Готово!')
