import os
import pickle

import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import MyModel, ImageDataset
from config import *


if __name__ == '__main__':
    print('Загрузка моделей...')
    eff_model = MyModel('tf_efficientnetv2_xl.in21k')
    vit_model = MyModel('vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k')
    dataset = ImageDataset('dataset', eff_model.transform, vit_model.transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_embeddings_eff = []
    all_embeddings_vit = []
    path_list = []

    for images_eff, images_vit, paths in tqdm(dataloader, desc='Индексирование изображений'):
        embeddings_eff = eff_model.run(images_eff)
        embeddings_vit = vit_model.run(images_vit)
        all_embeddings_eff.append(embeddings_eff)
        all_embeddings_vit.append(embeddings_vit)
        path_list += paths

    all_embeddings_eff = np.vstack(all_embeddings_eff)
    all_embeddings_vit = np.vstack(all_embeddings_vit) * VIT_WEIGHT

    print('Сохранение базы данных...')
    if not os.path.exists('db'):
        os.makedirs('db')

    pca_eff = PCA(n_components=min(830, all_embeddings_eff.shape[0]))
    all_embeddings_eff = pca_eff.fit_transform(all_embeddings_eff) * EFF_WEIGHT
    with open('db/pca_eff.pickle', 'wb') as f:
        pickle.dump(pca_eff, f, protocol=5)

    all_embeddings = np.hstack((all_embeddings_eff, all_embeddings_vit))
    np.save('db/embeddings', all_embeddings)

    with open('db/paths.txt', 'w') as f:
        f.write('\n'.join(path_list))
    print('Готово!')
