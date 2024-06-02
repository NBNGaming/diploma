import pickle

import numpy as np
import streamlit as st
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from config import *
from models import MyModel, preprocess_image, get_sift_desc


@st.cache_data
def load_db():
    with open('db/paths.txt', mode='r', encoding='utf-8') as f:
        paths = f.read().splitlines()

    embeddings_eff = np.load('db/embeddings_eff.npy')
    embeddings_vit = np.load('db/embeddings_vit.npy')

    pca_eff = PCA(n_components=830)
    embeddings_eff_pca = pca_eff.fit_transform(embeddings_eff)
    all_embeddings = np.hstack((embeddings_eff_pca * EFF_WEIGHT, embeddings_vit * VIT_WEIGHT))

    all_sift = np.load('db/sift.npy')
    with open('db/sift_model.pickle', 'rb') as f:
        sift_model = pickle.load(f)

    container = {
        'eff_nbrs': NearestNeighbors(n_neighbors=N_NEIGHBORS, n_jobs=-1, p=1).fit(embeddings_eff),
        'vit_nbrs': NearestNeighbors(n_neighbors=N_NEIGHBORS, n_jobs=-1, p=1).fit(embeddings_vit),
        'my': {
            'nbrs': NearestNeighbors(n_neighbors=N_NEIGHBORS, n_jobs=-1, p=1).fit(all_embeddings),
            'pca_eff': pca_eff,
        },
        'sift': {
            'nbrs': NearestNeighbors(n_neighbors=N_NEIGHBORS, n_jobs=-1, metric='cosine').fit(all_sift),
            'model': sift_model,
        },
        'paths': paths,
    }
    return container


@st.cache_resource
def load_models():
    eff = MyModel('tf_efficientnetv2_xl.in21k')
    vit = MyModel('vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k')
    return eff, vit


db = load_db()
eff_model, vit_model = load_models()

st.title('Поиск изображений')
option = st.selectbox(
    'Модель',
    ('Bag of Visual Words', 'EfficientNetV2', 'Vision Transformer', 'Ensemble'),
    index=None,
    placeholder='Выберите модель...',
)
uploaded_file = st.file_uploader('Выберите изображение...', type=['png', 'jpg', 'jpeg', 'webp', 'tiff'])

if option is not None and uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image)

    vec = None
    neighbours = None
    if option == 'Bag of Visual Words':
        image = preprocess_image(image)
        des = get_sift_desc(image)
        classes = db['sift']['model'].predict(des)
        hist, _ = np.histogram(classes, SIFT_CLUSTERS, density=True)
        vec = hist.reshape(1, -1)
        neighbours = db['sift']['nbrs']
    else:
        vec_eff = None
        vec_vit = None
        if option in ['EfficientNetV2', 'Ensemble']:
            vec_eff = eff_model.run(eff_model.transform(image).unsqueeze(0))
        if option in ['Vision Transformer', 'Ensemble']:
            vec_vit = vit_model.run(vit_model.transform(image).unsqueeze(0))
        if option == 'Ensemble':
            vec_eff_pca = db['my']['pca_eff'].transform(vec_eff)
            vec = np.hstack((vec_eff_pca * EFF_WEIGHT, vec_vit * VIT_WEIGHT))
            neighbours = db['my']['nbrs']
        elif option == 'EfficientNetV2':
            vec = vec_eff
            neighbours = db['eff_nbrs']
        elif option == 'Vision Transformer':
            vec = vec_vit
            neighbours = db['vit_nbrs']

    st.divider()
    column = st.columns(3)

    indices = neighbours.kneighbors(vec, return_distance=False)[0]
    for i, path_idx in enumerate(indices):
        col_idx = i % 3
        with column[col_idx]:
            st.image(db['paths'][path_idx], caption=db['paths'][path_idx])
