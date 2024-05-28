import pickle

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist

from models import MyModel, color_hist_3d
from config import *


@st.cache_data
def load_db():
    with open('db/pca_eff.pickle', 'rb') as f:
        pca_eff = pickle.load(f)
    all_embeddings = np.load('db/embeddings.npy')
    with open('db/paths.txt', mode='r', encoding='utf-8') as f:
        paths = f.read().splitlines()

    nbrs = NearestNeighbors(n_neighbors=10, n_jobs=-1, p=1)
    nbrs.fit(all_embeddings)

    return pca_eff, nbrs, paths


@st.cache_resource
def load_models():
    eff_model = MyModel('tf_efficientnetv2_xl.in21k')
    vit_model = MyModel('vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k')
    return eff_model, vit_model


pca_eff, neighbours, paths = load_db()

# embeddings_eff = np.load('embeddings_eff.npy')
# embeddings_vit = np.load('embeddings_vit.npy')

eff_model, vit_model = load_models()
st.title('Поиск изображений')
uploaded_file = st.file_uploader('Выберите изображение...', type=['png', 'jpg', 'jpeg', 'webp', 'tiff'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image)

    vec_eff = eff_model.run(eff_model.transform(image).unsqueeze(0))
    vec_vit = vit_model.run(vit_model.transform(image).unsqueeze(0)) * VIT_WEIGHT
    vec_eff = pca_eff.transform(vec_eff) * EFF_WEIGHT
    vec = np.hstack((vec_eff, vec_vit))

    st.divider()
    column = st.columns(3)

    indices = neighbours.kneighbors(vec, return_distance=False)[0]
    for i, path_idx in enumerate(indices):
        col_idx = i % 3
        with column[col_idx]:
            st.image(paths[path_idx], caption=paths[path_idx])
