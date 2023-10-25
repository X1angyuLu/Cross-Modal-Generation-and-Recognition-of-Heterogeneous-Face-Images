from sklearn.neighbors import NeighborhoodComponentsAnalysis
import joblib
import cfg
from kerasvgg import get_embedding, match, fusion_match, getmodel
import numpy as np


def make_label(photo_domain, sketch_domain):
    x = []
    y = []
    for i in range(1194):
        x.append(photo_domain[i][0])
        y.append(i)
    for i in range(1194):
        x.append(sketch_domain[i][0])
        y.append(i)
    return x, y


def split(X, y, test_size, begin):
    if test_size + begin <= 1194:
        X_train = X[begin:begin + test_size] + X[1194 + begin:1194 + begin + test_size]
        y_train = y[begin:begin + test_size] + y[1194 + begin:1194 + begin + test_size]
        A_test = X[0:begin] + X[begin + test_size:1194]
        B_test = X[1194:1194 + begin] + X[1194 + begin + test_size:]
    else:
        X_train = X[0:begin + test_size - 1194] + X[begin:1194] + \
                  X[1194:begin + test_size] + X[begin + 1194:]
        y_train = y[0:begin + test_size - 1194] + y[begin:1194] + \
                  y[1194:begin + test_size] + y[begin + 1194:]
        A_test = X[begin + test_size - 1194:begin]
        B_test = X[begin + test_size:begin + 1194]
    return X_train, y_train, A_test, B_test


def train(X, y, epochs=1, interval=100, test_size=500):
    nca = NeighborhoodComponentsAnalysis(n_components=512, random_state=0)
    for epoch in range(epochs):
        X_train, y_train, A_test, B_test = split(X, y, test_size, (interval*epoch)%1194)
        nca.fit(X_train, y_train)

        nca_photo_embedding = nca.transform(A_test)
        nca_sketch_embedding = nca.transform(B_test)
        acc_photo, acc_sketch = match(nca_photo_embedding, nca_sketch_embedding, 694)
        print('epoch:', epoch)
        print(acc_photo, acc_sketch)
    return nca


def fusion_train(real_sketch, real_photo, fake_sketch, fake_photo, test_size=500):
    x_pm, y_pm = make_label(real_photo, fake_photo)
    x_sm, y_sm = make_label(fake_sketch, real_sketch)

    nca_pm = NeighborhoodComponentsAnalysis(n_components=64, random_state=0)
    nca_sm = NeighborhoodComponentsAnalysis(n_components=64, random_state=0)

    pm_train, pm_label, pm_test_real_photo, pm_test_fake_photo = split(x_pm, y_pm, test_size, 0)
    sm_train, sm_label, sm_test_fake_sketch, sm_test_real_sketch = split(x_sm, y_sm, test_size, 0)

    nca_pm.fit(pm_train, pm_label)
    nca_sm.fit(sm_train, sm_label)

    nca_real_photo = nca_pm.transform(pm_test_real_photo)
    nca_fake_photo = nca_pm.transform(pm_test_fake_photo)
    nca_fake_sketch = nca_sm.transform(sm_test_fake_sketch)
    nca_real_sketch = nca_sm.transform(sm_test_real_sketch)

    fusion_match(nca_real_sketch, nca_real_photo, nca_fake_sketch, nca_fake_photo, 1194-test_size)

    # joblib.dump(nca_pm, "C:\\Users\\ludandan\\Desktop\\VGG16\\nca\\nca_pm.pkl")
    # joblib.dump(nca_sm, "C:\\Users\\ludandan\\Desktop\\VGG16\\nca\\nca_sm.pkl")

    return nca_pm, nca_sm


if __name__=='__main__':
    # pm_model = getmodel(cfg.photo_matching_model_path, 'model')
    # sm_model = getmodel(cfg.sketch_matching_model_path, 'model_2')
    # # get original embeddings
    # real_photo = get_embedding(cfg.photo_label_path, cfg.photo_path, model=pm_model)
    # fake_sketch = get_embedding(cfg.photo_label_path, cfg.photo_path, is_trans=True, target_domain='B', model=sm_model)
    # real_sketch = get_embedding(cfg.sketch_label_path, cfg.sketch_path, model=sm_model)
    # fake_photo = get_embedding(cfg.sketch_label_path, cfg.sketch_path, is_trans=True, target_domain='A', model=pm_model)

    real_photo = np.load("C:\\Users\\ludandan\\Desktop\\VGG16\\vgg_em_for_train_nca\\real_photo.npy")
    real_sketch = np.load("C:\\Users\\ludandan\\Desktop\\VGG16\\vgg_em_for_train_nca\\real_sketch.npy")
    fake_photo = np.load("C:\\Users\\ludandan\\Desktop\\VGG16\\vgg_em_for_train_nca\\fake_photo.npy")
    fake_sketch = np.load("C:\\Users\\ludandan\\Desktop\\VGG16\\vgg_em_for_train_nca\\fake_sketch.npy")

    fusion_train(real_sketch, real_photo, fake_sketch, fake_photo, test_size=500)

