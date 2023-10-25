import joblib
import cfg
from kerasvgg import fusion_match, getmodel, get_embedding
from nca import make_label
import numpy as np

def fusion_test(real_sketch, real_photo, fake_sketch, fake_photo, num):
    nca_pm = joblib.load("C:\\Users\\ludandan\\Desktop\\VGG16\\nca\\nca_pm.pkl")
    nca_sm = joblib.load("C:\\Users\\ludandan\\Desktop\\VGG16\\nca\\nca_sm.pkl")

    real_sketch = nca_sm.transform(real_sketch)
    real_photo = nca_pm.transform(real_photo)
    fake_sketch = nca_sm.transform(fake_sketch)
    fake_photo = nca_pm.transform(fake_photo)

    np.save("C:\\Users\\ludandan\\Desktop\\VGG16\\nca\\nca_real_photo.npy", real_photo)
    np.save("C:\\Users\\ludandan\\Desktop\\VGG16\\nca\\nca_fake_photo.npy", fake_photo)
    np.save("C:\\Users\\ludandan\\Desktop\\VGG16\\nca\\nca_real_sketch.npy", real_sketch)
    np.save("C:\\Users\\ludandan\\Desktop\\VGG16\\nca\\nca_fake_sketch.npy", fake_sketch)

    fusion_match(real_sketch, real_photo, fake_sketch, fake_photo, num)

    return

if __name__=='__main__':
    # pm_model = getmodel(cfg.photo_matching_model_path, 'model')
    # sm_model = getmodel(cfg.sketch_matching_model_path, 'model_2')
    # # get original embeddings
    # real_photo = get_embedding(cfg.photo_label_path, cfg.photo_path, model=pm_model)
    # fake_sketch = get_embedding(cfg.photo_label_path, cfg.photo_path, is_trans=True, target_domain='B', model=sm_model)
    # real_sketch = get_embedding(cfg.sketch_label_path, cfg.sketch_path, model=sm_model)
    # fake_photo = get_embedding(cfg.sketch_label_path, cfg.sketch_path, is_trans=True, target_domain='A', model=pm_model)
    #
    # p, _ = make_label(real_photo, fake_photo)
    # s, _ = make_label(fake_sketch, real_sketch)
    #
    # real_photo = p[0:1194]
    # fake_photo = p[1194:]
    # fake_sketch = s[0:1194]
    # real_sketch = s[1194:]

    real_photo = np.load("C:\\Users\\ludandan\\Desktop\\VGG16\\nca\\nca_real_photo.npy")[500:]
    real_sketch = np.load("C:\\Users\\ludandan\\Desktop\\VGG16\\nca\\nca_real_sketch.npy")[500:]
    fake_photo = np.load("C:\\Users\\ludandan\\Desktop\\VGG16\\nca\\nca_fake_photo.npy")[500:]
    fake_sketch = np.load("C:\\Users\\ludandan\\Desktop\\VGG16\\nca\\nca_fake_sketch.npy")[500:]
    fusion_match(real_sketch, real_photo, fake_sketch, fake_photo, 694)

    #0.6613832853025937 0.5878962536023055 0.8328530259365994