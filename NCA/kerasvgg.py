import imageio
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.preprocessing import image
import numpy as np
import cv2
from sklearn.preprocessing import normalize
from keras.models import Model
from vggface_finetune import VggFace
from PIL import Image
import base_cyclegan
import vggface_finetune
import cfg


def getmodel(model_path, name):
    #model = VGGFace()
    #model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    #model.load_weights('C:\\Users\\ludandan\\Desktop\\VGG16\\weights\\')
    embedding_model, triplet_model = vggface_finetune.GetModel()
    triplet_model.load_weights(model_path)
    triplet_model.summary()
    model = triplet_model.get_layer(name)
    return model

def get_VGGFace():
    model = VGGFace(model='vgg16', include_top=True, input_shape=(224, 224, 3), pooling='avg')
    model = Model(inputs=model.input, outputs=model.get_layer('fc7').output)
    return model

def get_embedding(label_path, read_path, is_trans=False, target_domain='B', model=None):
    dir = open(label_path)
    lines = dir.readlines()
    lists = []
    for line in lines:
        lists.append(line.split())
    embedding = [0 for i in range(len(lines))]
    if is_trans:
        g_AB, g_BA = base_cyclegan.get_syn_model()
    for i in range(len(lines)):
        if is_trans:
            img = imageio.imread(read_path + str(lists[i][0])).astype(np.float)
            imgs = []
            imgs.append(img)
            imgs = np.array(imgs) / 127.5 - 1.

            if target_domain == 'B':
                img = g_AB.predict(imgs)[0]
            else:
                img = g_BA.predict(imgs)[0]
            img = 127.5 * img + 127.5
            #img = Image.fromarray(img, 'RGB')
            img_array = cv2.resize(img,(224,224))
        else:
            img = image.load_img(read_path+str(lists[i][0]), target_size=(224, 224))
            img_array = image.img_to_array(img)


        # apply VGGFace preprocess_input to the array
        img_array = preprocess_input(img_array, version=2)

        # create samples (incase of one image at once)
        samples = np.expand_dims(img_array, 0)

        # get embedding
        embedding[i] = model.predict(samples)
        # print(embedding[i].shape)  #(1,4096) [[a,b,c,...,z]]
        # print(embedding[i])


    # print(embedding)
    # print(embedding.shape)

    return embedding


def DistanceEuclidean(X, Y):
    X = X.reshape(1, -1)
    Y = Y.reshape(1, -1)
    diff = (normalize(X) - normalize(Y))
    #print(diff)
    return (diff**2).sum()


def match(photo_embedding, sketch_embedding, num):
    n_acc_photo = 0
    n_acc_sketch = 0
    for i in range(num):
        min_l2 = 100000
        label_min = -1
        for j in range(num):

            l2 = np.linalg.norm(photo_embedding[i] - sketch_embedding[j], ord=2, axis=None, keepdims=True)
            #l2 = DistanceEuclidean(photo_embedding[i], sketch_embedding[j])
            if l2 < min_l2:
                min_l2 = l2
                label_min = j
        if label_min == i:
            n_acc_photo  = n_acc_photo + 1

    for i in range(num):
        min_l2 = 100000
        label_min = -1
        for j in range(num):

            l2 = np.linalg.norm(sketch_embedding[i] - photo_embedding[j], ord=2, axis=None, keepdims=True)
            #l2 = DistanceEuclidean(sketch_embedding[i], photo_embedding[j])
            if l2<min_l2:
                min_l2 = l2
                label_min = j
        if label_min == i:
            n_acc_sketch  = n_acc_sketch + 1

    acc_photo = n_acc_photo / num
    acc_sketch = n_acc_sketch / num
    print(n_acc_photo,n_acc_sketch)
    return acc_photo,acc_sketch


def fusion_match(real_sketch, real_photo, fake_sketch, fake_photo, num):
    n_acc_pm = 0
    n_acc_sm = 0
    n_acc_fm = 0
    for i in range(num):
        min_photo_matching = 100000
        min_sketch_matching = 100000
        min_l2 = min_sketch_matching + min_photo_matching
        pm_label = -1
        sm_label = -1
        fm_label = -1
        for j in range(num):
            l2_photo_matching = np.linalg.norm(fake_photo[i] - real_photo[j], ord=2, axis=None, keepdims=True)
            l2_sketch_matching = np.linalg.norm(real_sketch[i] - fake_sketch[j], ord=2, axis=None, keepdims=True)
            if l2_sketch_matching < min_sketch_matching:
                min_sketch_matching = l2_sketch_matching
                sm_label = j
            if l2_photo_matching < min_photo_matching:
                min_photo_matching = l2_photo_matching
                pm_label = j
            if l2_photo_matching + l2_sketch_matching < min_l2:
                min_l2 = l2_photo_matching + l2_sketch_matching
                fm_label = j
        if pm_label == i:
            n_acc_pm = n_acc_pm + 1
        if sm_label == i:
            n_acc_sm = n_acc_sm + 1
        if fm_label == i:
            n_acc_fm = n_acc_fm + 1

    acc_pm = n_acc_pm / num
    acc_sm = n_acc_sm / num
    acc_fm = n_acc_fm / num
    print(n_acc_pm, n_acc_sm, n_acc_fm)
    print(acc_pm, acc_sm, acc_fm)
    return acc_pm, acc_sm, acc_fm


if __name__=='__main__':
    # pm_model = getmodel(cfg.photo_matching_model_path, 'model')
    # sm_model = getmodel(cfg.sketch_matching_model_path, 'model_2')
    # real_photo = get_embedding(cfg.photo_label_path, cfg.photo_path, model=pm_model)
    # fake_sketch = get_embedding(cfg.photo_label_path, cfg.photo_path, is_trans=True, target_domain='B', model=sm_model)
    # real_sketch = get_embedding(cfg.sketch_label_path, cfg.sketch_path, model=sm_model)
    # fake_photo = get_embedding(cfg.sketch_label_path, cfg.sketch_path, is_trans=True, target_domain='A', model=pm_model)
    #
    # # np.save("C:\\Users\\ludandan\\Desktop\\VGG16\\vgg_em\\real_photo.npy", real_photo)
    # # np.save("C:\\Users\\ludandan\\Desktop\\VGG16\\vgg_em\\fake_sketch.npy", fake_sketch)
    # # np.save("C:\\Users\\ludandan\\Desktop\\VGG16\\vgg_em\\real_sketch.npy", real_sketch)
    # # np.save("C:\\Users\\ludandan\\Desktop\\VGG16\\vgg_em\\fake_photo.npy", fake_photo)
    #
    # #acc_photo, acc_sketch = match(photo_embedding, sketch_embedding, 1194)
    # fusion_match(real_sketch, real_photo, fake_sketch, fake_photo, 1194)
    #
    # #0.5879396984924623 0.518425460636516 0.7772194304857621




    #base vggface
    model = get_VGGFace()
    real_photo = get_embedding(cfg.photo_label_path, cfg.photo_path, model=model)
    fake_sketch = get_embedding(cfg.photo_label_path, cfg.photo_path, is_trans=True, target_domain='B', model=model)
    real_sketch = get_embedding(cfg.sketch_label_path, cfg.sketch_path, model=model)
    fake_photo = get_embedding(cfg.sketch_label_path, cfg.sketch_path, is_trans=True, target_domain='A', model=model)
    fusion_match(real_sketch, real_photo, fake_sketch, fake_photo, 1194)
   #0.23115577889447236 0.24958123953098826 0.30067001675041877

