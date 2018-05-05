import pandas as pd
import os
from PIL import Image
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, \
    GlobalMaxPool2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras import backend as K

from tqdm import tqdm
from collections import Counter

def read_and_resize(filepath, input_shape=(256, 256)):
    im = Image.open((filepath)).convert('RGB')
    im = im.resize(input_shape)
    im_array = np.array(im, dtype="uint8")#[..., ::-1]
    return np.array(im_array / (np.max(im_array)+ 0.001), dtype="float32")

datagen = ImageDataGenerator(
    rotation_range=6,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1)

def augment(im_array):
    im_array = datagen.random_transform(im_array)
    return im_array

def gen(df, batch_size=32, aug=False):
    df = df.sample(frac=1)

    dict_age = {'(0, 2)' : 0,
                '(4, 6)' : 1,
                '(8, 12)' : 2,
                '(15, 20)' : 3,
                '(25, 32)' : 4,
                '(38, 43)' : 5,
                '(48, 53)' : 6,
                '(60, 100)' : 7}

    while True:
        for i, batch in enumerate([df[i:i+batch_size] for i in range(0,df.shape[0],batch_size)]):
            if aug:
                images = np.array([augment(read_and_resize(file_path)) for file_path in batch.path.values])
            else:
                images = np.array([read_and_resize(file_path) for file_path in batch.path.values])


            labels = np.array([dict_age[g] for g in batch.age.values])
            labels = labels[..., np.newaxis]

            yield images, labels


def get_model(n_classes=1):

    base_model = ResNet50(weights='imagenet', include_top=False)

    #for layer in base_model.layers:
    #    layer.trainable = False

    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.5)(x)
    if n_classes == 1:
        x = Dense(n_classes, activation="sigmoid")(x)
    else:
        x = Dense(n_classes, activation="softmax")(x)

    base_model = Model(base_model.input, x, name="base_model")
    if n_classes == 1:
        base_model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer="adam")
    else:
        base_model.compile(loss="sparse_categorical_crossentropy", metrics=['acc'], optimizer="adam")

    return base_model

def create_path(df, base_path):

    df['path'] = df.apply(lambda x: base_path+"aligned/"+x['user_id']+"/landmark_aligned_face.%s.%s"
                                                                      %(x['face_id'], x['original_image']), axis=1)

    return df
def filter_df(df):

    dict_age = {'(0, 2)' : 0,
                '(4, 6)' : 1,
                '(8, 12)' : 2,
                '(15, 20)' : 3,
                '(25, 32)' : 4,
                '(38, 43)' : 5,
                '(48, 53)' : 6,
                '(60, 100)' : 7}


    df['f'] = df.age.apply(lambda x: int(x in dict_age))
    df = df[df.f == 1]
    return df

if __name__ == "__main__":
    base_path = "/media/ml/data_ml/face_age_gender/"

    dict_age = {'(0, 2)' : 0,
                '(4, 6)' : 1,
                '(8, 12)' : 2,
                '(15, 20)' : 3,
                '(25, 32)' : 4,
                '(38, 43)' : 5,
                '(48, 53)' : 6,
                '(60, 100)' : 7}

    bag = 3

    all_indexes = list(range(5))

    accuracies = []

    for test_id in tqdm(all_indexes):

        train_id = [j for j in all_indexes if j!=test_id]
        print(train_id, test_id)

        train_df = pd.concat([pd.read_csv(base_path+"fold_%s_data.txt"%i, sep="\t") for i in train_id])
        test_df = pd.read_csv(base_path+"fold_%s_data.txt"%test_id, sep="\t")

        train_df = filter_df(train_df)
        test_df = filter_df(test_df)

        print(train_df.shape, test_df.shape)

        train_df = create_path(train_df, base_path=base_path)
        test_df = create_path(test_df, base_path=base_path)

        cnt_ave = 0
        predictions = 0

        test_images = np.array([read_and_resize(file_path) for file_path in test_df.path.values])
        test_labels = np.array([dict_age[a] for a in test_df.age.values])

        for k in tqdm(range(bag)):


            tr_tr, tr_val = train_test_split(train_df, test_size=0.1)

            file_path = "baseline_age.h5"

            checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

            early = EarlyStopping(monitor="val_acc", mode="max", patience=10)

            reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=3)

            callbacks_list = [checkpoint, early, reduce_on_plateau]  # early

            model = get_model(n_classes=len(dict_age))
            model.fit_generator(gen(tr_tr, aug=True), validation_data=gen(tr_val), epochs=200, verbose=2, workers=4,
                           callbacks=callbacks_list, steps_per_epoch=50, validation_steps=30)

            model.load_weights(file_path)

            predictions += model.predict(test_images)
            cnt_ave += 1

            test_images = test_images[:, :, ::-1, :]

            predictions += model.predict(test_images)
            cnt_ave += 1

            K.clear_session()

        predictions = predictions/cnt_ave

        predictions = predictions.argmax(axis=-1)

        acc = accuracy_score(test_labels, predictions)

        print("accuracy : %s " %acc)

        accuracies.append(acc)

    print("mean acc : %s (%s) " %  (np.mean(accuracies), np.std(accuracies)))