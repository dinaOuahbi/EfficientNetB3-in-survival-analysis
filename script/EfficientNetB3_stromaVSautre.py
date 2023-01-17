#!/usr/bin/env python3.7

"""
# ---------------------------------------------
# Programme: DenseNet.py
# Auteur EB
# Date 15/11/2021
#
# Construction dun CNN permettant la classification
# des tuiles Stromales vs Tumorales/Normales
# -------------------------------------------------------------
"""

# On importe tous les modules dont on a besoin
import tensorflow as tf
from tensorflow.keras.applications import * #Efficient Net included here
#import tools
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
import keras.backend as K
import numpy as np
import os
import pandas as pd
import sys
from tensorflow.keras.callbacks import LearningRateScheduler
import math
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import DenseNet121
from sklearn.utils import class_weight
import pathlib
import functools
from keras.utils import np_utils
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
import time
start = time.time()

import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121

from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import Model

print("\n\n+++++ Running CNN EfficientNetB3_CNN1+++++\n\n")

##### Tuto : CNN with Tensorflow|Keras for Fashion MNIST dans Kaggle et TensorFlow CNN, Data Augmentation: Prostate Cancer pour l'augmentation des données
##### https://penseeartificielle.fr/tp-reseau-de-neurones-convolutifs/

# On charge les donnees
test_datagen = ImageDataGenerator()
train_datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest")
train_dir = "../TCGA_annot_DB10/Train"
val_dir = "../TCGA_annot_DB10/Val"
test_dir = "../TCGA_annot_DB10/Test"


batch_size = 32
# Générateurs
train_generator = train_datagen.flow_from_directory( # Takes the path to a directory & generates batches of augmented data
    directory=train_dir,
    target_size=(402, 402), # Taille souhaitee
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=42, subset = 'training'
)
val_generator = train_datagen.flow_from_directory(
    directory=val_dir,
    target_size=(402, 402),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=42
)
test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(402, 402),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=42
)


# Les pixels des images doivent etre entre 0 et 1
resize_and_rescale = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Rescaling(1./255)])


# On utilisera la structure du reseau DensetNet pour l'entrainer entierement
base_model = EfficientNetB3(weights="imagenet",include_top=False, input_shape=(402,402,3)) # transfert learning
base_model.trainable = True

# Construction du modele
inputs = tf.keras.Input(shape=(402, 402, 3))
x = resize_and_rescale(inputs)
x = base_model(x, training=True)
x= GlobalAveragePooling2D()(x)
outputs=Dense(3,activation='softmax')(x)
model = Model(inputs, outputs)

# On compile le modele
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                   optimizer=tf.keras.optimizers.Adam(0.00001), metrics = ["accuracy"])

# On calcule le poids de chacune des trois classes
counter = Counter(train_generator.classes)
max_val = float(max(counter.values()))
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}

# Entrainement
history = model.fit(train_generator,
                    epochs=10,class_weight = class_weights,
                    validation_data=val_generator, verbose = 2)

model.save('../model/StromaVSautre_EfficientNetB3')
model.save('../model/StromaVSautre_EfficientNetB3.h5')

hist_df = pd.DataFrame(history.history)
hist_csv_file = '../model/StromaVSautre_EfficientNetB3_history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

output = '../Resultats/StromaVSautre_DB10/'

# Predictions
train_generator = test_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(402, 402),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    seed=42
)
val_generator = test_datagen.flow_from_directory(
    directory=val_dir,
    target_size=(402, 402),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    seed=42
)
test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(402, 402),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    seed=42
)

myName = train_generator.filenames
test_predictions_baseline = model.predict(train_generator)
myPrediction = test_predictions_baseline
myPrediction = pd.DataFrame(myPrediction, index = myName)
myPrediction.to_csv(output+"myPrediction_train.csv", sep=',', encoding='utf-8',
               index=True, header = None)

myName = val_generator.filenames
test_predictions_baseline = model.predict(val_generator)
myPrediction = test_predictions_baseline
myPrediction = pd.DataFrame(myPrediction, index = myName)
myPrediction.to_csv(output+"myPrediction_val.csv", sep=',', encoding='utf-8',
               index=True, header = None)

myName = test_generator.filenames
test_predictions_baseline = model.predict(test_generator)
myPrediction = test_predictions_baseline
myPrediction = pd.DataFrame(myPrediction, index = myName)
myPrediction.to_csv(output+"myPrediction_test.csv", sep=',', encoding='utf-8',
               index=True, header = None)

Besancon_dir="../Besancon_x40_r/"
Besancon_generator = test_datagen.flow_from_directory(
    directory=Besancon_dir,
    target_size=(402, 402),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='input',
    shuffle=False,
    seed=42
)

myName = Besancon_generator.filenames
test_predictions_baseline = model.predict(Besancon_generator)
myPrediction = test_predictions_baseline
myPrediction = pd.DataFrame(myPrediction, index = myName)
myPrediction.to_csv(output+"myPrediction_Besancon.csv", sep=',', encoding='utf-8',
               index=True, header = None)
