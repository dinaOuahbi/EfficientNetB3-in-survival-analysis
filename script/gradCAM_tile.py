#!/usr/bin/env python3.7
# ---------------------------------------------
# Programme: gradCAM
# Auteur DO
# Date: Aout 2022
#
# Visualisation des parties de l'image que le CNN regarde pour predire la tumeur ou le tissu normal
# https://www.kaggle.com/miklgr500/eda-g2net-efficientnetb1-embeding-grad-cam
# input args : path tile and model
# Modules python utilises
import torch
import numpy as np
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import img_to_array, array_to_img, save_img
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import * #Efficient Net included here
import sys
from pathlib import Path
from tensorflow.keras.applications import DenseNet121
import cv2

'''
LE SCRIPT PREND EN PARAMS :
    IMG Path        : LE CHEMIN VERS L'IMAGE D'INTERET / IMG ARE REQUIRED
    MODEL           : SI ON VEUT LE MODEL DENSENET ==> 'densenet' OU ON VEUT 'EfficientNetB3'  / MODEL ARE REQUIRED !
    ARG VAR         : SI ON VEUT N'ACTIVER QUE LES VARIABLE CLÉS (SURVIVAL ANALYSIS) SUR LE GRADIENT ==> 'key_var' SINON 'None' / NON REQUIRED ARGS
    INDEX CLASS     : 0 ==> normal / 1 ==> tumor / None ==> la classe predite
'''


# On charge l'image sur laquelle on souhaite obtenir gradCAM
img_path = sys.argv[1] #../TCGA_annot_DB11/Train/Tumor/TCGA-2J-AAB8-01Z-00-DX1_7343.jpeg
arg_model = sys.argv[2] # EfficientNetB3 / densenet
try:
    arg_var = sys.argv[3] # key_var / None
except IndexError:
    arg_var = None

try:
    pred_index = sys.argv[4] # 1 / 0
except IndexError:
    pred_index = None

#img_path="../TCGA_annot_DB11/Train/Normal/TCGA-2L-AAQA-01Z-00-DX1_11185.jpeg"
img = keras.preprocessing.image.load_img(img_path)
array = keras.preprocessing.image.img_to_array(img)
array = array/255 # ATTENTION : pour que le modele fonctionne correctement, il faut mettre les pixels entre 0 et 1
print('=========> Mon image a une forme de {}'.format(array.shape))
if arg_model == 'EfficientNetB3':
    base = EfficientNetB3(weights="imagenet",include_top=False, input_shape=(402,402,3)) # transfert learning
else:
    base = DenseNet121(input_shape= (402,402,3),include_top=False,weights='imagenet')
# On reconstruit un modele avec la meme structure que le
# modele entraine sur lequel on souhaite applique gradCAM
# La seule difference sera quon enleve les premieres couches
def build_model_gradcam(size, base):
    base = base
    x= GlobalAveragePooling2D()(base.output)
    x=Dense(2,activation='softmax')(x)
    model = tf.keras.Model(inputs=base.input, outputs=x)
    opt = tf.optimizers.SGD(learning_rate=1e-3)
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=opt,loss=loss,metrics=['AUC'])
    return model

# On charge le modele permettant de predire la tumeur et le tissu normal CNN2
if arg_model == 'EfficientNetB3':
    model = tf.keras.models.load_model('../model/NvsT_EfficientNetB3.h5', compile = False)
elif arg_model == 'densenet':
    model = tf.keras.models.load_model('/work/shared/ptbc/CNN_Pancreas_V2/Analyses_stat/Resultats/StromaVSautre_DB10/NvsT_densenet', compile = False)

else:
    sys.exit('model is required ! ')
# On construit le modele gradcam
model_gradcam = build_model_gradcam(402, base)
model_gradcam.set_weights(model.get_weights()) #  et on lui donne les poids du CNN2 que l'on a entraine prealablement

last_conv_layer_name = list(filter(lambda x: isinstance(x, keras.layers.Activation), model_gradcam.layers))[-1].name
#last_conv_layer_name = list(filter(lambda x: isinstance(x, keras.layers.Conv2D), model_gradcam.layers))[-1].name
print(f"============> Ma couche d'interet de GRADCAM s'appelle {last_conv_layer_name}")
print(f'Le shape de ma derniere couche de model_gradcam est : {model_gradcam.get_layer(last_conv_layer_name).output.shape}')

# So this error
#                           "InternalError: Exception encountered when calling layer "conv2_block1_1_conv" (type Conv2D)."
# is caused by TF32 related optimizations.
#Therefore we can disable this function following this, which is adding this line to the python code.
tf.config.experimental.enable_tensor_float_32_execution(False)


#pfs_features = [153,203,388,410,480,950,989,1119,1218,1314,1468]
pfs_features = [151,180,213,423,572,611,648,959,1002]
def modify_tensor(my_grad, feature_list):
    my_l = [0 for i in range(my_grad.shape[0])]
    for i in feature_list:
        my_l[i] = my_grad[i]
    new_tensor = tf.constant(np.array(my_l), dtype=tf.float32)
    return new_tensor

# divide tensor
def divide_grad(my_grad):
    B = my_grad.numpy()
    new_b = np.divide(B, 100)
    tensor = tf.convert_to_tensor(new_b)
    return tensor



# Fonctions venant du tuto : https://keras.io/examples/vision/grad_cam/
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(np.expand_dims(img_array, axis=0))
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer

    grads = tape.gradient(class_channel, last_conv_layer_output)
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1,2))
    if arg_var == 'key_var':
        pooled_grads = modify_tensor(pooled_grads, pfs_features)
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    #pooled_grads = divide_grad(pooled_grads)
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis] #add new dimension at the end of the pooled grads | @ means multiply then sum
    heatmap = tf.squeeze(heatmap)
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), preds.numpy()[0]


def display_gradcam(img, heatmap,alpha,output_file):
    # Load the original image
    img = keras.preprocessing.image.img_to_array(img)
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("gnuplot") #ROSE => BLEU https://matplotlib.org/stable/tutorials/colors/colormaps.html
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img * (1 - alpha)
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    keras.preprocessing.image.save_img(output_file,superimposed_img)

# creation de la heatmap gradcam qui viendra se superposer a la tuile pour montrer ou se concentre le modele
if isinstance(pred_index, str):
    pred_index = int(pred_index)

heatmap, preds = make_gradcam_heatmap(array, model_gradcam, last_conv_layer_name, pred_index = pred_index)
ind = np.argmax(preds)
prob = preds[ind]
print(f'tuile predite comme {ind} avec une certitude de {prob}')
print(f'prob de la classe 1 = {preds[1]}')
display_gradcam(array, heatmap, 0.009, output_file = f"../Resultats/CAM/tuiles/{Path(img_path).stem}_{arg_model}_{arg_var}_{pred_index}.jpeg") # on superpose l'image de la tuile et de la heatmap et on lenregistre

print('DONE')
