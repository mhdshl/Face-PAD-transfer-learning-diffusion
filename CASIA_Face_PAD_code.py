import cv2
import math
import numpy as np
import os
import glob
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, losses, models, Model
from keras.layers import BatchNormalization
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
import scipy.io as sio
from scipy.io import loadmat
from random import sample
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, matthews_corrcoef, balanced_accuracy_score, precision_recall_fscore_support
import skimage.io as io
import scipy.ndimage.filters as flt
import warnings
from PIL import Image


def read_all_frames(video_path):
  frame_list = []
  video = cv2.VideoCapture(video_path)
  success = True
  while success:
    success,frame = video.read()
    if success == True:
      frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
      frame_list.append(frame)
    else:
      break
  return np.asarray(frame_list)

def my_frames(data_path, N = 15):
    c_frames = np.empty((0, 224, 224, 3))
    for file in os.listdir(data_path):
        if file.endswith('.avi'): 
            path = os.path.join(data_path, file)
            al_frames = read_all_frames(path)
            total_frames = list(np.asarray(range(0,al_frames.shape[0],1)))
            selected_samples = sample(total_frames,N)
            selected_frames = al_frames[selected_samples]
            c_frames = np.concatenate((c_frames, selected_frames), axis = 0)
    return c_frames


data_path_train_real = '/home/ubuntu/Desktop/Sohail/CASIA_faceAntisp/train_release/real/'
data_path_train_fixed = '/home/ubuntu/Desktop/Sohail/CASIA_faceAntisp/train_release/attack/'

data_path_devel_real = '/home/ubuntu/Desktop/Sohail/CASIA_faceAntisp/devel_release/real/'
data_path_devel_fixed = '/home/ubuntu/Desktop/Sohail/CASIA_faceAntisp/devel_release/attack/'

data_path_test_real = '/home/ubuntu/Desktop/Sohail/CASIA_faceAntisp/test_release/real/'
data_path_test_fixed = '/home/ubuntu/Desktop/Sohail/CASIA_faceAntisp/test_release/attack/'


def load_all_data(data_path_train_real, data_path_train_fixed,
                data_path_devel_real, data_path_devel_fixed,
                  data_path_test_real, data_path_test_fixed, Nr):
  train_real = my_frames(data_path_train_real, Nr)  # 25
  train_fixed = my_frames(data_path_train_fixed)
  yr = train_real.shape[0]
  ya = train_fixed.shape[0] 
  y_real = np.zeros(yr, dtype=int)
  y_attack = np.ones(ya, dtype=int)
  x_train = np.concatenate((train_real, train_fixed), axis = 0)
  y_train = np.concatenate((y_real, y_attack), axis = 0)
  devel_real = my_frames(data_path_devel_real, Nr)
  devel_fixed = my_frames(data_path_devel_fixed)
  yrd = devel_real.shape[0]
  yad = devel_fixed.shape[0] 
  yd_real = np.zeros(yrd, dtype=int)
  yd_attack = np.ones(yad, dtype=int)
  x_val = np.concatenate((devel_real, devel_fixed), axis = 0)
  y_val = np.concatenate((yd_real, yd_attack), axis = 0)
  test_real = my_frames(data_path_test_real, Nr)
  test_fixed = my_frames(data_path_test_fixed)
  ytr = test_real.shape[0]
  yta = test_fixed.shape[0] 
  yt_real = np.zeros(ytr, dtype=int)
  yt_attack = np.ones(yta, dtype=int)
  x_test = np.concatenate((test_real, test_fixed), axis = 0)
  y_test = np.concatenate((yt_real, yt_attack), axis = 0)
  return x_train, y_train, x_val, y_val, x_test, y_test


x_train, y_train, x_val, y_val, x_test, y_test = load_all_data(data_path_train_real, data_path_train_fixed,
                                                                data_path_devel_real, data_path_devel_fixed,
                                                               data_path_test_real, data_path_test_fixed, 40)
print(x_train.shape, x_val.shape, x_test.shape)


def my_model():
  input_tensor = K.Input(shape=(224, 224, 3))
  model = tf.keras.applications.MobileNet(
                                input_shape=(224, 224, 3),
                                alpha=1.0,
                                depth_multiplier=1,
                                dropout=0.001,
                                include_top=False,
                                weights="imagenet",
                                input_tensor=input_tensor,
                                pooling=None,
                                classes=1000,
                                classifier_activation=None
                            ) 
  for layer in model.layers:
      layer.trainable = True
  output = model.layers[-1].output
  flat = K.layers.Flatten()(output)
  dense_1 = K.layers.Dense(512, activation='relu', name = 'dense_1')(flat)
  batch_1 = K.layers.BatchNormalization()(dense_1)
  classify = K.layers.Dense(1, activation='sigmoid', name = 'class_output')(batch_1)
  antispoof = K.models.Model(input_tensor, classify)
  antispoof.compile(optimizer = K.optimizers.Adam(learning_rate=1e-4), loss = K.losses.BinaryCrossentropy(),
                   metrics = 'accuracy')
  return antispoof

model = my_model()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=25)

checkpoint = ModelCheckpoint('/home/ubuntu/Desktop/Sohail/Access_review/CASIA_trainable/CASIA_224_inter_mobilenet_trainable.h5',
                                verbose=0, monitor='val_loss',save_best_only=True, mode='auto')

history = model.fit(x_train, y_train,
              epochs=50,
              batch_size=64,
              shuffle=True,
              validation_data=(x_test, y_test),
              callbacks = [es, checkpoint],
              verbose = 1
              )
del model  
model = load_model('/home/ubuntu/Desktop/Sohail/Access_review/CASIA_trainable/CASIA_224_inter_mobilenet_trainable.h5')


y_test_pred = model.predict(x_test,batch_size=64, verbose=0)
y_test_pred = np.round(np.squeeze(y_test_pred))


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if int(y_actual[i])==int(y_hat[i])==0:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==1:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

tp, fp, tn, fn = perf_measure(y_test, y_test_pred)
print(tp, fp, tn, fn)

acc = (tp+tn)/(tp+tn+fp+fn)

Y_I = (tp/(tp+fn)) + (tn/(tn+fp)) - 1
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
f1score = 2 * tp / (2 * tp + fp + fn)
FAR = fp/(fp + tn)
FRR = fn/(fn + tp)
HTER = (FAR + FRR)/2
EER = (fp+fn)/(tn+fp+fn+tp)
print('CASIA 224 test Results')
print(70*'-')
print('Acc:', acc, 'YI:', Y_I, 'Sen:', sensitivity, 'Spe:', specificity, '\n F1:', f1score, 'HTER:', HTER, 'EER:', EER)



# tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred)
# print(tn, fp, fn, tp)

# acc_train = accuracy_score(y_train, y_train_pred)
# Y_I_train =(tp/(tp+fn)) + (tn/(tn+fp)) - 1
# sensitivity_train = tp / (tp + fn)
# specificity_train = tn / (tn + fp)
# f1score_train = 2 * tp / (2 * tp + fp + fn)
# FAR = fp/(fp + tn)
# FRR = fn/(fn + tp)
# HTER_train = (FAR + FRR)/2
# train_bacc = balanced_accuracy_score(y_train, y_train_pred)
# print('Training Results')
# print(70*'-')
# print('Acc:', acc_train,'YI:', Y_I_train, 'Sen:', sensitivity_train, 'Spe:', specificity_train, '\n F1:', f1score_train, 'HTER:', HTER_train, 'BACC:', train_bacc)

# tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred)
# print(tn, fp, fn, tp)

# acc_val = accuracy_score(y_val, y_val_pred)
# Y_I_val =(tp/(tp+fn)) + (tn/(tn+fp)) - 1
# sensitivity_val = tp / (tp + fn)
# specificity_val = tn / (tn + fp)
# f1score_val = 2 * tp / (2 * tp + fp + fn)
# FAR = fp/(fp + tn)
# FRR = fn/(fn + tp)
# HTER_val = (FAR + FRR)/2
# val_bacc = balanced_accuracy_score(y_val, y_val_pred)
# print('Validation Results')
# print(70*'-')
# print('Acc:', acc_val,'YI:', Y_I_val, 'Sen:', sensitivity_val, 'Spe:', specificity_val, '\n F1:', f1score_val, 'HTER:', HTER_val, 'BACC:', val_bacc)

# tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred)
# print(tn, fp, fn, tp)

# acc_val = accuracy_score(y_test, y_test_pred)
# Y_I_val =(tp/(tp+fn)) + (tn/(tn+fp)) - 1
# sensitivity_val = tp / (tp + fn)
# specificity_val = tn / (tn + fp)
# f1score_val = 2 * tp / (2 * tp + fp + fn)
# FAR = fp/(fp + tn)
# FRR = fn/(fn + tp)
# HTER_val = (FAR + FRR)/2
# val_bacc = balanced_accuracy_score(y_test, y_test_pred)
# print('Testing Results')
# print(70*'-')
# print('Acc:', acc_val,'YI:', Y_I_val, 'Sen:', sensitivity_val, 'Spe:', specificity_val, '\n F1:', f1score_val, 'HTER:', HTER_val, 'BACC:', val_bacc)