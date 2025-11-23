
import numpy as np
import tensorflow as tf
import keras as kr
import sklearn
import kagglehub


####
# Prefer tensorflow.keras but fall back to standalone keras if the environment or linter
# cannot resolve tensorflow.keras (helps editors that show "Unable to import 'tensorflow.keras'").

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import to_categorical

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.callbacks import ReduceLROnPlateau

# Cargar MNIST y dividir en train / test
def cargar_dataset():

	path = kagglehub.dataset_download("muhammetzahitaydn/hardhat-vest-dataset-v3")

	print("Path to dataset files:", path)

	# En directory vas a insertar la dirección del dataset, el cual no esta en el repositorio por que pesa 4GB.
	# class_names tiene los nombres de las carpetas, las cataloga automaticamente
	# image_size creo que deberia ser (640,640)
	# labels va a nombrar los grupos de imagenes automaticamente con numeros, sino espera una tupla de enteros. Con class names los cambia.
	
	
	# directory= "A:/Escritorio/Facultad/Cuatri/TP_IA/NOSUBIR"
	testDS= kr.preprocessing.image_dataset_from_directory(path,
										labels='inferred',
										label_mode='int',
										class_names=None,
										color_mode='rgb',
										batch_size=32,
										image_size=(640, 640),
										shuffle=True,
										seed=1,
										validation_split=None,
										subset=None,
										interpolation='bilinear',
										follow_links=False,
										crop_to_aspect_ratio=False,
										pad_to_aspect_ratio=False,
										data_format=None,
										verbose=True
		)
	
	trainDS= kr.preprocessing.image_dataset_from_directory(path,
										labels='inferred',
										label_mode='int',
										class_names=None,
										color_mode='rgb',
										batch_size=32,
										image_size=(640, 640),
										shuffle=True,
										seed=1,
										validation_split=None,
										subset=None,
										interpolation='bilinear',
										follow_links=False,
										crop_to_aspect_ratio=False,
										pad_to_aspect_ratio=False,
										data_format=None,
										verbose=True
		)
	
	normalization_layer = kr.layers.Rescaling(1./255)
	trainDS = trainDS.map(lambda x, y: (normalization_layer(x), y))
	testDS = testDS.map(lambda x, y: (normalization_layer(x), y))

	return trainDS, testDS

# Normalización de tonos de pixel
def prep_pixels(train, test):
	# Transformar integers en floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# Normalizar 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# Devolver normalizado
	return train_norm, test_norm


def build_model():
	print("\n"*3)
	model = Sequential()
	#
	model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same',
					activation ='relu', input_shape = (640,640,3)))
	model.add(MaxPool2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	#
	model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same',
					activation ='relu'))
	model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
	model.add(Dropout(0.25))
	# fully connected
	model.add(Flatten())
	model.add(Dense(256, activation = "relu"))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation = "softmax"))

	optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

	model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

	

	trainDS, testDS = cargar_dataset()

	epochs = 10
	batch_size = 250

	history = model.fit(trainDS, batch_size=batch_size,epochs = epochs,validation_data=testDS)

build_model()