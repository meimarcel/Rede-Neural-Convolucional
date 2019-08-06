"""
Created on Tue Jul 30 03:24:06 2019

@author: meimarcel
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Criaç]ao da rede neural
rede_neural = Sequential()
rede_neural.add(Conv2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
rede_neural.add(MaxPooling2D(pool_size = (2, 2)))

rede_neural.add(Conv2D(32, (3, 3), activation = 'relu'))
rede_neural.add(MaxPooling2D(pool_size = (2, 2)))

rede_neural.add(Flatten())

rede_neural.add(Dense(units = 128, activation = 'relu'))
rede_neural.add(Dense(units = 1, activation = 'sigmoid'))

rede_neural.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Normalização da imagens de entrada
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dogs-and-cats-dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dogs-and-cats-dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

#Treinar a rede
rede_neural.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 2000)


#Salvar a rede e os pesos
rede_neural_json = rede_neural.to_json()
with open("ModeloConv2D.json","w") as json_file:
    json_file.write(rede_neural_json)
    
rede_neural.save_weights("ModeloConv2D.h5")
