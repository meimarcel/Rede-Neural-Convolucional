#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:49:38 2019

@author: meimarcel
"""

from keras.models import model_from_json

#Ler a rede neural treinada com os pesos
json_file = open("ModeloConv2D_7e.json","r")
modelo_json = json_file.read()
json_file.close()

modelo = model_from_json(modelo_json)
modelo.load_weights("ModeloConv2D_7e.h5")
modelo.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

import numpy as np
from keras.preprocessing import image

#Função para ler e normalizar a imagem
def read_image(path):
    test_image = image.load_img(path, target_size = (64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    return test_image


test_image = read_image("teste/1.jpg")

result = modelo.predict(test_image)

if result[0][0] == 1:
    prediction = 'Cachorro'
else:
    prediction = 'Gato'
print("Imagem reconhecida como: ", prediction)