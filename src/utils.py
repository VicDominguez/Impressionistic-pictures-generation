import platform
import os
from datetime import datetime


def obtener_directorio_absoluto(subdirectorio):
    directorio_base = os.getcwd().replace(sep + 'src', '')
    return directorio_base + subdirectorio


def timestamp_fancy():
    return datetime.now().strftime("%d-%m-%Y %H.%M.%S")


def timestamp():
    return datetime.now()


sep = "\\" if platform.system() == "Windows" else "/"
momento_inicio = timestamp_fancy()

dataset = "monet2photo"

ruta_dataset = obtener_directorio_absoluto(sep + "datasets") + sep + dataset
ruta_imagenes = obtener_directorio_absoluto(sep + "imagenes") + sep + dataset
ruta_modelo = obtener_directorio_absoluto(sep + "modelos") + sep + dataset
ruta_checkpoints_imagenes = obtener_directorio_absoluto(sep + "imagenes_test") + sep + dataset + sep + momento_inicio
ruta_checkpoints_modelo = obtener_directorio_absoluto(sep + "checkpoints") + sep + dataset + sep + momento_inicio

imagen_muestra_pintor = ruta_dataset + sep + "testA" + sep + "00960.jpg"
imagen_muestra_real = ruta_dataset + sep + "testB" + sep + "2014-08-15 08_48_43.jpg"

ancho = 128
alto = 128
dimensiones = (ancho, alto)
canales = 3
forma = (ancho, alto, canales)

os.makedirs(ruta_checkpoints_imagenes, exist_ok=True)
os.makedirs(ruta_checkpoints_modelo, exist_ok=True)
os.makedirs(ruta_imagenes, exist_ok=True)
os.makedirs(ruta_modelo, exist_ok=True)