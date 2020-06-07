import os


def obtener_directorio_absoluto(subdirectorio):
    directorio_base = os.getcwd().replace('\src', '')
    return directorio_base + subdirectorio


dataset = "monet2photo"

ruta_padre_datasets = obtener_directorio_absoluto("\\datasets")
ruta_padre_imagenes = obtener_directorio_absoluto("\\imagenes_entrenamiento")
ruta_padre_modelos = obtener_directorio_absoluto("\\modelos")

ruta_dataset = ruta_padre_datasets + "\\" + dataset
ruta_imagenes = ruta_padre_imagenes + "\\" + dataset
ruta_modelo = ruta_padre_modelos + "\\" + dataset

imagen_muestra_monet = ruta_dataset + "\\testA\\" + "00960.jpg"
imagen_muestra_real = ruta_dataset + "\\testB\\" + "2014-08-15 08_48_43.jpg"

ancho = 128
alto = 128
dimensiones = (ancho, alto)
