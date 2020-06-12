import funciones
import platform
import os


class Constantes(object):
    __instance = None
    
    sep = "\\" if platform.system() == "Windows" else "/"
    momento_inicio = funciones.timestamp_fancy()

    dataset = "monet2photo"

    ruta_dataset = funciones.obtener_directorio_absoluto(sep, "datasets") + sep + dataset
    ruta_imagenes = funciones.obtener_directorio_absoluto(sep, "imagenes") + sep + dataset
    ruta_modelo = funciones.obtener_directorio_absoluto(sep, "modelos") + sep + dataset
    ruta_checkpoints_modelo = funciones.obtener_directorio_absoluto(
        sep, "checkpoints") + sep + dataset + sep + momento_inicio
    ruta_logs = funciones.obtener_directorio_absoluto(
        sep, "logs") + sep + dataset + sep + momento_inicio
    ruta_logs_train = funciones.obtener_directorio_absoluto(
        sep, "logs") + sep + dataset + sep + momento_inicio + sep + "train" + sep
    ruta_logs_test = funciones.obtener_directorio_absoluto(
        sep, "logs") + sep + dataset + sep + momento_inicio + sep + "test" + sep

    bucket_gcp = "gs://tfg-impresionismo/"

    imagen_muestra_pintor = ruta_dataset + sep + "testA" + sep + "00960.jpg"
    imagen_muestra_real = ruta_dataset + sep + "testB" + sep + "2014-08-15 08_48_43.jpg"

    ancho = 128
    alto = 128
    dimensiones = (ancho, alto)
    canales = 3
    forma = (ancho, alto, canales)

    def __new__(cls):
        if Constantes.__instance is None:
            Constantes.__instance = object.__new__(cls)
            Constantes.__instance.crear_directorios()
        return Constantes.__instance

    def crear_directorios(self):
        os.makedirs(self.ruta_checkpoints_modelo, exist_ok=True)
        os.makedirs(self.ruta_imagenes, exist_ok=True)
        os.makedirs(self.ruta_modelo, exist_ok=True)
        os.makedirs(self.ruta_logs, exist_ok=True)
