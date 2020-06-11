import os
from datetime import datetime


def obtener_directorio_absoluto(sep, subdirectorio):
    directorio_base = os.getcwd().replace(sep + 'src', '')
    return directorio_base + sep + subdirectorio


def timestamp_fancy():
    return datetime.now().strftime("%d-%m-%Y %H.%M.%S")


def timestamp():
    return datetime.now()
