import logging
import pathlib
import sys
from datetime import datetime

from tensorflow.keras.utils import get_file

from src.singleton import Singleton


def timestamp_fancy():
    """Timestamp con la fecha (dia-mes-año) y la hora (hora.minuto.segundo). Apto para directorios"""
    return timestamp().strftime("%d-%m-%Y %H.%M.%S")


def timestamp():
    """Obtener el timestamp actual"""
    return datetime.now()


def obtener_rutas_imagenes(ruta, expresion_regular='*.jpg'):
    """Obtiene las rutas de las imágenes de una carpeta"""
    return list(map(_ruta_a_string, list(ruta.glob(expresion_regular))))


def _ruta_a_string(path):
    """Convertir archivo del tipo Path a un string con la ruta absoluta."""
    return str(path.resolve())


class Utilidades(metaclass=Singleton):
    __instance = None
    momento_inicio = timestamp_fancy()

    dataset = "monet2photo"

    _ruta_dataset = pathlib.Path("../datasets", dataset)
    _repo_dataset = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/" + dataset + ".zip"

    _ruta_dataset_train_pintor = _ruta_dataset / "trainA"
    _ruta_dataset_train_real = _ruta_dataset / "trainB"
    _ruta_dataset_test_pintor = _ruta_dataset / "testA"
    _ruta_dataset_test_real = _ruta_dataset / "testB"

    _ruta_logs = pathlib.Path("../logs", dataset, momento_inicio)
    _ruta_logs_train = _ruta_logs / "train"
    _ruta_logs_test = _ruta_logs / "test"

    _ruta_imagenes = pathlib.Path("../imagenes", dataset, momento_inicio)
    _ruta_modelo = pathlib.Path("../modelos", dataset)
    _ruta_checkpoints_modelo = pathlib.Path("../checkpoints", dataset)

    _cache = pathlib.Path("../tfcache", dataset)

    bucket_gcp = "gs://tfg-impresionismo/"

    _ruta_archivo_muestra_pintor = _ruta_dataset_test_pintor / "00960.jpg"
    _ruta_archivo_muestra_real = _ruta_dataset / 'testB' / '2014-08-15 08_48_43.jpg'

    _mascara_formateo_log = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    def __init__(self):
        self._inicializar_directorios()
        self._logger = self.obtener_logger("utilidades")

    def _inicializar_directorios(self):
        """Crea los directorios a usar"""
        self._ruta_imagenes.mkdir(parents=True, exist_ok=True)
        self._ruta_logs.mkdir(parents=True, exist_ok=True)
        self._ruta_logs_train.mkdir(parents=True, exist_ok=True)
        self._ruta_logs_test.mkdir(parents=True, exist_ok=True)
        self._ruta_modelo.mkdir(parents=True, exist_ok=True)
        self._ruta_checkpoints_modelo.mkdir(parents=True, exist_ok=True)
        self._cache.mkdir(parents=True, exist_ok=True)

    def asegurar_dataset(self):
        if not self._ruta_dataset.exists():
            self._logger.info("Descarga del dataset del repositorio")
            get_file(origin=self._repo_dataset, fname=self.dataset, extract=True)

        assert self._ruta_dataset_train_pintor.exists(), "No existe el directorio de train pintor"
        assert self._ruta_dataset_train_real.exists(), "No existe el directorio de train real"
        assert self._ruta_dataset_test_pintor.exists(), "No existe el directorio de test pintor"
        assert self._ruta_dataset_test_real.exists(), "No existe el directorio de test real"
        assert self._ruta_archivo_muestra_pintor.exists(), "No existe la imagen de muestra del pintor"
        assert self._ruta_archivo_muestra_real.exists(), "No existe la imagen de muestra real"

    def obtener_rutas_imagenes_train_pintor(self):
        return obtener_rutas_imagenes(self._ruta_dataset_train_pintor)

    def obtener_rutas_imagenes_train_real(self):
        return obtener_rutas_imagenes(self._ruta_dataset_train_real)

    def obtener_rutas_imagenes_test_pintor(self):
        return obtener_rutas_imagenes(self._ruta_dataset_test_pintor)

    def obtener_rutas_imagenes_test_real(self):
        return obtener_rutas_imagenes(self._ruta_dataset_test_real)

    def obtener_ruta_train_pintor(self):
        return _ruta_a_string(self._ruta_dataset_train_pintor)

    def obtener_ruta_train_real(self):
        return _ruta_a_string(self._ruta_dataset_train_real)

    def obtener_ruta_test_pintor(self):
        return _ruta_a_string(self._ruta_dataset_test_pintor)

    def obtener_ruta_test_real(self):
        return _ruta_a_string(self._ruta_dataset_test_real)

    def obtener_ruta_logs(self):
        return _ruta_a_string(self._ruta_logs)

    def obtener_ruta_logs_train(self):
        return _ruta_a_string(self._ruta_logs_train)

    def obtener_ruta_logs_test(self):
        return _ruta_a_string(self._ruta_logs_test)

    def obtener_archivo_muestra_pintor(self):
        return _ruta_a_string(self._ruta_archivo_muestra_pintor)

    def obtener_archivo_muestra_real(self):
        return _ruta_a_string(self._ruta_archivo_muestra_real)

    def obtener_ruta_checkpoints_modelo(self):
        return _ruta_a_string(self._ruta_checkpoints_modelo)

    def obtener_archivo_cache(self, nombre):
        return _ruta_a_string(self._cache / (nombre + ".tfcache"))

    def obtener_ruta_imagenes(self):
        return _ruta_a_string(self._ruta_imagenes)

    def obtener_archivo_imagen_a_guardar(self, nombre):
        return _ruta_a_string(self._ruta_imagenes / (nombre + ".png"))

    def obtener_ruta_modelo(self):
        return _ruta_a_string(self._ruta_modelo)

    def obtener_archivo_modelo_a_guardar(self, nombre):
        return _ruta_a_string(self._ruta_modelo / (nombre + ".h5"))

    def _obtener_archivo_logger(self, nombre):
        return _ruta_a_string(self._ruta_logs / (nombre + ".log"))

    def obtener_logger(self, nombre):
        logger = logging.getLogger(nombre)
        formatter = logging.Formatter(self._mascara_formateo_log)
        logger.setLevel(logging.INFO)

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(self._obtener_archivo_logger(nombre))
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stdout_handler)

        return logger
