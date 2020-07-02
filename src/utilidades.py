import json
import logging
import pathlib
import platform
import shutil
import subprocess
import sys
from datetime import datetime

from tensorflow.keras.utils import get_file

from singleton import Singleton


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


def is_tool(name):
    """Check whether `name` is on PATH and marked as executable.
    Recogida de https://stackoverflow.com/questions/11210104/check-if-a-program-exists-from-a-python-script"""

    return shutil.which(name) is not None


def gsutil_disponible():
    """Revisa si está disponible la herramienta gsutil, necesaria para las ejecuciones en GCP"""
    return is_tool("gsutil")


def obtener_nombre_relativo_desde_string(archivo):
    """Devuelve el nombre del archivo a partir de un string que contiene la ruta absoluta del mismo.

    Parámetros:
        archivo: string con la ruta de fichero"""
    return archivo.split("\\")[-1] if platform.system() == "Windows" else archivo.split("/")[-1]


class Utilidades(metaclass=Singleton):
    __slots__ = ["_version", "_dataset", "_ruta_logs", "_ruta_modelos", "_ruta_raiz_dataset", "_ruta_tfcache",
                 "_ruta_logs_entreno", "_ruta_logs_test", "_ruta_modelo_modelos", "_ruta_modelo_configuracion",
                 "_ruta_modelo_imagenes", "_archivo_dataset", "_ruta_dataset", "_ruta_dataset_entreno_pintor",
                 "_ruta_dataset_entreno_foto", "_ruta_dataset_test_pintor", "_ruta_dataset_test_foto", "_ruta_cache",
                 "_tasa_aprendizaje", "_lambda_reconstruccion", "_lambda_validacion", "_lambda_identidad",
                 "_dimensiones", "_epochs", "_tamanio_buffer", "_tamanio_batch", "_filtros_generador",
                 "_filtros_discriminador", "_url_datasets", "_url_api_aumento", "_gcp_bucket", "_imagen_pintor_muestra",
                 "_imagen_foto_muestra", "_ruta_archivo_muestra_pintor", "_ruta_archivo_muestra_foto", "_mascara_logs",
                 "_logger"]

    def __init__(self, version, dataset, archivo_configuracion):
        self._version = str(version)
        self._dataset = str(dataset)

        # Leemos el fichero json
        ruta_configruacion = pathlib.Path("../configuracion", archivo_configuracion).resolve()

        with open(ruta_configruacion) as archivo:
            datos_json = json.load(archivo)

        # Configuramos el resto de parámetros
        self._ruta_logs = pathlib.Path("../logs", self._dataset, self._version)
        self._ruta_modelos = pathlib.Path("../modelos", self._dataset, self._version)
        self._ruta_raiz_dataset = pathlib.Path("../datasets")
        self._ruta_tfcache = pathlib.Path("../tfcache")

        self._ruta_logs_entreno = self._ruta_logs / "entreno"
        self._ruta_logs_test = self._ruta_logs / "test"

        self._ruta_modelo_modelos = self._ruta_modelos / "modelo"
        self._ruta_modelo_configuracion = self._ruta_modelos / "config"
        self._ruta_modelo_imagenes = self._ruta_modelos / "imagenes"

        self._archivo_dataset = self._dataset + ".zip"
        self._ruta_dataset = self._ruta_raiz_dataset / self._dataset

        self._ruta_dataset_entreno_pintor = self._ruta_dataset / "trainA"
        self._ruta_dataset_entreno_foto = self._ruta_dataset / "trainB"
        self._ruta_dataset_test_pintor = self._ruta_dataset / "testA"
        self._ruta_dataset_test_foto = self._ruta_dataset / "testB"

        self._ruta_cache = self._ruta_tfcache / self._dataset / self._version
        # las caches son excluyentes entre iteraciones

        # Leemos los parámetros del modelo
        self._tasa_aprendizaje = float(datos_json["configuracion_modelo"]["tasa_aprendizaje"])
        self._lambda_reconstruccion = float(datos_json["configuracion_modelo"]["lambda_reconstruccion"])
        self._lambda_validacion = int(datos_json["configuracion_modelo"]["lambda_validacion"])
        self._lambda_identidad = int(datos_json["configuracion_modelo"]["lambda_identidad"])
        _ancho = int(datos_json["configuracion_modelo"]["ancho"])
        _alto = int(datos_json["configuracion_modelo"]["alto"])
        _canales = int(datos_json["configuracion_modelo"]["canales"])
        self._dimensiones = (_ancho, _alto, _canales)
        self._epochs = int(datos_json["configuracion_modelo"]["epochs"])
        self._tamanio_buffer = int(datos_json["configuracion_modelo"]["tamanio_buffer"])
        self._tamanio_batch = int(datos_json["configuracion_modelo"]["tamanio_batch"])
        self._filtros_generador = int(datos_json["configuracion_modelo"]["filtros_generador"])
        self._filtros_discriminador = int(datos_json["configuracion_modelo"]["filtros_discriminador"])

        self._url_datasets = datos_json["url"]["datasets"] + self._archivo_dataset
        self._url_api_aumento = datos_json["url"]["api_aumento"]

        self._gcp_bucket = datos_json["gcp"]["bucket"]

        self._imagen_pintor_muestra = datos_json["dataset"][self._dataset]["imagen_pintor_muestra"]
        self._imagen_foto_muestra = datos_json["dataset"][self._dataset]["imagen_foto_muestra"]

        if platform.system() == "Windows":  # Compatibilidad entre Windows y Linux. Windows no admite :
            self._imagen_foto_muestra = self._imagen_foto_muestra.replace(":", "_")
        else:
            self._imagen_foto_muestra = self._imagen_foto_muestra.replace("_", ":")

        self._ruta_archivo_muestra_pintor = self._ruta_dataset_test_pintor / self._imagen_pintor_muestra
        # Windows no admite :
        self._ruta_archivo_muestra_foto = self._ruta_dataset_test_foto / self._imagen_foto_muestra

        self._mascara_logs = datos_json["varios"]["mascara_logs"]

        self._inicializar_directorios()
        self._logger = self.obtener_logger("utilidades")

    def _inicializar_directorios(self):
        """Crea los directorios a usar"""
        self._ruta_logs.mkdir(parents=True, exist_ok=True)
        self._ruta_logs_entreno.mkdir(parents=True, exist_ok=True)
        self._ruta_logs_test.mkdir(parents=True, exist_ok=True)
        self._ruta_modelos.mkdir(parents=True, exist_ok=True)
        self._ruta_modelo_configuracion.mkdir(parents=True, exist_ok=True)
        self._ruta_modelo_modelos.mkdir(parents=True, exist_ok=True)
        self._ruta_modelo_imagenes.mkdir(parents=True, exist_ok=True)
        if self._ruta_tfcache.exists():  # TODO probar esto en detalle
            shutil.rmtree(_ruta_a_string(self._ruta_tfcache), ignore_errors=False, onerror=None)
        self._ruta_cache.mkdir(parents=True, exist_ok=True)

    def asegurar_dataset(self):
        """Si no está el dataset en local, se descarga"""
        if not self._ruta_dataset.exists():
            self._logger.info("Descarga del dataset del repositorio")
            self._ruta_dataset.mkdir(parents=True)
            get_file(origin=self._url_datasets, fname=self._archivo_dataset, extract=True,
                     cache_dir=self._ruta_raiz_dataset, cache_subdir="./")

        assert self._ruta_dataset_entreno_pintor.exists(), "No existe el directorio de entreno pintor"
        assert self._ruta_dataset_entreno_foto.exists(), "No existe el directorio de entreno real"
        assert self._ruta_dataset_test_pintor.exists(), "No existe el directorio de test pintor"
        assert self._ruta_dataset_test_foto.exists(), "No existe el directorio de test real"
        assert self._ruta_archivo_muestra_pintor.exists(), "No existe la imagen de muestra del pintor"
        assert self._ruta_archivo_muestra_foto.exists(), "No existe la imagen de muestra real"

    def copiar_logs_gcp(self):
        """Ejecuta el proceso de copiar los logs al bucket de gcp."""
        subprocess.run(["gsutil", "cp", "-r", self.obtener_ruta_logs(), self._gcp_bucket])

    def obtener_logger(self, nombre):
        """Devuelve un logger que escribe tanto en fichero como en la salida estándar.

        Parámetros:

            nombre: nombre del logger a crear. Suele utilizarse el nombre de la clase/módulo."""
        logger = logging.getLogger(nombre)
        formateador = logging.Formatter(self._mascara_logs)
        logger.setLevel(logging.INFO)

        manejador_salida_estandar = logging.StreamHandler(sys.stdout)
        manejador_salida_estandar.setFormatter(formateador)
        manejador_archivo = logging.FileHandler(self._obtener_archivo_logger(nombre))
        manejador_archivo.setFormatter(formateador)

        logger.addHandler(manejador_archivo)
        logger.addHandler(manejador_salida_estandar)

        return logger

    # Funciones para obtener los atributos privados
    def obtener_rutas_imagenes_entreno_pintor(self):
        return obtener_rutas_imagenes(self._ruta_dataset_entreno_pintor)

    def obtener_rutas_imagenes_entreno_foto(self):
        return obtener_rutas_imagenes(self._ruta_dataset_entreno_foto)

    def obtener_rutas_imagenes_test_pintor(self):
        return obtener_rutas_imagenes(self._ruta_dataset_test_pintor)

    def obtener_rutas_imagenes_test_foto(self):
        return obtener_rutas_imagenes(self._ruta_dataset_test_foto)

    def obtener_ruta_logs(self):
        return _ruta_a_string(self._ruta_logs)

    def obtener_ruta_logs_entreno(self):
        return _ruta_a_string(self._ruta_logs_entreno)

    def obtener_ruta_logs_test(self):
        return _ruta_a_string(self._ruta_logs_test)

    def obtener_archivo_muestra_pintor(self):
        return _ruta_a_string(self._ruta_archivo_muestra_pintor)

    def obtener_archivo_muestra_foto(self):
        return _ruta_a_string(self._ruta_archivo_muestra_foto)

    def obtener_archivo_cache(self, nombre):
        return _ruta_a_string(self._ruta_cache / (nombre + ".tfcache"))

    def obtener_archivo_imagen_a_guardar(self, nombre):
        return _ruta_a_string(self._ruta_modelo_imagenes / (str(nombre) + ".png"))

    def obtener_ruta_archivo_modelo_parametros(self):
        return _ruta_a_string(self._ruta_modelo_configuracion / "parametros.pkl")

    def obtener_ruta_archivo_modelo_objeto(self):
        return _ruta_a_string(self._ruta_modelo_configuracion / "cyclegan.pkl")

    def obtener_ruta_archivo_modelo_esquema(self):
        return _ruta_a_string(self._ruta_modelo_configuracion / "modelo_combinado.png")

    def obtener_ruta_archivo_discriminador_pintor_esquema(self):
        return _ruta_a_string(self._ruta_modelo_configuracion / "discriminador_pintor.png")

    def obtener_ruta_archivo_discriminador_foto_esquema(self):
        return _ruta_a_string(self._ruta_modelo_configuracion / "discriminador_foto.png")

    def obtener_ruta_archivo_generador_pintor_esquema(self):
        return _ruta_a_string(self._ruta_modelo_configuracion / "generador_pintor.png")

    def obtener_ruta_archivo_generador_foto_esquema(self):
        return _ruta_a_string(self._ruta_modelo_configuracion / "generador_foto.png")

    def obtener_ultimos_pesos(self):
        lista_pesos = list(pathlib.Path(self._ruta_modelo_modelos).rglob("pesos*.h5"))  # obtenemos la lista de pesos
        if lista_pesos:
            maximo = max(list(map(lambda x: int(str(x).split("-")[-1].split(".")[0]), lista_pesos)))
            # procesamos (si hay) el nombre de fichero:
            # nos quedamos con la parte dcha del guion y la izquierda del punto, es decir
            # el numero de epoch. Obtenemos el maximo
            return _ruta_a_string(self._ruta_modelo_modelos / ("pesos-" + str(maximo) + ".h5")), maximo
        else:
            return None, 0

    def obtener_ruta_fichero_modelo(self):
        return _ruta_a_string(self._ruta_modelo_modelos / "modelo_combinado.h5")

    def obtener_ruta_fichero_discriminador_pintor(self):
        return _ruta_a_string(self._ruta_modelo_modelos / "discriminador_pintor.h5")

    def obtener_ruta_fichero_discriminador_foto(self):
        return _ruta_a_string(self._ruta_modelo_modelos / "discriminador_foto.h5")

    def obtener_ruta_fichero_generador_pintor(self):
        return _ruta_a_string(self._ruta_modelo_modelos / "generador_pintor.h5")

    def obtener_ruta_fichero_generador_foto(self):
        return _ruta_a_string(self._ruta_modelo_modelos / "generador_foto.h5")

    def obtener_ruta_fichero_modelo_por_epoch(self, epoch):
        return _ruta_a_string(self._ruta_modelo_modelos / ("pesos-" + str(epoch) + ".h5"))

    def _obtener_archivo_logger(self, nombre):
        return _ruta_a_string(self._ruta_logs / (nombre + ".log"))

    def obtener_dimensiones(self):
        return self._dimensiones

    def obtener_tamanio_batch(self):
        return self._tamanio_batch

    def obtener_tamanio_buffer(self):
        return self._tamanio_buffer

    def obtener_tasa_aprendizaje(self):
        return self._tasa_aprendizaje

    def obtener_lambda_reconstruccion(self):
        return self._lambda_reconstruccion

    def obtener_lambda_validacion(self):
        return self._lambda_validacion

    def obtener_lambda_identidad(self):
        return self._lambda_identidad

    def obtener_filtros_generador(self):
        return self._filtros_generador

    def obtener_filtros_discriminador(self):
        return self._filtros_discriminador

    def obtener_epochs(self):
        return self._epochs

    def obtener_url_api_aumento(self):
        return self._url_api_aumento
