import pathlib
from datetime import datetime
from tensorflow.keras.utils import get_file


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


class Utilidades:
    __slots__ = ['momento_inicio', 'dataset', '_ruta_dataset', '_repo_dataset', '_ruta_dataset_train_pintor',
                 '_ruta_dataset_train_real', '_ruta_dataset_test_pintor', '_ruta_dataset_test_real', '_ruta_imagenes',
                 '_ruta_logs', '_ruta_logs_train', '_ruta_logs_test', '_ruta_modelo', '_ruta_checkpoints_modelo',
                 'bucket_gcp', '_ruta_imagen_muestra_pintor', '_ruta_imagen_muestra_real', 'ancho', 'alto', 'canales']

    def __init__(self):
        """Crea los directorios necesarios y comprueba si existen las imágenes de muestra"""
        self.momento_inicio = timestamp_fancy()

        self.dataset = "monet2photo"

        self._ruta_dataset = pathlib.Path("../datasets", self.dataset)
        self._repo_dataset = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/" + self.dataset + ".zip"

        self._ruta_dataset_train_pintor = self._ruta_dataset / "trainA"
        self._ruta_dataset_train_real = self._ruta_dataset / "trainB"
        self._ruta_dataset_test_pintor = self._ruta_dataset / "testA"
        self._ruta_dataset_test_real = self._ruta_dataset / "testB"

        self._ruta_logs = pathlib.Path("../logs", self.dataset, self.momento_inicio)
        self._ruta_logs_train = self._ruta_logs / "train"
        self._ruta_logs_test = self._ruta_logs / "test"

        self._ruta_imagenes = pathlib.Path("../imagenes", self.dataset, self.momento_inicio)
        self._ruta_modelo = pathlib.Path("../modelos", self.dataset)
        self._ruta_checkpoints_modelo = pathlib.Path("../checkpoints", self.dataset)

        self.bucket_gcp = "gs://tfg-impresionismo/"

        self._ruta_imagen_muestra_pintor = self._ruta_dataset_test_pintor / "00960.jpg"
        self._ruta_imagen_muestra_real = self._ruta_dataset / 'testB' / '2014-08-15 08_48_43.jpg'

        self._inicializar_directorios()

    def _inicializar_directorios(self):
        """Crea los directorios a usar"""
        self._ruta_imagenes.mkdir(parents=True, exist_ok=True)
        self._ruta_logs.mkdir(parents=True, exist_ok=True)
        self._ruta_logs_train.mkdir(parents=True, exist_ok=True)
        self._ruta_logs_test.mkdir(parents=True, exist_ok=True)
        self._ruta_modelo.mkdir(parents=True, exist_ok=True)
        self._ruta_checkpoints_modelo.mkdir(parents=True, exist_ok=True)

    def asegurar_dataset(self):
        if not self._ruta_dataset.exists():
            get_file(origin=self._repo_dataset, fname=self.dataset, extract=True)

        assert self._ruta_dataset_train_pintor.exists(), "No existe el directorio de train pintor"
        assert self._ruta_dataset_train_real.exists(), "No existe el directorio de train real"
        assert self._ruta_dataset_test_pintor.exists(), "No existe el directorio de test pintor"
        assert self._ruta_dataset_test_real.exists(), "No existe el directorio de test real"
        assert self._ruta_imagen_muestra_pintor.exists(), "No existe la imagen de muestra del pintor"
        assert self._ruta_imagen_muestra_real.exists(), "No existe la imagen de muestra real"

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

    def obtener_ruta_imagen_muestra_pintor(self):
        return _ruta_a_string(self._ruta_imagen_muestra_pintor)

    def obtener_ruta_imagen_muestra_real(self):
        return _ruta_a_string(self._ruta_imagen_muestra_real)

    def obtener_ruta_checkpoints_modelo(self):
        return _ruta_a_string(self._ruta_checkpoints_modelo)

    def obtener_ruta_imagenes(self):
        return _ruta_a_string(self._ruta_imagenes)

    def obtener_ruta_imagen_a_guardar(self, nombre):
        return _ruta_a_string(self._ruta_imagenes / (nombre + ".png"))

    def obtener_ruta_modelo(self):
        return _ruta_a_string(self._ruta_modelo)

    def obtener_ruta_modelo_a_guardar(self, nombre):
        return _ruta_a_string(self._ruta_modelo / (nombre + ".h5"))
