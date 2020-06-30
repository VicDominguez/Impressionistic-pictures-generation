import pathlib
from pathlib import Path

from src.cyclegan import CycleGAN
from src.cargador_imagenes import CargadorImagenes
from src.utilidades import Utilidades
from PIL import Image

if __name__ == "__main__":
    utils = Utilidades()
    logger = utils.obtener_logger("lanzadera produccion")

    utils.asegurar_dataset()
    logger.info("Dataset en linea")

    gan = CycleGAN("resnet")
    logger.info("Red neuronal lista")
    gan.cargar_pesos()
    logger.info("Pesos cargados")

    imagenes_entrada = [imagen.resolve() for imagen in pathlib.Path("../input/").rglob("*.jpg")]
    ruta_raiz_salida = pathlib.Path("../output/").resolve()
    imagenes_salida = [ruta_raiz_salida / imagen.parts[-1] for imagen in imagenes_entrada]

    for entrada, salida in zip(imagenes_entrada, imagenes_salida):
        logger.info("Procesando " + salida.parts[-1])
        if salida.exists():
            continue
        imagen_cruda = gan.convertir_imagen(str(entrada), "pintor")
        with open(salida, "wb") as archivo:
            archivo.write(imagen_cruda)
