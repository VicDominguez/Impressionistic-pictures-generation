import argparse
import pathlib
from pathlib import Path

from cyclegan import CycleGAN
from cargador_imagenes import CargadorImagenes
from utilidades import Utilidades
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version",
                        help="Especifica el nombre de la versión con la que se guardarán los archivos",
                        type=str,
                        default="version_por_defecto")
    parser.add_argument("-d", "--dataset",
                        help="Especifica el dataset a utilizar",
                        type=str,
                        default="monet2photo")
    parser.add_argument("-c", "--configuracion",
                        help="Especifica el fichero de configuración que se va a usar",
                        type=str,
                        default="configuracion.json")
    args = vars(parser.parse_args())

    utils = Utilidades(args["version"], args["dataset"], args["configuracion"])
    logger = utils.obtener_logger("lanzadera produccion")

    utils.asegurar_dataset()
    logger.info("Dataset en linea")

    gan = CycleGAN("resnet")
    logger.info("Red neuronal lista")
    gan.cargar_pesos()
    logger.info("Pesos cargados")

    imagenes_entrada = [imagen.resolve() for imagen in pathlib.Path("../input/").rglob("*.jpg")]
    ruta_raiz_salida = (pathlib.Path("../output/") / args["dataset"] / args["version"]).resolve()
    ruta_raiz_salida.mkdir(parents=True, exist_ok=True)
    imagenes_salida = [ruta_raiz_salida / imagen.parts[-1] for imagen in imagenes_entrada]

    for entrada, salida in zip(imagenes_entrada, imagenes_salida):
        logger.info("Procesando " + salida.parts[-1])
        if salida.exists():
            continue
        imagen_cruda = gan.convertir_imagen(str(entrada), "pintor")
        with open(salida, "wb") as archivo:
            archivo.write(imagen_cruda)
