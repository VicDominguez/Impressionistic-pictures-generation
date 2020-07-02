import argparse
import pathlib

from cyclegan import CycleGAN  # TODO Quitar esto
from cargador_imagenes import CargadorImagenes
from utilidades import Utilidades

if __name__ == "__main__":
    # Leemos los argumentos
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
                        default="configuracion_128.json")
    parser.add_argument("-a", "--arquitectura",
                        help="Especifica el tipo de arquitectura: unet o resnet",
                        type=str,
                        default="resnet")
    args = vars(parser.parse_args())

    # Comprobamos que los parámetros son válidos
    assert pathlib.Path( "../configuracion", args["configuracion"]).exists(), "El archivo no existe"
    assert args["dataset"] in ["monet2photo", "cezanne2photo", "ukiyoe2photo", "vangogh2photo"], \
        "El dataset no es valido"
    assert args["arquitectura"] in ["resnet", "unet"], "El dataset no es valido"

    utils = Utilidades(args["version"], args["dataset"], args["configuracion"])

    logger = utils.obtener_logger("lanzadera entrenamiento")

    utils.asegurar_dataset()
    logger.info("Dataset en linea")

    logger.info("Creacion de la red neuronal")
    gan = CycleGAN(args["arquitectura"])
    logger.info("Red neuronal lista")

    logger.info("Iniciando el lector de imágenes")
    lector = CargadorImagenes()
    logger.info("Imágenes listas")

    logger.info("Comienzo del entrenamiento")
    gan.train(lector)
    gan.serializar_red()
    logger.info("Fin del entrenamiento")
