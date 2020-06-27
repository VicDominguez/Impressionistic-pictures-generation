from cyclegan import CycleGAN
from lector_imagenes import LectorImagenes
from utilidades import Utilidades

if __name__ == "__main__":
    utils = Utilidades()
    logger = utils.obtener_logger("lanzadera produccion")

    utils.asegurar_dataset()
    logger.info("Dataset en linea")

    gan = CycleGAN("resnet")
    logger.info("Red neuronal lista")
    gan.cargar_pesos()
    logger.info("Pesos cargados")
    gan.convertir_imagen(utils.obtener_archivo_muestra_foto(), "pintor")
    gan.convertir_imagen(utils.obtener_archivo_muestra_pintor(), "foto")
