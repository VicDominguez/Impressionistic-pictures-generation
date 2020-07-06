import argparse
import pathlib

from cyclegan import CycleGAN
from utilidades import Utilidades, obtener_nombre_relativo_desde_string

if __name__ == "__main__":

    # Comandos entrada
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
    args = vars(parser.parse_args())

    # inicializamos la clase de utilidades y la red
    utils = Utilidades(args["version"], args["dataset"], args["configuracion"])
    logger = utils.obtener_logger("lanzadera produccion")

    utils.asegurar_dataset()
    logger.info("Dataset en linea")

    gan = CycleGAN(restaurar=True)
    logger.info("Red neuronal lista")

    # listamos las imagenes que queremos traducir
    tipos_imagenes_admitidas = ['jpg', 'jpeg', 'bmp', 'png']

    imagenes_entrada_usuario = [(item.resolve()) for item in pathlib.Path("../input").rglob("*")
                                if item.parts[-1].split(".")[-1] in tipos_imagenes_admitidas]

    imagenes_entrada_test_para_foto = utils.obtener_rutas_imagenes_test_pintor()
    imagenes_entrada_test_para_pintor = utils.obtener_rutas_imagenes_test_foto()

    # creamos las carpetas de salidas
    ruta_raiz_salida_usuario = (pathlib.Path("../output/") / args["dataset"] / args["version"]).resolve()
    ruta_raiz_test_foto = (pathlib.Path("../output/") / args["dataset"] / args["version"] / "test_foto").resolve()
    ruta_raiz_test_pintor = (pathlib.Path("../output/") / args["dataset"] / args["version"] / "test_pintor").resolve()

    ruta_raiz_salida_usuario.mkdir(parents=True, exist_ok=True)
    ruta_raiz_test_foto.mkdir(parents=True, exist_ok=True)
    ruta_raiz_test_pintor.mkdir(parents=True, exist_ok=True)

    # especificamos las rutas de salida
    imagenes_salida_usuario = [ruta_raiz_salida_usuario / imagen.parts[-1] for imagen in imagenes_entrada_usuario]
    imagenes_salida_test_para_foto = [ruta_raiz_test_foto / obtener_nombre_relativo_desde_string(imagen)
                                      for imagen in imagenes_entrada_test_para_foto]
    imagenes_salida_test_para_pintor = [ruta_raiz_test_pintor / obtener_nombre_relativo_desde_string(imagen)
                                        for imagen in imagenes_entrada_test_para_pintor]

    # calculamos
    for entrada, salida in zip(imagenes_entrada_usuario, imagenes_salida_usuario):
        logger.info("Procesando " + salida.parts[-1])
        if salida.exists():
            continue
        imagen_cruda = gan.convertir_imagen(str(entrada), "pintor", 8)
        with open(salida, "wb") as archivo:
            archivo.write(imagen_cruda)

    for entrada, salida in zip(imagenes_entrada_test_para_foto, imagenes_salida_test_para_foto):
        logger.info("Procesando " + salida.parts[-1])
        if salida.exists():
            continue
        imagen_cruda = gan.convertir_imagen(str(entrada), "foto", 4)
        with open(salida, "wb") as archivo:
            archivo.write(imagen_cruda)

    for entrada, salida in zip(imagenes_entrada_test_para_pintor, imagenes_salida_test_para_pintor):
        logger.info("Procesando " + salida.parts[-1])
        if salida.exists():
            continue
        imagen_cruda = gan.convertir_imagen(str(entrada), "pintor", 4)
        with open(salida, "wb") as archivo:
            archivo.write(imagen_cruda)
