from src.cyclegan import CycleGAN
from src.lector_imagenes import LectorImagenes
from src.utilidades import Utilidades

if __name__ == "__main__":
    utils = Utilidades()
    utils.asegurar_dataset()
    gan = CycleGAN()
    gan.train(LectorImagenes(), epochs=200)

