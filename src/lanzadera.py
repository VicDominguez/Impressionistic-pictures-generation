from cyclegan import CycleGAN

if __name__ == "__main__":
    gan = CycleGAN()
    gan.train(epochs=200)
