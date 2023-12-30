from imports.common_imports import *

from data_process.dataloader import Generator

from model.trainer import Trainer
from model.utils import *
from model.att_unet import *

# Normaliza la imagen entre -1 y 1
def custom_preprocessing(image):
    return (image / 255.0) * 2.0 - 1.0

if __name__ == "__main__":

    data_dir = "./_data"

    seed = 24
    batch_size = 10
    target_size=(64, 64)
    target_channels=3

    img_datagen = ImageDataGenerator(preprocessing_function=custom_preprocessing)
    data_generator = Generator(img_datagen)
    train_generator = data_generator.generator(data_dir, target_size, "rgb", seed, batch_size)

    trainer = Trainer(
        train_generator,
        batch_size=batch_size,
        img_size=64,
        time_dim=256,
        epochs=100,
        noise_step = 200
    )

    model = Unet((target_size[0], target_size[1], target_channels), (1, ))
    print(model.summary())

    weights_init = "_models/7.h5"
    trainer.fit()