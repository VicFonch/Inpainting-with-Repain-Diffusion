from imports.common_imports import * 

class Generator:
    def __init__(self, image_datagen):
        self.image_datagen = image_datagen

    def generator(self, directory, target_size, color_mode, seed, batch_size):
        return self.image_datagen.flow_from_directory(
            directory,
            target_size=target_size,
            color_mode = color_mode,
            batch_size=batch_size,
            class_mode=None,
            seed=seed
        )
