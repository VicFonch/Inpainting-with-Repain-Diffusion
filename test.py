from imports.common_imports import *

from model.diffusion import Diffusion
from model.att_unet import *
from model.utils import *

from data_process.dataloader import Generator

batch_size = 1
target_size = (64, 64)
U = 5

def custom_preprocessing(image):
    return (image / 255.0) * 2.0 - 1.0

if __name__ == "__main__":
    #diffusion = Diffusion(noise_step=1000, batch_size=8, img_size=64)
    
    diffusion = Diffusion()#noise_step = 1000, img_size = target_size[0], img_channels=3)
    # Cargar el modelo de data_dir
    model_dir = "/home/est_posgrado_victor.fonte/Proyectos_y_tareas/ML II/Tarea3-DDIM/_checkpoints/9.h5"
    unet = Unet((64, 64, 3), (1,))
    unet.load_weights(model_dir)
    print(unet.summary())

    data_dir = "_data_test"
    seed = 42
    face_datagen = ImageDataGenerator(preprocessing_function=custom_preprocessing)
    face_generator = Generator(face_datagen)
    mask_datagen = ImageDataGenerator(rescale=1./255)
    mask_generator = Generator(mask_datagen)

    mask_generator  = mask_generator.generator(os.path.join(data_dir, 'mask'), target_size, "grayscale", seed, batch_size)
    face_generator  = face_generator.generator(os.path.join(data_dir, 'face'), target_size, "rgb", seed, batch_size)

    face = next(face_generator)
    mask = next(mask_generator)

    sampled_img = diffusion.denoise_RePint_DDPM(unet, face, mask)

    sampled_img = np.array(sampled_img)
    sampled_img = ((sampled_img + 1.)/2.).astype(np.float32)
    face = np.array(face)
    face = ((face + 1.)/2.).astype(np.float32)

    save_dir = "_outputs/RePaint_output/RePaint.png"
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(face[0])
    ax[1].imshow(mask[0])
    ax[2].imshow(sampled_img[0])
    plt.savefig(save_dir)
    
    print("Done!")