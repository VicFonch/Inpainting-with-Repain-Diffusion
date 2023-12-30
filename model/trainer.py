from imports.common_imports import *

from model.att_unet import *
from model.diffusion import *
from model.utils import *

class Trainer:
    def __init__(
            self,
            train_dataloader,
            batch_size = 8,
            img_size = 64,
            epochs = 100,
            time_dim = 256,
            noise_step = 200
    ):
        self.batch_size = batch_size
        self.train_dataloader = train_dataloader
        self.img_size = img_size
        self.epochs = epochs
        self.time_dim = time_dim
        assert 50000%batch_size==0, "Batch size must be a divisor of 50000"
        self.batch_per_epoch = 50000//batch_size

        self.diffusion = Diffusion(noise_step, batch_size=batch_size, img_size=img_size, beta_scheduler="scaled_linear")
        self.unet = Unet((img_size, img_size, 3), (1, )) 

    def weights_init(self, h2_dir):
        self.unet.load_weights(h2_dir)

    def fit(self, weights_init = None):
        if weights_init is not None:
            self.weights_init(weights_init)
        for epoch in range(self.epochs):
            with tqdm(total = self.batch_per_epoch) as pbar:
                train_loss = []
                for i, batch in enumerate(self.train_dataloader):

                    t = self.diffusion.sample_timestep(self.batch_size)
                    noise_img, epsilon = self.diffusion.noising_process(batch, t)

                    t = tf.expand_dims(t, axis=-1)
                    loss = self.unet.train_on_batch([noise_img, t], epsilon)
                    train_loss.append(loss)
                    
                    if i == self.batch_per_epoch: break
                    
                    pbar.set_description(f"Train Loss: {loss}")
                    pbar.update(1)
            
            print(f"Train Loss: {np.mean(train_loss)}")
            
            sampled_img = self.diffusion.denoising_DDIM(self.unet)
            output_path=f"/home/est_posgrado_victor.fonte/Proyectos_y_tareas/ML II/Tarea3-DDIM/_outputs/pretrain_DDIM/epoca_{epoch}.png"
            save_grid(sampled_img, output_path=output_path)
            self.unet.save(os.path.join("./_checkpoints", f"{epoch}.h5"))

        
