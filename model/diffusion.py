from imports.common_imports import *

    
class Diffusion:
    def __init__(
            self,
            noise_step = 1000,
            beta_start = 1e-4,
            beta_end = 0.02,
            beta_scheduler = "scaled_linear",
            batch_size = 16,
            img_size = 64,
            img_channels = 3
    ):
        assert beta_scheduler in ["linear", "scaled_linear"], "beta_scheduler must be either 'linear' or 'scaled_linear'"
        self.noise_step = noise_step
        
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_channels = img_channels

        if beta_scheduler == "linear":
            self.beta = tf.linspace(beta_start, beta_end, self.noise_step)
        else:
            self.beta = tf.linspace(beta_start**(0.5), beta_end**(0.5), self.noise_step)**2
        self.alpha = 1. - self.beta
        self.alpha_hat = tf.math.cumprod(self.alpha, axis=0, exclusive=True)
    
    def sample_timestep(self, n):
        random_tensor = tf.random.uniform((n,), minval=1, maxval=1000, dtype=tf.int32)
        return random_tensor

    def noising_process(self, x, t, noise = None):
        sqrt_alpha_hat = tf.sqrt(tf.gather(self.alpha_hat, t))[:, tf.newaxis, tf.newaxis, tf.newaxis]
        sqrt_one_minus_alpha_hat = tf.sqrt(1. - tf.gather(self.alpha_hat, t))[:, tf.newaxis, tf.newaxis, tf.newaxis]
        if noise is None:
            noise = tf.random.normal(shape=tf.shape(x))
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def denoising_DDPM(self, model):
        x = tf.random.normal(shape=(self.batch_size, self.img_size, self.img_size, self.img_channels))
        with tqdm(total = self.noise_step) as pbar:
            for i in reversed(range(1, self.noise_step)):
                
                t = tf.fill((self.batch_size,), i)
                t = tf.cast(t, tf.int64)
                
                alpha = tf.gather(self.alpha, t)[:, tf.newaxis, tf.newaxis, tf.newaxis]
                alpha_hat = tf.gather(self.alpha_hat, t)[:, tf.newaxis, tf.newaxis, tf.newaxis]
                beta = tf.gather(self.beta, t)[:, tf.newaxis, tf.newaxis, tf.newaxis]
            
                predicted_noise = model.predict([x, t], verbose=0, batch_size=self.batch_size)

                if i > 1:
                    noise = tf.random.normal(shape=tf.shape(x))
                else: 
                    noise = tf.zeros(shape=tf.shape(x))

                x = 1 / tf.sqrt(alpha) * (x - ((1 - alpha) / (tf.sqrt(1 - alpha_hat))) * predicted_noise) + tf.sqrt(beta) * noise

                pbar.update(1) 
                
        return x
    
    def denoising_DDIM(self, model):
        x = tf.random.normal((self.batch_size, self.img_size, self.img_size, self.img_channels))
        with tqdm(total = self.noise_step) as pbar:
            for i in reversed(range(1, self.noise_step + 1)):
                
                t = tf.fill((self.batch_size,), i)
                t = tf.cast(t, tf.int64)
                
                alpha_hat_t = tf.gather(self.alpha_hat, t)[:, tf.newaxis, tf.newaxis, tf.newaxis]
                alpha_hat_tm1 = tf.gather(self.alpha_hat, t - 1)[:, tf.newaxis, tf.newaxis, tf.newaxis]
                
                t = tf.expand_dims(t, axis=-1)
                predicted_noise = model.predict([x, t], verbose=0, batch_size=self.batch_size)


                z0 = (1/tf.sqrt(alpha_hat_t))*(x - tf.sqrt(1 - alpha_hat_t) * predicted_noise)
                x = tf.sqrt(alpha_hat_tm1) * z0 + tf.sqrt(1 - alpha_hat_tm1) * predicted_noise
                pbar.update(1)
            
        return x

    def denoise_RePint_DDPM(self, model, image, mask, U = 5):
        x_t = tf.random.normal(shape=(self.batch_size, self.img_size, self.img_size, self.img_channels))
        with tqdm(total = self.noise_step) as pbar:
            for i in reversed(range(self.noise_step)):
                
                t = tf.fill((self.batch_size,), i)
                t = tf.cast(t, tf.int32)

                alpha = tf.gather(self.alpha, t)[:, tf.newaxis, tf.newaxis, tf.newaxis]
                alpha_hat = tf.gather(self.alpha_hat, t)[:, tf.newaxis, tf.newaxis, tf.newaxis]
                beta = tf.gather(self.beta, t)[:, tf.newaxis, tf.newaxis, tf.newaxis]
                
                for u in range(U): 
                    
                    if t[0]>0:
                        noise = tf.random.normal(shape=tf.shape(image), dtype=image.dtype)
                        x_known, _ = self.noising_process(image, t-1, noise)
                    else: 
                        noise = tf.zeros(shape=tf.shape(image), dtype=image.dtype) 
                        x_known, _ = self.noising_process(image, t, noise)

                    predicted_noise = model.predict([x_t, t], verbose=0, batch_size=self.batch_size)

                    x_unknown = 1 / tf.sqrt(alpha) * (x_t - ((1 - alpha) / (tf.sqrt(1 - alpha_hat))) * predicted_noise) + tf.sqrt(beta) * noise
                    
                    x_tm1 = mask*x_known + (1-mask)*x_unknown

                    if u < U and t[0] > 1:
                        sqrt_beta = tf.sqrt(tf.gather(self.beta, t - 1))[:, tf.newaxis, tf.newaxis, tf.newaxis]
                        sqrt_one_minus_beta = tf.sqrt(1 - tf.gather(self.beta, t - 1))[:, tf.newaxis, tf.newaxis, tf.newaxis]
                        noise = tf.random.normal(shape=tf.shape(x_tm1))
                        x_t = sqrt_one_minus_beta * x_t + sqrt_beta * noise

                pbar.update(1)

        return x_tm1