from imports.common_imports import *


class TempEmbedding(Layer):
    def __init__(self, time_dim = 256):
        super(TempEmbedding, self).__init__()
        self.time_dim = time_dim

    def call(self, t):
        inv_freq = 1.0 / (
            10000
            ** (tf.range(0, self.time_dim, 2, dtype=tf.float32) / self.time_dim)
        )
        t = tf.repeat(t, repeats=[self.time_dim // 2], axis=1)
        pos_enc_a = tf.math.sin(t * inv_freq)
        pos_enc_b = tf.math.cos(t * inv_freq)
        pos_enc = tf.concat([pos_enc_a, pos_enc_b], axis=-1)
        return pos_enc
    

class DoubleConv(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
        residual: bool = False,
    ):
        super(DoubleConv, self).__init__()
        
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = Sequential([
            Conv2D(mid_channels, kernel_size=3, padding='same', use_bias=False),
            GroupNormalization(groups=mid_channels), 
            ReLU(),
            Conv2D(out_channels, kernel_size=3, padding='same', use_bias=False),
            LayerNormalization(),
        ])
    
    def call(self, x):
        if self.residual:
            return tf.nn.elu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(Layer):
    def __init__(
            self, 
            in_channels: int,
            out_channels: int,
    ):
        super(Down, self).__init__()
        self.max_pool = Sequential([
            AveragePooling2D(pool_size=2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        ])

        self.emb_layer = Sequential([
            ELU(),
            Dense(units=out_channels),
        ])
    
    def call(self, x, t):
        x = self.max_pool(x)

        t = tf.cast(t, tf.float32)
        emb = self.emb_layer(t)

        emb = tf.expand_dims(emb, axis=1) 
        emb = tf.expand_dims(emb, axis=1)  
        emb = tf.tile(emb, (1, x.shape[1], x.shape[2], 1))
        
        return x + emb
    
class Up(Layer):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            emb_dim: int = 256,       
    ):
        super(Up, self).__init__()

        self.up = Conv2DTranspose(in_channels//2, kernel_size=2, strides=2, padding='same')
        self.conv = Sequential([
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        ])
        
        self.emb_layer = Sequential([
            ELU(),
            Dense(units=out_channels),
        ])
    
    def call(self, x, skip_x, t):
        x = self.up(x)
        x = Concatenate(axis=-1)([skip_x, x])
        x = self.conv(x)

        t = tf.cast(t, tf.float32)
        emb = self.emb_layer(t)

        emb = tf.expand_dims(emb, axis=1) 
        emb = tf.expand_dims(emb, axis=1)  
        emb = tf.tile(emb, (1, x.shape[1], x.shape[2], 1))
        
        return x + emb
        
class SelfAttention(Layer):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = MultiHeadAttention(num_heads=4, key_dim=channels, batch_size=None)
        self.ln = LayerNormalization(epsilon=1e-6)
        self.ff_self = keras.Sequential([
            LayerNormalization(epsilon=1e-6),
            Dense(units=channels, activation='gelu'),
            Dense(units=channels)
        ])

    def call(self, x):
        batch_size = tf.shape(x)[0]
        x_reshaped = tf.reshape(x, (batch_size, self.size * self.size, self.channels))
        x_ln = self.ln(x_reshaped)
        attention_value = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x_reshaped
        attention_value = self.ff_self(attention_value) + attention_value
        return tf.reshape(attention_value, (batch_size, self.size, self.size, self.channels))

def Unet(img_size, t_size):
    img_input = Input(shape=img_size)
    t_input = Input(shape=t_size)

    t = tf.cast(t_input, tf.float32)
    t = TempEmbedding(256)(t_input)

    c = 224
    
    x1 = DoubleConv(3, c)(img_input)
    x2 = Down(c, 2*c)(x1, t)
    x2 = SelfAttention(2*c, 32)(x2)
    x3 = Down(2*c, 4*c)(x2, t)
    x3 = SelfAttention(4*c, 16)(x3)
    x4 = Down(4*c, 4*c)(x3, t)   
    x4 = SelfAttention(4*c, 8)(x4) 

    x4 = DoubleConv(4*c, 8*c)(x4)
    x4 = DoubleConv(8*c, 8*c)(x4)
    x4 = DoubleConv(8*c, 4*c)(x4)

    x = Up(8*c, 2*c)(x4, x3, t)
    x = SelfAttention(2*c, 16)(x)
    x = Up(4*c, c)(x, x2, t)
    x = SelfAttention(c, 32)(x)
    x = Up(2*c, c)(x, x1, t)
    x = SelfAttention(c, 64)(x)
    output_layer = Conv2D(3, kernel_size=3, padding='same', use_bias=False)(x)

    model = Model(inputs=[img_input, t_input], outputs=output_layer)
    optimizer = Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss='mse')

    return model