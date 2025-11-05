import keras
import keras.layers as nn
import tensorflow as tf

def build_generator(summary=False,weights=None):
    x = inputs = nn.Input((128,))
    x = nn.Dense(4*4*1024)(x)
    x = nn.LayerNormalization()(x)
    x = nn.ReLU()(x)
    x = nn.Reshape((4,4,1024))(x)

    x = nn.Conv2DTranspose(512,3,2,'same')(x)
    x = nn.LayerNormalization()(x)
    x = nn.ReLU()(x)
    x = nn.Conv2DTranspose(256,3,2,'same')(x)
    x = nn.LayerNormalization()(x)
    x = nn.ReLU()(x)
    x = nn.Conv2DTranspose(128,3,2,'same')(x)
    x = nn.LayerNormalization()(x)
    x = nn.ReLU()(x)
    x = nn.Conv2D(3,3,1,'same',activation='tanh')(x)


    generator = keras.Model(inputs,x, name='Generator')
    if summary: generator.summary()
    if weights is not None: generator.load_weights(weights)
    return generator

def build_discriminator(summary=False,weights=None):
    x = inputs = nn.Input((32,32,3))
    x = nn.SpectralNormalization(nn.Conv2D(128,5,2,'same',kernel_initializer='he_normal'))(x)
    x = nn.LeakyReLU(.2)(x)
    x = nn.SpectralNormalization(nn.Conv2D(256,5,2,'same',kernel_initializer='he_normal'))(x)
    x = nn.LeakyReLU(.2)(x)
    x = nn.SpectralNormalization(nn.Conv2D(256,5,2,'same',kernel_initializer='he_normal'))(x)
    x = nn.LeakyReLU(.2)(x)
    x = nn.SpectralNormalization(nn.Conv2D(512,5,1,'same',kernel_initializer='he_normal'))(x)
    x = nn.LeakyReLU(.2)(x)
    x = nn.SpectralNormalization(nn.Conv2D(1,3,1,'same'))(x)

    discriminator = keras.Model(inputs,x,name='Discriminator')
    if summary: discriminator.summary()
    if weights is not None: discriminator.load_weights(weights)