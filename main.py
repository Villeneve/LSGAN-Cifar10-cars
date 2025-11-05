#%%
import keras
import keras.layers as nn
import tensorflow as tf
from tqdm import tqdm

from src.models import *
from src.utils import train_step

import numpy as np
import matplotlib.pyplot as plt

#%%
l2 = keras.regularizers.l2(1e-4)
batch_size = 32

#%%
(trainx,trainy),(_,_) = keras.datasets.cifar10.load_data()
trainx = tf.cast(trainx[trainy[:,0]==1],tf.float32)/127.5-1
data = tf.data.Dataset.from_tensor_slices((trainx)).cache().shuffle(1024).map(lambda x: tf.image.random_flip_left_right(x)).batch(batch_size,drop_remainder=True).prefetch(tf.data.AUTOTUNE)

#%%
generator = build_generator()
discriminator = build_discriminator()

#%%
adam = keras.optimizers.Adam
optimizers = [adam(1e-4,beta_1=.5),adam(1e-4,beta_1=.5)]

#%%
gl, dl = [],[]
gl_mean, dl_mean = keras.metrics.Mean(),keras.metrics.Mean()

#%%
tqdm_loader = tqdm(range(1000+1),'Training',unit='epoch')

for epoch in tqdm_loader:
    tqdm_data = data

    for batch in tqdm_data:
        gl_,dl_,norm_grad = train_step([generator,discriminator],batch,optimizers)
        gl_mean.update_state(gl_)
        dl_mean.update_state(dl_)
    gl_ = gl_mean.result()
    dl_ = dl_mean.result()
    gl.append(gl_)
    dl.append(dl_)
    gl_mean.reset_state()
    dl_mean.reset_state()
    tqdm_loader.set_postfix(G_loss=f'{gl_:.4f}', D_loss=f'{dl_:.4f}', normD=f'{norm_grad:.4f}')
    if epoch%10 == 0:
        discriminator.save_weights('dis.weights.h5')
        generator.save_weights('gen.weights.h5')
    if epoch%100 == 0:
        k = 10
        fig, ax = plt.subplots(k,k,figsize=(8,8))
        ax = ax.ravel()
        noises = tf.random.normal((k**2,128),0,1)
        clip = tf.clip_by_value(noises,-1,1)
        nums = generator(clip).numpy()
        for i in range(k**2):
            ax[i].imshow(np.uint8(nums[i]*127.5+127.5))
            ax[i].axis(False)
        plt.tight_layout(pad=0)
        plt.savefig('img.png')
        plt.close()
    if epoch%10 == 0:
        plt.plot(gl,label='gl')
        plt.plot(dl,label='dl')
        plt.legend()
        plt.ylim([-.1,1.5])
        plt.savefig('training_curve.png')
        plt.close()
