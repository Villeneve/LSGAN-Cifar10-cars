import keras
import keras.layers as nn
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import sys

from src.models import build_generator

def main():
    if len(sys.argv) < 2:
        print('Give grid lenght for images')
        return
    generator = build_generator(summary=True, weights='weights/gen.weights.h5')
    noise = tf.random.normal((5,128),0,1)
    imgs = generator.predict(noise)
    k = int(sys.argv[1])
    fig, ax = plt.subplots(k,k,figsize=(8,8))
    ax = ax.ravel()
    noises = tf.random.normal((k**2,128),0,1)
    clip = tf.clip_by_value(noises,-1,1)
    nums = generator(clip).numpy()
    for i in range(k**2):
        ax[i].imshow(tf.cast((nums[i]*127.5+127.5),tf.uint8))
        ax[i].axis(False)
    plt.tight_layout(pad=0)
    plt.savefig('generate_batch.png')
    plt.close()

if __name__ == '__main__':
    main()