import tensorflow as tf

@tf.function
def train_step(gan,batch,opt):

    gen = gan[0]
    dis = gan[1]
    opt_gen = opt[0]
    opt_dis = opt[1]
    batch_size = tf.shape(batch)[0]

    noise = tf.random.normal((batch_size,128))

    with tf.GradientTape(persistent=True) as tape:
        fake_imgs = gen(noise,training=True)
        fake_logits = dis(fake_imgs,training=True)
        true_logits = dis(batch,training=True)

        gen_loss = tf.reduce_mean(tf.square(fake_logits-1))
        dis_loss = tf.reduce_mean(tf.square(true_logits-1) + tf.square(fake_logits))/2.
    
    grads = tape.gradient(gen_loss,gen.trainable_variables)
    norm_grad = tf.linalg.global_norm(grads)
    opt_gen.apply_gradients(zip(grads,gen.trainable_variables))

    grads = tape.gradient(dis_loss,dis.trainable_variables)
    opt_dis.apply_gradients(zip(grads,dis.trainable_variables))

    del tape

    return gen_loss, dis_loss, norm_grad