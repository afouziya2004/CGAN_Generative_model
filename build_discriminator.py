def build_discriminator():
    image = layers.Input(shape=(32, 32, 3))
    label = layers.Input(shape=(1,), dtype='int32')

    label_embedding = layers.Embedding(num_classes, 32*32*3)(label)
    label_embedding = layers.Flatten()(label_embedding)
    label_embedding = layers.Reshape((32, 32, 3))(label_embedding)

    model_input = layers.Concatenate()([image, label_embedding])

    x = layers.Flatten()(model_input)
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(0.2)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model([image, label], output)
