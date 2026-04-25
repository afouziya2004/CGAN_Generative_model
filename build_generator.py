from tensorflow.keras import layers

def build_generator():
    noise = layers.Input(shape=(100,))
    label = layers.Input(shape=(1,), dtype='int32')

    label_embedding = layers.Embedding(num_classes, 100)(label)
    label_embedding = layers.Flatten()(label_embedding)

    model_input = layers.multiply([noise, label_embedding])

    x = layers.Dense(256)(model_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(32 * 32 * 3, activation='tanh')(x)

    output = layers.Reshape((32, 32, 3))(x)

    return tf.keras.Model([noise, label], output)
