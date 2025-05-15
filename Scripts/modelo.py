from tensorflow import keras
from sdx import *

# Aqui tá importando o MNIST
(train_images, train_labels), (test_images, test_labels) = (
    keras.datasets.mnist.load_data()
)

# ──> Pré-processamento: normalização e canal explícito
train_images = train_images.reshape(-1, 28, 28, 1).astype("float32") / 255.0
test_images = test_images.reshape(-1, 28, 28, 1).astype("float32") / 255.0

print(train_images.shape)


def compile_and_summary(model):
    model.compile(
        optimizer=keras.optimizers.Adam(),  # adiciona otimizador
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],  # adiciona métrica
    )
    model.summary()


def fit_default_parameters(model):
    history = model.fit(
        train_images,
        train_labels,
        epochs=32,
        batch_size=32,
        validation_data=(test_images, test_labels),
    )
    return history


model = keras.Sequential(
    [
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Flatten(),
        keras.layers.Dense(397, activation="relu"),  # sugerido: ativação ReLU
        keras.layers.Dense(10),  # logits para SparseCategoricalCrossentropy
    ]
)

# compile_and_summary(model)

# # ──> Executa o treinamento
# history = fit_default_parameters(model)
