import pickle
from functools import partial
from pathlib import Path
import time
import jax
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.data import Dataset
import tensorflow as tf
from jax import random, Array, vmap, grad, jit
import jax.numpy as jnp
from jax.nn import swish, logsumexp, one_hot

ROWS = 3
COLS = 10
CHANNELS = 1
HEIGHT = WIDTH = 28
NUM_PIXELS = HEIGHT * WIDTH * CHANNELS
LABELS = 10

# Jax NN
LAYER_SIZES = [NUM_PIXELS, 512, LABELS]
PARAM_SCALE = 0.01

# Training
INIT_LR = 1.0
DECAY_RATE = 0.95
DECAY_STEPS = 5
NUM_EPOCHS = 15

MODEL_OUTPUT_FILE = "mlp_weights.pickle"


def print_dataset_sample(dataset: Dataset) -> None:
    fig, ax = plt.subplots(ROWS, COLS, figsize=(10, 5))
    for i, (image, label) in enumerate(dataset.take(ROWS * COLS)):
        ax[int(i / COLS), i % COLS].axis("off")
        ax[int(i / COLS), i % COLS].set_title(f"{label.numpy()}")
        ax[int(i / COLS), i % COLS].imshow(np.reshape(image, (HEIGHT, WIDTH)), cmap="gray")

    plt.show()


def preprocess(img, label) -> tuple[tf.Tensor, tf.Tensor]:
    return tf.cast(img, tf.float32) / 255.0, label


def init_network_params(sizes, key=random.PRNGKey(0), scale=1e-2):
    def random_layer_params(m, n, key, scale: float = 1e-2) -> Array:
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k, scale) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def predict(params: list[Array], image: Array) -> Array:
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = swish(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits


def loss(params, images, targets, batched_predict):
    logits = batched_predict(params, images)

    logs_preds = logits - logsumexp(logits)

    return -jnp.mean(targets * logs_preds)


@partial(jax.jit, static_argnames=["batched_predict"])
def update(params, x, y, epoch_numer, batched_predict):
    loss_value, grads = jax.value_and_grad(loss)(params, x, y, batched_predict)
    lr = INIT_LR * DECAY_RATE ** (epoch_numer / DECAY_STEPS)

    return [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)], loss_value


@partial(jax.jit, static_argnames=["batched_predict"])
def batch_accuracy(params, images, targets, batched_predict):
    images = jnp.reshape(images, (len(images), NUM_PIXELS))
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == targets)


def accuracy(params, data, batched_predict):
    accs = []
    for images, targets in data:
        accs.append(batch_accuracy(params, images, targets, batched_predict))
    return jnp.mean(jnp.array(accs))


def main() -> None:
    dataset_dir = Path("D:/") / "datasets"

    data, info = tfds.load(name="mnist", data_dir=dataset_dir, with_info=True, as_supervised=True)

    data_train = data["train"]
    data_test = data["test"]

    train_data = tfds.as_numpy(data_train.map(preprocess).batch(32).prefetch(1))
    test_data = tfds.as_numpy(data_test.map(preprocess).batch(32).prefetch(1))

    params = init_network_params(LAYER_SIZES, random.PRNGKey(42), scale=PARAM_SCALE)

    random_flattened_image = random.normal(
        random.PRNGKey(1),
        (
            32,
            NUM_PIXELS,
        ),
    )

    batched_predict = vmap(predict, in_axes=(None, 0))
    batched_preds = batched_predict(params, random_flattened_image)

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        losses = []
        for x, y in train_data:
            x = jnp.reshape(x, (len(x), NUM_PIXELS))
            y = one_hot(y, LABELS)
            params, loss_value = update(params, x, y, epoch, batched_predict)
            losses.append(loss_value)
        epoch_time = time.time() - start_time

        start_time = time.time()
        train_acc = accuracy(params, train_data, batched_predict)
        test_acc = accuracy(params, test_data, batched_predict)
        eval_time = time.time() - start_time
        print(f"Epoch {epoch} in {epoch_time:.2f} sec")
        print(f"Evaluation time: {eval_time:.2f} sec")
        print(f"Train loss: {jnp.mean(jnp.array(losses)):.4f}")
        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        print("=" * 10)

    with open(MODEL_OUTPUT_FILE, "wb") as f:
        pickle.dump(params, f)




if __name__ == "__main__":
    main()