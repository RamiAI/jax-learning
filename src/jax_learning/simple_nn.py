from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import tree_util


def init_mlp_params(layer_widths):
    params = []

    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        params.append({'weights': np.random.normal(size=(n_in, n_out)) * np.sqrt(2 / n_in),
                       'biases': np.random.normal(size=(n_out,))})
    return params


def forward(params, x):
    *hidden, last = params

    for layer in hidden:
        x = jax.nn.relu(x @ layer['weights'] + layer['biases'])

    return jnp.dot(x, last['weights']) + last['biases']


def loss_fn(params, x, y):
    return jnp.mean((forward(params, x) - y) ** 2)


@partial(jax.jit, static_argnames=['lr'])
def update(params, x, y, lr):
    grads = jax.grad(loss_fn)(params, x, y)

    return tree_util.tree_map(
        lambda p, g: p - lr * g,
        params,
        grads,
    )


def main():
    lr = 1e-2

    params = init_mlp_params([1, 64, 64, 1])
    print(tree_util.tree_map(lambda x: x.shape, params))

    xs = np.random.normal(size=(128, 1))
    ys = np.sin(3 * xs)

    num_epochs = 5_000
    for _ in range(num_epochs):
        params = update(params, xs, ys, lr)

    plt.scatter(xs, ys)
    plt.scatter(xs, forward(params, xs), label="Model prediction")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
