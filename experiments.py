"""Shared simulation/CLI for complex linear and real activation experiments.

No plotting imports. New runs use a documented v1 schema and new RNG streams;
legacy cluster archives remain readable, but are not bitwise rerun by this code.
"""
import argparse
from functools import lru_cache
from pathlib import Path
import numpy as np

ACTIVATIONS = ("linear", "relu", "leaky_relu")


def calibrated_scale(activation, alpha):
    return np.sqrt(
        2
        if activation == "relu"
        else 2 / (1 + alpha**2)
        if activation == "leaky_relu"
        else 1
    )


def numpy_chunk(
    seed,
    n,
    depth,
    size,
    ensemble,
    activation,
    alpha,
    scale,
    x_mode,
    initial_norm,
    model,
    lru_inputs="unit",
    weight_tying="tied",
):
    rng = np.random.default_rng(seed)
    if x_mode == "random_isotropic":
        initial_norm *= np.sqrt(n)

    def draw(shape):
        z = rng.standard_normal(shape)
        return (
            (z + 1j * rng.standard_normal(shape)) / np.sqrt(2)
            if ensemble == "complex"
            else z
        )

    w = scale * draw((size, n, n)) / np.sqrt(n)

    def draw_input():
        if lru_inputs == "gaussian":
            return rng.standard_normal((size, n))
        if x_mode == "e1":
            x = np.zeros((size, n))
            x[:, 0] = initial_norm
            return x
        x = draw((size, n))
        return x * initial_norm / np.linalg.norm(x, axis=1, keepdims=True)

    if model == "lru":
        h = draw_input()
    elif x_mode == "e1":
        h = np.zeros((size, n))
        h[:, 0] = initial_norm
    else:
        h = draw((size, n))
        h *= initial_norm / np.linalg.norm(h, axis=1, keepdims=True)
    for layer in range(depth):
        if weight_tying == "untied" and layer > 0:
            w = scale * draw((size, n, n)) / np.sqrt(n)
        h = np.einsum("bij,bj->bi", w, h)
        if model == "lru":
            h += draw_input()
        elif activation == "relu":
            h = np.maximum(h, 0)
        elif activation == "leaky_relu":
            h = np.where(h >= 0, h, alpha * h)
    return np.sum(np.abs(h) ** 2, axis=1)


@lru_cache(None)
def jax_kernel(n, depth, size, ensemble, activation, x_mode, model, lru_inputs,
               weight_tying="tied"):
    # Lazy import lets plotting and CPU-only installations work without JAX.
    import jax
    import jax.numpy as jnp

    @jax.jit
    def run(key, alpha, scale, initial_norm):
        wk, hk, ik = jax.random.split(key, 3)
        if x_mode == "random_isotropic":
            initial_norm = initial_norm * jnp.sqrt(float(n))

        def draw(k, shape):
            if ensemble == "complex":
                kr, ki = jax.random.split(k)
                return (
                    jax.random.normal(kr, shape) + 1j * jax.random.normal(ki, shape)
                ) / jnp.sqrt(2.0)
            return jax.random.normal(k, shape)

        w = scale * draw(wk, (size, n, n)) / jnp.sqrt(float(n))

        def draw_input(k):
            if lru_inputs == "gaussian":
                return jax.random.normal(k, (size, n))
            if x_mode == "e1":
                return jnp.zeros((size, n)).at[:, 0].set(initial_norm)
            x = draw(k, (size, n))
            return x * initial_norm / jnp.linalg.norm(x, axis=1, keepdims=True)

        if model == "lru":
            h = draw_input(hk)
        elif x_mode == "e1":
            h = jnp.zeros((size, n)).at[:, 0].set(initial_norm)
        else:
            h = draw(hk, (size, n))
            h *= initial_norm / jnp.linalg.norm(h, axis=1, keepdims=True)
        if ensemble == "complex":
            h = h.astype(jnp.complex64)

        def step(carry, layer):
            h, key = carry
            key, sub = jax.random.split(key)
            if weight_tying == "untied":
                # Independent layer keys; never materialize a depth x batch x n x n array.
                layer_w = jax.lax.cond(
                    layer == 0,
                    lambda _: w,
                    lambda _: scale * draw(jax.random.fold_in(wk, layer), (size, n, n))
                              / jnp.sqrt(float(n)),
                    operand=None,
                )
            else:
                layer_w = w
            h = jnp.einsum("bij,bj->bi", layer_w, h)
            if model == "lru":
                h += draw_input(sub)
            elif activation == "relu":
                h = jax.nn.relu(h)
            elif activation == "leaky_relu":
                h = jax.nn.leaky_relu(h, negative_slope=alpha)
            return (h, key), None

        (h, _), _ = jax.lax.scan(step, (h, ik), jnp.arange(depth))
        return jnp.sum(jnp.abs(h) ** 2, axis=1)

    return run


def compute_experiment_data(
    save_path,
    *,
    ensemble="complex",
    activation="linear",
    model="rnn",
    n_values=(100, 200, 500, 1000, 2000),
    c_min=0.0,
    c_max=5.0,
    num_c_points=20,
    num_samples=1000,
    chunk_size=32,
    seed=42,
    x_mode="e1",
    initial_norm=1.0,
    alpha=0.5,
    weight_scale=None,
    backend="numpy",
    overwrite=False,
    lru_inputs="unit",
    weight_tying="tied",
):
    path = Path(save_path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; choose another output or use --overwrite"
        )
    if (
        ensemble not in ("real", "complex")
        or activation not in ACTIVATIONS
        or model not in ("rnn", "lru")
    ):
        raise ValueError("Unknown ensemble, activation, or model")
    if weight_tying not in ("tied", "untied"):
        raise ValueError("weight_tying must be tied or untied")
    if weight_tying == "untied" and model != "rnn":
        raise ValueError("Untied weights currently support RNN experiments only")
    if activation != "linear" and (ensemble == "complex" or model == "lru"):
        raise ValueError("Nonlinear activations require a real RNN")
    if backend not in ("numpy", "jax") or x_mode not in ("e1", "random_unit", "random_isotropic"):
        raise ValueError("Unknown backend or initial state")
    if (
        min(num_samples, chunk_size, num_c_points) < 1
        or not n_values
        or min(n_values) < 1
    ):
        raise ValueError(
            "Widths, sample count, chunk size, and grid size must be positive"
        )
    scale = (
        calibrated_scale(activation, alpha) if weight_scale is None else weight_scale
    )
    if not np.all(np.isfinite([c_min, c_max, alpha, initial_norm, scale])) or not (
        0 <= c_min <= c_max and alpha >= 0 and initial_norm > 0 and scale > 0
    ):
        raise ValueError("Invalid depth range, alpha, initial norm, or weight scale")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if lru_inputs not in ("unit", "gaussian"):
        raise ValueError("Unknown LRU input distribution")
    if lru_inputs == "gaussian" and (
        model != "lru" or initial_norm != 1 or x_mode != "e1"
    ):
        raise ValueError(
            "Gaussian inputs require LRU mode and default unit-state options"
        )
    c_values = np.linspace(c_min, c_max, num_c_points)
    rows = []
    for n in n_values:
        for point, depth in enumerate(np.rint(c_values * np.sqrt(n)).astype(int)):
            count, mean, m2 = 0, 0.0, 0.0
            for chunk, start in enumerate(range(0, num_samples, chunk_size)):
                size = min(chunk_size, num_samples - start)
                chunk_seed = np.random.SeedSequence([seed, int(n), point, chunk])
                if backend == "numpy":
                    values = numpy_chunk(
                        chunk_seed,
                        n,
                        int(depth),
                        size,
                        ensemble,
                        activation,
                        alpha,
                        scale,
                        x_mode,
                        initial_norm,
                        model,
                        lru_inputs,
                        weight_tying,
                    )
                else:
                    import jax

                    key = jax.random.PRNGKey(int(chunk_seed.generate_state(1)[0]))
                    values = jax_kernel(
                        int(n),
                        int(depth),
                        size,
                        ensemble,
                        activation,
                        x_mode,
                        model,
                        lru_inputs,
                        weight_tying,
                    )(key, alpha, scale, initial_norm)
                values = np.asarray(values, dtype=np.float64)
                if not np.isfinite(values).all():
                    raise FloatingPointError(
                        f"Nonfinite samples at n={n}, depth={depth}; no result saved"
                    )
                cm = values.mean()
                delta = cm - mean
                new_count = count + size
                m2 += np.sum((values - cm) ** 2) + delta**2 * count * size / new_count
                mean += delta * size / new_count
                count = new_count
            std = np.sqrt(m2 / (count - 1)) if count > 1 else 0.0
            rows.append((n, depth, depth / np.sqrt(n), mean, std, std / np.sqrt(count)))
            print(
                f"{ensemble}/{activation}/{model}/{weight_tying}: n={n} t={depth} mean={mean:.6g} SE={std/np.sqrt(count):.3g}",
                flush=True,
            )
    arrays = dict(
        zip(("width", "depth", "c_eff", "mean", "std", "se"), np.array(rows).T)
    )
    arrays["width"] = arrays["width"].astype(int)
    arrays["depth"] = arrays["depth"].astype(int)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open the exact filename, preventing NumPy from silently adding an extension.
    with path.open("wb") as f:
        np.savez_compressed(
            f,
            **arrays,
            schema_version=1,
            ensemble=ensemble,
            activation=activation,
            model=model,
            weight_tying=weight_tying,
            alpha=alpha,
            weight_scale=scale,
            x_mode=x_mode,
            initial_norm=initial_norm,
            initial_norm_scaling="sqrt_width" if x_mode == "random_isotropic" else "constant",
            initial_energy=(arrays["width"] * initial_norm**2
                            if x_mode == "random_isotropic" or lru_inputs == "gaussian"
                            else np.full(len(rows), initial_norm**2)),
            n_values=n_values,
            c_values=c_values,
            num_samples=num_samples,
            chunk_size=chunk_size,
            seed=seed,
            backend=backend,
            observable="total_squared_norm",
            rng_scheme=("v1: seed,width,grid_index,chunk_index" if weight_tying == "tied"
                        else "v1 untied: seed,width,grid_index,chunk_index; "
                             "NumPy sequential layer draws; JAX fold_in(weight_key,layer)"),
            lru_inputs=lru_inputs,
            input_distribution=(
                "independent real N(0,I) per step"
                if lru_inputs == "gaussian"
                else "repeated e1"
                if x_mode == "e1"
                else "independent radius-sqrt(n) spherical input per step"
                if x_mode == "random_isotropic"
                else "independent random unit per step"
            )
            if model == "lru"
            else x_mode,
            arithmetic="float64/complex128"
            if backend == "numpy"
            else "JAX default precision",
        )
    return path


def main(default_ensemble, default_activation):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ensemble", choices=["real", "complex"], default=default_ensemble)
    p.add_argument("--activation", choices=ACTIVATIONS, default=default_activation)
    p.add_argument("--lru-inputs", choices=["unit", "gaussian"], default="unit")
    p.add_argument("--model", choices=["rnn", "lru"], default="rnn")
    p.add_argument("--weight-tying", choices=["tied", "untied"], default="tied",
                   help="tied: reuse W; untied: independent W at every layer (RNN only)")
    p.add_argument("--widths", nargs="+", type=int, default=[100, 200, 500, 1000, 2000])
    p.add_argument("--c-min", type=float, default=0.0)
    p.add_argument("--c-max", type=float, default=5.0)
    p.add_argument("--points", type=int, default=20)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--x-mode", choices=["e1", "random_unit", "random_isotropic"], default="e1",
                   help="random_unit: radius 1; random_isotropic: radius sqrt(n), covariance I; both scaled by --initial-norm")
    p.add_argument("--initial-norm", type=float, default=1.0,
                   help="Input radius; for random_isotropic, multiplier of sqrt(n)")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--weight-scale", type=float)
    p.add_argument("--backend", choices=["numpy", "jax"], default="numpy")
    p.add_argument("--overwrite", action="store_true")
    a = vars(p.parse_args())
    a["save_path"] = a.pop("output")
    a["n_values"] = a.pop("widths")
    a["num_c_points"] = a.pop("points")
    a["num_samples"] = a.pop("samples")
    print("Saved:", compute_experiment_data(**a))


if __name__ == "__main__":
    main(default_ensemble="real", default_activation="linear")
