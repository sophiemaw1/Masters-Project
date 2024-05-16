from sample_obj import get_flvae_data_from_objs
from ring_vae import ring_encoder, ring_decoder, Sampling

import jax
import jax.numpy as jnp
import jax.random as rnd
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn
from flax import serialization
from flax.training import train_state
from jax.random import split
from sklearn.model_selection import train_test_split

lambda_kl = 0.01


class FLVAE(nn.Module):
    num_samples: int = 64
    latent_dim: int = 8
    num_heads: int = 8
    features: int = 16

    def setup(self):
        self.encoder = ring_encoder(features=self.num_samples, latent_dim=self.latent_dim, num_heads=self.num_heads)
        self.decoder = ring_decoder(features=self.num_samples, out_features=3)
        self.sampling = Sampling()

    @nn.compact
    def __call__(self, batch, z_rng):
        ring_logs = batch['ring_logs']
        ring_pix = batch['ring_pix']
        pix_tri = batch['pix_tri']
        pix_logs = batch['pix_logs']

        mean, logvar = self.encoder(ring_logs, ring_pix)

        z_rng, sample_rng = split(z_rng)
        z = self.sampling(mean, logvar, sample_rng)

        def decode_fn(pix_tri_slice, pix_logs_slice):
            return self.decoder(z, pix_tri_slice, pix_logs_slice)

        # Vectorize the decoder operation over the batch and samples
        all_batches_recon_x = jax.vmap(jax.vmap(decode_fn))(pix_tri, pix_logs)

        return all_batches_recon_x, mean, logvar


def loss_function(flvae, batch, z_rng, params, debug=False):
    recon_x, mean, logvar = flvae({'params': params}, batch, z_rng)

    mean = jnp.abs(mean)
    logvar = jnp.abs(logvar)

    # Normalize the barycentric coordinates (pix_logs)
    pix_logs = batch['pix_logs']
    max_log = jnp.max(pix_logs, axis=-1, keepdims=True)
    weights = jnp.exp(pix_logs - max_log) / jnp.sum(jnp.exp(pix_logs - max_log), axis=-1, keepdims=True)
    weights = jnp.sum(weights, axis=-1, keepdims=True)
    weights = jnp.broadcast_to(weights, recon_x.shape[:-1] + (1,))

    # Barycentric interpolation
    interpolated_recon_x = jnp.sum(recon_x * weights, axis=2)

    debug_info = {}
    if debug:
        actual_rgb_numpy = jax.device_get(batch['ring_pix'][0, 0])
        predicted_rgb_numpy = jax.device_get(interpolated_recon_x[0, 0])
        ring_pix_min = jax.device_get(jnp.min(batch['ring_pix']))
        ring_pix_max = jax.device_get(jnp.max(batch['ring_pix']))
        recon_x_min = jax.device_get(jnp.min(recon_x))
        recon_x_max = jax.device_get(jnp.max(recon_x))

        debug_info = {
            'actual_rgb_numpy': actual_rgb_numpy,
            'predicted_rgb_numpy': predicted_rgb_numpy,
            'ring_pix_min': ring_pix_min,
            'ring_pix_max': ring_pix_max,
            'recon_x_min': recon_x_min,
            'recon_x_max': recon_x_max
        }

    recon_loss = jnp.mean((interpolated_recon_x - batch['ring_pix']) ** 2)

    logvar = jnp.where(logvar < 0, 0, logvar)
    kl_loss = -0.5 * jnp.mean(1 + logvar - jnp.square(mean) - jnp.exp(logvar.clip(min=-10)), axis=1)

    # Total loss
    total_loss = recon_loss + lambda_kl * jnp.mean(kl_loss)
    if debug:
        return total_loss, debug_info
    else:
        return total_loss


@jax.jit
def train_step(state, batch, z_rng):
    def loss_fn(params):
        return loss_function(state.apply_fn, batch, z_rng, params, False)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


@jax.jit
def validate_step_(state, batch, rng):
    return loss_function(state.apply_fn, batch, rng, state.params, True)


@jax.jit
def validate_step(state, batch, rng):
    return loss_function(state.apply_fn, batch, rng, state.params,True)


def train_model(num_epochs, vae, train_dataset, test_dataset, rng):
    train_losses, val_losses = [], []

    num_train_samples = train_dataset['ring_logs'].shape[0] * train_dataset['ring_logs'].shape[1]
    decay_epochs = 10
    steps_per_epoch = num_train_samples
    transition_steps = steps_per_epoch * decay_epochs

    lr_schedule = optax.exponential_decay(init_value=1e-3, transition_steps=transition_steps, decay_rate=0.99, staircase=True)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients at norm 1
        optax.adam(learning_rate=lr_schedule)
    )

    # Use a sample batch for initialization
    sample_batch = {k: v[0] for k, v in train_dataset.items()}
    params = vae.init(rnd.PRNGKey(0), sample_batch, rng)['params']
    state = train_state.TrainState.create(apply_fn=vae.apply, params=params, tx=optimizer)

    # Early stopping variables
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    val_increase_count = 0
    train_no_decrease_count = 0
    early_stopping_patience = 30

    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_losses, epoch_val_losses = [], []

        for obj_idx in range(train_dataset['ring_logs'].shape[0]):
            for one_ring_idx in range(train_dataset['ring_logs'].shape[1]):
                batch = {k: v[obj_idx, one_ring_idx] for k, v in train_dataset.items()}
                batch = {k: v[jnp.newaxis, ...] for k, v in batch.items()}  # Add batch dimension
                state, loss = train_step(state, batch, rng)
                epoch_losses.append(loss)

        for obj_idx in range(test_dataset['ring_logs'].shape[0]):
            for one_ring_idx in range(test_dataset['ring_logs'].shape[1]):
                batch = {k: v[obj_idx, one_ring_idx] for k, v in test_dataset.items()}
                batch = {k: v[jnp.newaxis, ...] for k, v in batch.items()}  # Add batch dimension

                val_loss, debug_info = validate_step(state, batch, rng)

                # Debug print to verify the condition
                # Debug print to verify the condition
                if (obj_idx == 0 and one_ring_idx == 0) and (epoch % 10 == 0):
                    print("Debug Info:")

                    # Handle debug info outside JIT
                    print(
                        f"First RGB: actual={debug_info['actual_rgb_numpy']}, predict={debug_info['predicted_rgb_numpy']}")

                epoch_val_losses.append(val_loss)

        train_losses.append(np.mean(epoch_losses))
        val_losses.append(np.mean(epoch_val_losses))

        avg_train_loss = np.mean(epoch_losses)
        avg_val_loss = np.mean(epoch_val_losses)

        print(f"Epoch: {epoch + 1}, Train Loss: {np.mean(epoch_losses)}, Val Loss: {np.mean(epoch_val_losses)}")

        # Early stopping check for validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            val_increase_count = 0
        else:
            val_increase_count += 1

        # Early stopping check for training loss
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            train_no_decrease_count = 0
        else:
            train_no_decrease_count += 1

        # Check if early stopping criteria are met
        if val_increase_count >= early_stopping_patience:
            print("Early stopping triggered due to validation loss not decreasing for 20 epochs.")
            break

        if train_no_decrease_count >= early_stopping_patience:
            print("Early stopping triggered due to training loss not decreasing for 20 epochs.")
            break

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.show()

    return state


def save_model(state,model_num, filename='model_state.msgpack'):
    filename = f'model_state_{model_num}.msgpack'
    bytes_data = serialization.to_bytes(state.params)
    with open(filename, 'wb') as f:
        f.write(bytes_data)
    print("Model saved to", filename)


def train_flvae_and_save(num_objs, num_samples, num_one_rings, num_epochs):
    rng = jax.random.PRNGKey(1)
    vae = FLVAE(num_samples=num_samples, latent_dim=8, num_heads=8, features=num_one_rings)


    ring_logs, ring_pix, pix_tri, pix_logs = get_flvae_data_from_objs(num_one_rings=num_one_rings,
                                                                          num_objs=num_objs,
                                                                          num_samples=num_samples)

    # Split each array into training and testing sets
    train_ring_logs, test_ring_logs = train_test_split(ring_logs, train_size=0.8, random_state=42)
    train_ring_pix, test_ring_pix = train_test_split(ring_pix, train_size=0.8, random_state=42)
    train_pix_tri, test_pix_tri = train_test_split(pix_tri, train_size=0.8, random_state=42)
    train_pix_logs, test_pix_logs = train_test_split(pix_logs, train_size=0.8, random_state=42)

    train_dataset = {
        'ring_logs': train_ring_logs,
        'ring_pix': train_ring_pix,
        'pix_tri': train_pix_tri,
        'pix_logs': train_pix_logs
    }
    test_dataset = {
        'ring_logs': test_ring_logs,
        'ring_pix': test_ring_pix,
        'pix_tri': test_pix_tri,
        'pix_logs': test_pix_logs
    }
    print(f"train_ring_logs: {train_ring_logs.shape}")
    print(f"train_ring_pix: {train_ring_pix.shape}")
    print(f"train_pix_tri: {train_pix_tri.shape}")
    print(f"train_pix_logs: {train_pix_logs.shape}")

    state = train_model(num_epochs, vae, train_dataset, test_dataset, rng)
    save_model(state,model_num=num_one_rings)


if __name__ == '__main__':
    test = [1,50,100,200]
    for n in test:
        num_objs = 750
        num_samples = 64
        num_one_rings = 200
        num_epochs = 200
        train_flvae_and_save(num_objs=num_objs, num_samples=num_samples, num_one_rings=num_one_rings, num_epochs=num_epochs)
