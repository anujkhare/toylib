from dataclasses import dataclass
from tensorboardX import SummaryWriter
from typing import Callable, Any, Optional
import jax
import jax.numpy as jnp
import optax
import time


@dataclass
class TrainerConfig:
    model_fn: Callable[[Any, jnp.ndarray], jnp.ndarray]
    loss_fn: Callable[[Any, tuple], jnp.ndarray]
    init_params_fn: Callable[[jax.random.PRNGKey], Any]
    optimizer_fn: Callable[[int], optax.GradientTransformation]  # receives num_steps
    train_batch_fn: Callable[[], tuple]
    val_batch_fn: Optional[Callable[[], tuple]] = None
    num_steps: int = 1000
    log_every: int = 100
    validate_every: int = 500
    log_dir: str = "./runs"
    rng: jax.random.PRNGKey = jax.random.PRNGKey(0)


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.cfg = config
        self.params = self.cfg.init_params_fn(self.cfg.rng)
        self.optimizer = self.cfg.optimizer_fn(self.cfg.num_steps)
        self.opt_state = self.optimizer.init(self.params)
        self.loss_and_grad = jax.value_and_grad(self.cfg.loss_fn)
        self.loss_fn = self.cfg.loss_fn  # for eval
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(f"{self.cfg.log_dir}/{timestamp}")

    def train(self):
        for step in range(self.cfg.num_steps):
            batch = self.cfg.train_batch_fn()
            loss, grads = self.loss_and_grad(self.params, batch)
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.params = optax.apply_updates(self.params, updates)

            if step % self.cfg.log_every == 0:
                print(f"[Step {step}] Train Loss: {loss:.4f}")
                self.writer.add_scalar("train/loss", float(loss), step)

            if self.cfg.val_batch_fn and step % self.cfg.validate_every == 0:
                val_batch = self.cfg.val_batch_fn()
                val_loss = self.loss_fn(self.params, val_batch)
                print(f"[Step {step}] Validation Loss: {val_loss:.4f}")
                self.writer.add_scalar("val/loss", float(val_loss), step)

        self.writer.close()
        return self.params