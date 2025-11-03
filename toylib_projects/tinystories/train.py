import jax
import optax

# import local modules
from toylib_projects.tinystories import data
from toylib_projects.tinystories import decoder_only_model
from toylib_projects.tinystories import experiment


def main():
    config = experiment.Config(
        model_config=decoder_only_model.ModelConfig(
            vocab_size=50257,  # GPT-2 tokenizer vocab size
        ),
        training_config=experiment.TrainingConfig(),
    )

    # Dataloader
    dataset = data.BatchedTokenizedHFDataset(
        bos_token=1000, batch_size=128, seq_len=512, tokenizer_batch_size=8
    )

    # Model
    model = decoder_only_model.DecoderOnlyTransformer(
        config=config.model_config, key=jax.random.PRNGKey(0)
    )

    # Logger
    logger = experiment.TensorBoardLogger(config, output_path="./tensorboard_logs")

    # Optimizer
    optimizer = optax.adam(learning_rate=config.training_config.learning_rate)

    def log_metrics(logger: experiment.Logger, step: int, loss_val: float, updates):
        leaves, _ = jax.tree_util.tree_flatten(updates)
        metrics = {
            "train/loss": float(loss_val),
            "train/learning_rate": config.training_config.learning_rate,
            "gradients/0/mean": leaves[0].mean(),
            "gradients/1/mean": leaves[1].mean(),
            "gradients/2/mean": leaves[2].mean(),
        }
        logger.log(step=step, metrics=metrics)

    # Optimizer
    opt_state = optimizer.init(model)

    # Value and gradient
    loss_and_grad_fn = jax.jit(jax.value_and_grad(decoder_only_model.train_step))

    step = 0

    # Training loop
    for epoch in range(config.training_config.num_epochs):
        for batch in dataset:
            inputs, targets = batch["inputs"], batch["targets"]
            mask = jax.numpy.ones_like(inputs)

            # Compute loss and gradients
            loss_val, grads = loss_and_grad_fn(model, inputs, mask, targets)

            # Apply gradients
            updates, opt_state = optimizer.update(grads, opt_state)
            model = optax.apply_updates(model, updates)

            # Log metrics
            log_metrics(logger, step, loss_val, updates)

            # Increment step
            step += 1


if __name__ == "__main__":
    main()
