# Model
model = SelfAttentionClassifier(
    input_dim=config.embedding_dim,
    output_dim=1,
    num_layers=2,
    num_heads=4,
    qkv_dim=256,
    key=jax.random.PRNGKey(10)
)

# Optimizer
optimizer = optax.adam(learning_rate=config.learning_rate)
opt_state = optimizer.init(model)

# Value and gradient
loss_and_grad_fn = jax.value_and_grad(loss_fn)

# TensorBoard writer
writer = SummaryWriter(logdir="./4-attention/" + time.strftime("%Y%m%d-%H%M%S"))

# Training loop
step = 0
for epoch in range(config.num_epochs):
    for  batch in train_dataloader:
        
        loss_val, grads = loss_and_grad_fn(model, batch)
        
        # Apply gradients
        updates, opt_state = optimizer.update(grads, opt_state)
        leaves, _ = jax.tree_util.tree_flatten(updates)
        model = optax.apply_updates(model, updates)

        # Log to TensorBoard
        writer.add_scalar("train/loss", float(loss_val), step)
        writer.add_scalar("train/learning_rate", config.learning_rate, step)
        writer.add_scalar("gradients/0/mean", leaves[0].mean(), step)
        writer.add_scalar("gradients/1/mean", leaves[1].mean(), step)
        writer.add_scalar("gradients/2/mean", leaves[2].mean(), step)

        num_missing = np.mean(batch['embedding_missing'].sum(axis=1) - batch['num_pad'])
        writer.add_scalar("data/padding", batch['num_pad'].mean(), step)
        writer.add_scalar("data/num_missing", num_missing, step)
        writer.add_scalar("label/mean", batch['label'].mean(), step)


        # Increment step
        step += 1

    writer.flush()
writer.close()
