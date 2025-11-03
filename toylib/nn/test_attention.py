
# in_ = 10
# out_ = (8, 12)
# x = np.zeros((4, in_))
# w = np.zeros((in_, 4, 12))
# jax.numpy.dot(x, w).shape

# nn.attention.scaled_dot_product_attention(q, k, v)
# mha = nn.attention.MultiHeadAttention(num_heads=2, qkv_dim=8, key=jax.random.PRNGKey(10))

# values, att_mask = mha(q, k, v, mask=None)
# print(values.shape, att_mask.shape)

# in_ = 10
# out_ = (8, 12)
# x = np.zeros((4, in_))
# out = encoder(x, mask=None)
# print(out.shape)