# Draft: Does a GPT read, or just remember?

Status: predictions only. Everything below waits for the d20 run.

## Predictions

Written on 2026-11-01, before the run. Each is about a 20-block, 1,280-dim model with 477M
parameters, trained on 5.7B tokens of web text, about 0.9 of a pass over 120 shards.

1. **Loss.** Validation loss ends near 2.9 nats per token, or about 0.9 bits per byte. The
   trigram table comes in around 7 bits per token, against about 4.2 for the model.
2. **Samples.** By the end, paragraphs are grammatical and stay on topic, but facts are
   often wrong.
3. **Context.** Text keeps helping far past what a small model used. Most of the gain still
   arrives by 128 tokens, and the curve is flat by 1,024.
4. **Random repetition.** It finally helps, by over 0.5 nats against the same tokens
   shuffled. The run is inside the range where Olsson et al. saw induction heads, though
   web text rarely repeats a long string exactly.
5. **Finishing an invented name.** The median name's last token gets P > 0.1 once the name
   has appeared, and the margin passes 3 nats.
6. **Bringing a name back.** Samples write the invented name back in over a third of
   continuations. This is the prediction that separates reading from remembering.
7. **Depth.** Early blocks supply most of the general improvement, and name-related
   predictions appear only in the last third of the blocks.
