# Draft: How does a small GPT keep a story straight?

Status: first draft around a deliberately under-trained local model. The numbers and
figures will change after the TPU run.

## Predictions

Written on 2026-10-31, before the long local run. Each is about a 4-block, 256-dim model
trained on about 40M tokens, under a tenth of one pass over TinyStories.

1. **Loss.** Validation loss ends near 2.0 nats per token, or about 2.9 bits. The trigram
   table comes in around 4 bits, so the model's advantage is real but not huge.
2. **Samples.** By the end, sentences are grammatical and characters mostly stay the
   same, but the plot wanders.
3. **Context.** Most of the gain arrives by 32 tokens of context. Going from 128 to 255
   tokens changes almost nothing.
4. **Random repetition.** Repeating a random 40-token string doesn't help. Induction
   heads formed at 2.5 to 5 billion tokens in
   [Olsson et al., 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html),
   about 100 times this budget, and random strings look nothing like a story.
5. **Invented names.** Finishing an invented name does help. Stories repeat their
   characters' names constantly, so copying pays from early on, and a name is a far
   easier cue than a random string. Expect a margin of over 1 nat.
6. **Bringing a name back.** This one is harder than finishing a name, because the
   model has to decide unprompted that the name belongs in this slot. Expect a positive
   margin, but smaller than the one for finishing.
7. **Depth.** The first block supplies most of the improvement from the logit lens, and
   name-related predictions appear only from the second block on.

## How the predictions did

Checked against the finished run, 41M tokens. The predictions above are unchanged. The
run was repeated on 2026-11-01 with our own tokenizer and 24 invented names, and these
are the repeat's numbers. Prediction 6 held on the first run, with five names.

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 1 | 2.0 nats (2.9 bits); trigram about 4 bits | 1.74 nats at the last eval, 2.45 bits on the final model; trigram 4.47 | Too pessimistic |
| 2 | Grammatical, characters stay, plot wanders | That's what the samples do | Held |
| 3 | Most gain by 32 tokens; 128 to 255 changes almost nothing | 94% of the gain by 32; 128 to 255 is flat | Held |
| 4 | Random repetition doesn't help | 0.04 nats over the same tokens shuffled | Held |
| 5 | Finishing an invented name helps, by over 1 nat | 3.0 nats on average, but -1.0 to 8.8 by name, and usually still improbable | Held as worded, misleading as a claim |
| 6 | Bringing a name back is positive but smaller | Positive, but larger: 3.8 against 3.0 nats | Wrong on the bigger set |
| 7 | Block 1 gives most of the improvement; names from block 2 on | Block 1 gives 47% of it, the largest share; names show up only in block 4 (0.5 nats at block 3) | Half wrong |

The prediction that mattered most was also the one I framed badly. I asked whether a
name would help, and it does. The better question was whether the model would use it,
and it almost never does: invented names were written back in 3 of 288 sampled stories,
and common names in 68 of 72.

## What a small model does with a name

Give this model a story about Lily, and when a name is due, Lily gets about a 30% chance.
Give it a story about an invented name, and across 24 such names the same slot gives
that name's first token about 1 in 800. That's about 4,000 times likelier than with no
name in the story at all, and far too unlikely to be written. Nor is all of that the name
itself: a different invented name in the story already lifts it about 90 times.

So for now, keeping a story straight looks more like recognition than copying. The model
has learned familiar names as things in their own right, which the output
layer shows plainly: Lily sits with Mia, Anna and Ben. An invented name has no such place,
and the machinery that would carry an arbitrary token forward is only starting to form.
The curves are still climbing at step 5,000, which is the case for the TPU run.

Whether a bigger model copies, or just knows more names, is the question the post should
answer. This draft can't.
