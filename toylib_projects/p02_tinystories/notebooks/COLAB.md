# Running the d20 notebook on Colab

`01_d20_climbmix_colab.ipynb` trains a 20-block GPT on ClimbMix, the web text nanochat uses.
This uses the same Colab CLI, service account and W&B login as post 1, so do the one-time
setup in [post 1's guide](../../p01_cnn_mnist/notebooks/COLAB.md) first. Everything below
is what differs. Commands run from the root of the repo.

| Setting | Value |
| --- | --- |
| Model | 20 blocks, 1,280 dims, 10 heads, 477M parameters |
| Bucket folder | `gs://tinystories-checkpoints/toylib-p02-climbmix` |
| W&B project | `toylib-p02-climbmix` |

Colab gets the code from GitHub, so **push first**. The notebook's `BRANCH` is
`p02-tinystories`. A runtime is billed while it exists, even when idle.

## Which runtime

Use an A100. The model peaked at 21 GB of RAM on a CPU, even with a tiny batch, and the
earlier run of this model had 59 GB on an A100. A `v5e1` TPU has about 16 GB, so it's
unlikely to fit without a smaller batch.

## Prepare a runtime

Each new runtime needs the keys, as in post 1. This one is a CPU runtime, for the token
files in the next step:

```bash
colab --auth=adc new -s p02data
```

Send the W&B key:

```bash
K=$(mktemp) && uv run python -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2], end='')" > "$K" && colab --auth=adc upload -s p02data "$K" /content/.wandb_key; rm -f "$K"
```

Send a service-account key:

```bash
D=$(mktemp -d) && gcloud iam service-accounts keys create "$D/key.json" --iam-account=toylib-colab@practicalml-db49f.iam.gserviceaccount.com && colab --auth=adc upload -s p02data "$D/key.json" /content/.gcs_key.json; rm -rf "$D"
```

## Check everything with a smoke run

`SMOKE=1` runs a 2-block model on two shards through every step, including the bucket and
W&B, in a few minutes:

```bash
colab --auth=adc exec -s p02data -f toylib_projects/p02_tinystories/notebooks/01_d20_climbmix_colab.ipynb --timeout 3600 --env SMOKE=1 --env GOOGLE_APPLICATION_CREDENTIALS=/content/.gcs_key.json --env CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE=/content/.gcs_key.json --env GOOGLE_CLOUD_PROJECT=practicalml-db49f
```

It should end with a validation loss and a `smoke` run on W&B.

## Make the token files once

The same command without `SMOKE=1` downloads 120 shards (about 11 GB), trains the tokenizer,
tokenizes and copies the files to the bucket. On a CPU runtime it then stops with
`Token files are in the bucket`, which is expected. Nobody has timed this on Colab yet. Going
by a laptop, allow about an hour on two cores. If the runtime runs out of memory, add
`--high-mem` to `colab new`.

```bash
colab --auth=adc exec -s p02data -f toylib_projects/p02_tinystories/notebooks/01_d20_climbmix_colab.ipynb --timeout 14400 --env GOOGLE_APPLICATION_CREDENTIALS=/content/.gcs_key.json --env CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE=/content/.gcs_key.json --env GOOGLE_CLOUD_PROJECT=practicalml-db49f
```

```bash
colab --auth=adc stop -s p02data
```

## Train

Start an A100 runtime, and send both keys as above with `-s p02` in place of `-s p02data`:

```bash
colab --auth=adc new -s p02 --gpu A100
```

Then run the notebook:

```bash
colab --auth=adc exec -s p02 -f toylib_projects/p02_tinystories/notebooks/01_d20_climbmix_colab.ipynb --timeout 86400 --env GOOGLE_APPLICATION_CREDENTIALS=/content/.gcs_key.json --env CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE=/content/.gcs_key.json --env GOOGLE_CLOUD_PROJECT=practicalml-db49f
```

The timing cell prints seconds per step, the total hours and peak memory before training
starts. If that's more than you want to spend, stop here and lower `MAX_STEPS` in the
notebook, then commit and push. Changing the settings of a run calls for a new `RUN_ID`,
because reusing one resumes that run.

If it runs out of memory, halve `BATCH_PER_DEVICE` and double `NUM_MICROBATCHES`.

## Watch it and finish

```bash
gcloud storage ls gs://tinystories-checkpoints/toylib-p02-climbmix/d20_v32768/
```

Stop the runtime as soon as training ends, then delete the service-account keys as in post
1's guide:

```bash
colab --auth=adc stop -s p02
```

## If the runtime disappears

Repeat the keys and **Train**. It copies the token files back, and should print
`resuming at step ...`. You lose at most 2,500 steps. A checkpoint is several GB, so the
notebook keeps the newest three.
