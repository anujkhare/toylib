# Running the notebook on Colab

`01_imagenette_vit_colab.ipynb` trains runs from `train.RUNS` on a Colab machine. Google's
[Colab CLI](https://github.com/googlecolab/google-colab-cli) rents the machine and runs the
notebook on it from a shell, so these commands work from your terminal or a Claude Code
session.

| Run | Model | Bucket folder | W&B project |
| --- | --- | --- | --- |
| `vit_ti16`, the default | ViT-Ti/16's shape, 200 epochs at 160x160 | `toylib-p03-vit/vit_ti16_e200` | `toylib-p03-vit` |

Three things to know first:

- Colab gets the code from GitHub, not your working copy, so **push before you run**. The
  notebook's `BRANCH`, `p03-vit`, says which branch it clones.
- Files on the runtime disappear with it, so checkpoints go straight to
  `gs://tinystories-checkpoints/` and metrics go to W&B.
- A runtime is billed while it exists, even when idle. A `v5e1` costs about 3.1 compute
  units an hour. Stop it when you're done.

Before spending anything, check it locally. The tests run every cell of the notebook with
a stand-in model, and the `smoke` run trains the real ViT on 64 images through the same
code:

```bash
uv run pytest toylib_projects/p03_vit
```

```bash
uv run python -m toylib_projects.p03_vit.train --run smoke
```

Commands below run from the root of the repo.

## One-time setup

These are the same as post 1's, so skip any you've already done. Install the CLI, then
open a new terminal so `colab` is on your PATH:

```bash
uv tool install google-colab-cli
```

```bash
uv tool update-shell
```

Log in to Google with the scopes Colab needs. It worked if it ends with
`Credentials saved to file`:

```bash
gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

Log in to W&B with your key from <https://wandb.ai/authorize>:

```bash
uv run wandb login --relogin
```

Check both, and see your compute units. Always pass `--auth=adc`: without it, the CLI
uses a different login.

```bash
colab --auth=adc usage
```

The runtime reaches the bucket through the `toylib-colab` service account, which post 1's
`notebooks/COLAB.md` explains how to create. It needs `roles/storage.objectAdmin` and
`roles/storage.legacyBucketReader` on the bucket.

## Start a runtime

Each new runtime needs all three steps, one after another: stopping the runtime loses the
uploaded keys.

Request the TPU. If a request sits silent for a couple of minutes, press Ctrl-C and try
another type, or a GPU with `--gpu A100`. A `400` means your account has no quota for it.

```bash
colab --auth=adc new -s p03 --tpu v5e1
```

Send the W&B key as a file, not with `colab exec --env`, which saves values in plain text
in the CLI's local history:

```bash
K=$(mktemp) && uv run python -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2], end='')" > "$K" && colab --auth=adc upload -s p03 "$K" /content/.wandb_key; rm -f "$K"
```

Create a service-account key and send that too. Check that it prints `Uploaded`:

```bash
D=$(mktemp -d) && gcloud iam service-accounts keys create "$D/key.json" --iam-account=toylib-colab@practicalml-db49f.iam.gserviceaccount.com && colab --auth=adc upload -s p03 "$D/key.json" /content/.gcs_key.json; rm -rf "$D"
```

Each run adds a key, and an account can hold only 10, so delete it when you finish.

## Run

`--timeout` is per cell and defaults to 30 seconds, which would abandon the training cell,
so this allows 24 hours. The `--env` values are file paths and a project name, not
secrets. Add `--env RUNS_TO_TRAIN=<name>` to train a run other than `vit_ti16`.

```bash
colab --auth=adc exec -s p03 -f toylib_projects/p03_vit/notebooks/01_imagenette_vit_colab.ipynb --timeout 86400 --env GOOGLE_APPLICATION_CREDENTIALS=/content/.gcs_key.json --env CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE=/content/.gcs_key.json --env GOOGLE_CLOUD_PROJECT=practicalml-db49f
```

The notebook installs the repo, checks the bucket and the accelerator, prints the run's
settings, times it, and then trains it. It stops early if the runtime can't read the
bucket, has no accelerator, or has no W&B key. Live output appears only in the terminal
running this command, and `exec` saves the outputs to a gitignored `_output.ipynb` next to
the notebook when it ends.

## Watch it

```bash
colab --auth=adc status -s p03
```

```bash
gcloud storage ls gs://tinystories-checkpoints/toylib-p03-vit/vit_ti16_e200/
```

A checkpoint at step 0 and a W&B run mean the bucket and the key both work.

## Finish

Stop the runtime as soon as training ends, then check that nothing is left running:

```bash
colab --auth=adc stop -s p03
```

```bash
colab --auth=adc usage
```

List the service account's keys, and delete each one by its ID:

```bash
gcloud iam service-accounts keys list --iam-account=toylib-colab@practicalml-db49f.iam.gserviceaccount.com --managed-by=user
```

```bash
gcloud iam service-accounts keys delete KEY_ID --iam-account=toylib-colab@practicalml-db49f.iam.gserviceaccount.com --quiet
```

## If the runtime disappears

Repeat **Start a runtime** and **Run**. The run restores its newest checkpoint from the
bucket, so you lose at most `checkpoint_every_epochs`, 10 epochs by default. It should
print `resuming at step ...`. If it prints `starting fresh` instead, stop and check the
bucket for the run's folder.

## Change what runs

Add or edit a run in `train.RUNS`. Its name and epochs name its bucket folder and W&B
run, so reusing a name resumes that run: give a changed experiment a new name. Commit and
push before running.

## Getting results back

Code on your machine can load a checkpoint straight from the bucket. Set
`GOOGLE_CLOUD_PROJECT=practicalml-db49f` in the environment first:

```python
net = inference.load_model(template, "gs://tinystories-checkpoints/toylib-p03-vit/vit_ti16_e200")
```

## Known limits

- The batch must divide evenly across devices. 128 and 400 work with 1 or 8.
