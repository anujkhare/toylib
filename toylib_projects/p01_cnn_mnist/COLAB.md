# Running a notebook on a Colab TPU

Three notebooks train models too slow for a laptop CPU. Google's [Colab CLI](https://github.com/googlecolab/google-colab-cli) rents a Colab
machine and runs the notebook on it from a shell, so these commands work from your terminal
or a Claude Code session.

| Notebook | Model | Bucket folder | W&B project |
| --- | --- | --- | --- |
| `01_imagenette_colab.ipynb` | ResNet-18 shape, width 32, 160px | `toylib-p01-imagenette/resnet18w32_e200` | `toylib-p01-imagenette` |
| `01_imagenette_baselines.ipynb` | Linear, MLP and small CNN. Pick with `--env RUNS_TO_TRAIN=small_cnn` | `toylib-p01-imagenette/<run>_e200` | `toylib-p01-imagenette` |
| `01_resnets_colab.ipynb` | Wide ResNet-20, CIFAR-10 | `toylib-p01-cifar10/resnet20w32_cutout_e100` | `toylib-p01-cifar10` |

The commands below use the Imagenette notebook, so swap the names for the other one.

Three things to know first:

- Colab gets the code from GitHub, not your working copy, so **push before you run**. The
  notebook's `BRANCH` says which branch it clones.
- Files on the runtime disappear with it, so checkpoints go straight to
  `gs://tinystories-checkpoints/` and metrics go to W&B.
- A runtime is billed while it exists, even when idle. A `v5e1` costs about 3.1 compute
  units an hour. Stop it when you're done.

Commands below run from the root of the repo.

## One-time setup

Clone the repo and install it:

```bash
git clone https://github.com/anujkhare/toylib.git
```

```bash
cd toylib && uv sync --extra posts --extra dev
```

Install the CLI, then open a new terminal so `colab` is on your PATH:

```bash
uv tool install google-colab-cli
```

```bash
uv tool update-shell
```

Log in to Google with the scopes Colab needs. This also lets Python on your machine read the
bucket. It worked if it ends with `Credentials saved to file`.

```bash
gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

Log in to W&B with your key from <https://wandb.ai/authorize>:

```bash
uv run wandb login --relogin
```

Check both, and see your compute units. Always pass `--auth=adc`: without it, the CLI
installed here (0.7.5) uses a different login.

```bash
colab --auth=adc sessions
```

```bash
colab --auth=adc usage
```

### Service account for the bucket

The runtime needs its own access to GCS. Don't use `colab auth`, because Google blocks the
sign-in it uses (`doesn't comply with Google's OAuth 2.0 policy`). Instead, make a service
account that can only reach this bucket. It only has to be created once, and it may already
exist:

```bash
gcloud iam service-accounts create toylib-colab --project=practicalml-db49f --display-name="toylib Colab checkpoints"
```

```bash
gcloud storage buckets add-iam-policy-binding gs://tinystories-checkpoints --member=serviceAccount:toylib-colab@practicalml-db49f.iam.gserviceaccount.com --role=roles/storage.objectAdmin
```

The account also needs to read the bucket's metadata. Without this, a run fails with a
`403` and `storage.buckets.get` as soon as checkpoints already exist, which means it can't
resume:

```bash
gcloud storage buckets add-iam-policy-binding gs://tinystories-checkpoints --member=serviceAccount:toylib-colab@practicalml-db49f.iam.gserviceaccount.com --role=roles/storage.legacyBucketReader
```

If the first command says the IAM API is off, enable it and run the command again:

```bash
gcloud services enable iam.googleapis.com --project=practicalml-db49f
```

## Start a runtime

Each new runtime needs all three steps, and they must follow each other: stopping the runtime
loses the uploaded keys.

Request the TPU. `v5e1` was granted in seconds. `v6e1` hung for over 7 minutes with no output
and no session, so if a request sits silent for a couple of minutes, press Ctrl-C and try the
other type. A `400` means your account has no quota for it.

```bash
colab --auth=adc new -s p01 --tpu v5e1
```

Send the W&B key as a file, not with `colab exec --env`, which saves values in plain text in
the CLI's local history:

```bash
K=$(mktemp) && uv run python -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2], end='')" > "$K" && colab --auth=adc upload -s p01 "$K" /content/.wandb_key; rm -f "$K"
```

Create a service-account key and send that too. Check that it prints `Uploaded`:

```bash
D=$(mktemp -d) && gcloud iam service-accounts keys create "$D/key.json" --iam-account=toylib-colab@practicalml-db49f.iam.gserviceaccount.com && colab --auth=adc upload -s p01 "$D/key.json" /content/.gcs_key.json; rm -rf "$D"
```

Each run adds a key, and an account can only hold 10, so remove old ones when you finish.

## Run

`--timeout` is per cell and defaults to 30 seconds, which would abandon the training cell, so
this allows 24 hours. The `--env` values are file paths and a project name, not secrets.

```bash
colab --auth=adc exec -s p01 -f toylib_projects/p01_cnn_mnist/01_imagenette_colab.ipynb --timeout 86400 --env GOOGLE_APPLICATION_CREDENTIALS=/content/.gcs_key.json --env CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE=/content/.gcs_key.json --env GOOGLE_CLOUD_PROJECT=practicalml-db49f
```

The notebook installs the repo, checks the bucket and the accelerator, times the model, and
then trains it. It stops early if the runtime can't read the bucket, has no accelerator, or
has no W&B key. Live output appears only in the terminal running this command. Outputs are
saved to a `_output.ipynb` file next to the notebook when `exec` ends, and that file is gitignored.

If this command loses its connection or times out, the training cell should keep running on
the kernel.

## Watch it

```bash
colab --auth=adc status -s p01
```

```bash
gcloud storage ls gs://tinystories-checkpoints/toylib-p01-imagenette/resnet18w32_e200/
```

Curves are on W&B in the notebook's project. A checkpoint at step 0 and a W&B run
mean the bucket and the key both work.

## Finish

Stop the runtime as soon as training ends, then check that nothing is left running:

```bash
colab --auth=adc stop -s p01
```

```bash
colab --auth=adc usage
```

List the service account's keys, and delete each one with its ID:

```bash
gcloud iam service-accounts keys list --iam-account=toylib-colab@practicalml-db49f.iam.gserviceaccount.com --managed-by=user
```

```bash
gcloud iam service-accounts keys delete KEY_ID --iam-account=toylib-colab@practicalml-db49f.iam.gserviceaccount.com --quiet
```

## If the runtime disappears

Repeat **Start a runtime** and **Run**. The run restores its newest checkpoint from the
bucket, so you lose only the epochs since it, at most 10 on Imagenette. It should print `resuming at step ...`. If it prints
`starting fresh` instead, stop and check the bucket for its folder.

## Change what runs

The settings are in the notebook's imports cell. `RUNS`, `EPOCHS` and `CUTOUT_SIZE` set the
experiment, and each run's name and epochs set its W&B run and bucket folder. Reusing a name
resumes that run, so give a changed experiment a new name. Commit and push before running.

## If TPU isn't available

Use `--gpu T4` with `colab new`. Nothing else changes, because the notebook uses whatever JAX
finds.

## Getting results back

The run has a folder of checkpoints in the bucket. Code on your machine can
load them directly, using the bucket path as the checkpoint directory. Set
`GOOGLE_CLOUD_PROJECT=practicalml-db49f` in the environment first:

```python
net = inference.load_model(template, "gs://tinystories-checkpoints/toylib-p01-imagenette/resnet18w32_e200")
```

## Known limits

- The batch must divide evenly across devices. 128 and 400 work with 1 or 8.
