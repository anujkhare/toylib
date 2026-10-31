# Running the heavy models on Colab

`01_resnets_colab.ipynb` trains the runs that are too slow for a laptop CPU. Google's
[Colab CLI](https://github.com/googlecolab/google-colab-cli) rents a Colab machine and runs
the notebook on it from a shell, so these commands work from your terminal or a Claude Code
session.

Colab gets the code from GitHub, so **commits must be pushed before they reach the runtime**.
Files on the runtime disappear with it, so checkpoints go straight to
`gs://tinystories-checkpoints/toylib-p01-cifar10/` and metrics go to W&B. A runtime is
billed while it exists, so stop it when you're done.

## One-time setup

Install the CLI and put it on your PATH, then open a new terminal:

```bash
uv tool install google-colab-cli
```

```bash
uv tool update-shell
```

Log in with the scopes Colab needs. The same login lets Python on this Mac read the bucket.
Pass `--auth=adc` to `colab`, because the installed version defaults to another method.

```bash
gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

Refresh the W&B login:

```bash
uv run wandb login --relogin
```

Check that Colab accepts the login, and see your compute units:

```bash
colab --auth=adc sessions
```

```bash
colab --auth=adc usage
```

## Start a runtime

Each new runtime needs all three of these. A `400` from `colab new` means this account has
no quota for that accelerator, so try `--tpu v5e1` or `--gpu T4`.

```bash
colab --auth=adc new -s p01 --tpu v6e1
```

`colab auth` gives the runtime access to GCS. It needs you at a terminal, so an agent can't
run it:

```bash
colab --auth=adc auth -s p01
```

Send the W&B key as a file. `colab exec --env` would save it in plain text in the CLI's
local history.

```bash
K=$(mktemp) && uv run python -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2], end='')" > "$K" && colab --auth=adc upload -s p01 "$K" /content/.wandb_key; rm -f "$K"
```

## Run

`--timeout` is per cell and defaults to 30 seconds, which would abandon the training cell.
This allows 24 hours:

```bash
colab --auth=adc exec -s p01 -f toylib_projects/p01_cnn_mnist/01_resnets_colab.ipynb --timeout 86400
```

The notebook installs the repo, checks the bucket and the accelerator, times each model, and
then trains all three in order. It stops early if the runtime can't read the bucket or has
no accelerator. Outputs are written to `01_resnets_colab_output.ipynb` next to the notebook
when `exec` ends, and that file is gitignored.

If `exec` loses its connection or times out, the training cell keeps running on the kernel.

## Watch it

```bash
colab --auth=adc status -s p01
```

```bash
gcloud storage ls gs://tinystories-checkpoints/toylib-p01-cifar10/
```

Curves are on W&B in the project `toylib-p01-cifar10`.

## Finish

Stop the runtime as soon as training ends:

```bash
colab --auth=adc stop -s p01
```

## If the runtime disappears

Repeat **Start a runtime** and **Run**. Each run restores its newest checkpoint from the
bucket and finished runs are skipped, so you lose at most five epochs of the current model.

A run with a saved checkpoint should print `resuming at step ...`. If it prints
`starting fresh` instead, stop and check the bucket for that run's folder. The first
checkpoint is saved as training starts.

## Getting results back

Each run has a folder of checkpoints in the bucket, one per 5 epochs. To list them:

```bash
gcloud storage ls gs://tinystories-checkpoints/toylib-p01-cifar10/resnet56_e40/
```

Code on this Mac can load them directly, using the bucket path as the checkpoint directory:

```python
net = inference.load_model(template, "gs://tinystories-checkpoints/toylib-p01-cifar10/resnet56_e40")
```

## Known limits

- If `gcloud` can't find a project on the runtime, add
  `--env GOOGLE_CLOUD_PROJECT=<your project>` to `colab exec`. Don't pass secrets this way.
- The batch must divide evenly across devices. 128 and 400 work with 1 or 8.
