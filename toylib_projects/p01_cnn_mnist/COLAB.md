# Running the heavy models on Colab

`01_resnets_colab.ipynb` trains the runs that are too slow on a laptop CPU. You keep the
notebook open in your editor, while its cells run on a Colab machine through Google's
[Colab extension](https://github.com/googlecolab/colab-vscode).

The notebook's file is on your Mac, but its kernel, its disk and everything it writes are
on the Colab machine. Colab can only get the code from GitHub, so **commits must be pushed
before they reach the runtime**, and anything the run saves disappears with the runtime
unless it reaches Weights & Biases.

## Before you start

1. Get your W&B key from <https://wandb.ai/authorize>. The notebook asks for it once per
   runtime.
2. Refresh the login on this Mac as well, since the stored one has stopped working. You'll
   need it to download results here.

   ```bash
   uv run wandb login --relogin
   ```

3. Install the **Colab** extension (publisher: Google) in VS Code. Cursor gets it from
   Open VSX. Install **Jupyter** too if the editor asks.

## Connect

1. Open `toylib_projects/p01_cnn_mnist/01_resnets_colab.ipynb`.
2. Click **Select Kernel** at the top right, then **Colab**, then **New Colab Server**.
   Avoid Auto Connect: it may not give you a TPU.
3. Choose **TPU** and then a machine type. Give the server a name if asked.
4. Sign in with Google when the browser opens, then return to the editor.
5. Select the Python kernel if the editor asks.

## Run

Run the cells from the top, one at a time as far as the timing cell:

1. **Setup** clones the branch and installs the repo. Check that it ends without errors.
2. **W&B sign-in** opens a prompt at the top of the editor. Paste your key there.
3. **Imports** prints the JAX devices. They must say TPU. If they say CPU, the runtime is
   wrong: remove the server (**Colab: Remove Server**) and connect again.
4. **Timing** measures each model on this accelerator and prints the expected hours. It
   takes a few minutes, most of it compiling. This is the first real TPU test, so if
   something is going to break, it should break here.
5. **Train** runs all three models in order. Each prints a W&B link where you can follow it.

While it runs, keep the editor open and the Mac awake. Don't start other work on the runtime.

This keeps the Mac awake until you press Ctrl-C in its terminal:

```bash
caffeinate -i -d
```

Colab can end a session whenever it needs the machine back, more so on the free tier, so
treat the limits as unknown and rely on the checkpoints.

## If the runtime disappears

Connect to a new server and run every cell again, the timing cell excepted. Each run
restores its newest checkpoint from W&B and finished runs are skipped, so you lose at most
five epochs of the current model.

A run with a saved checkpoint should print `resuming at step ...`. If it prints
`starting fresh` instead, stop the cell and check that W&B has the run's
`<run>_e40-checkpoints` artifact. The first checkpoint uploads as training starts.

## If TPU isn't available

Choose **GPU** in the same menu. Nothing else changes: the notebook uses whatever JAX finds.

## Getting results back

When a run finishes, its checkpoints are W&B artifacts. This downloads the last one to the
Mac (step 15,600 is the end of epoch 40):

```bash
uv run wandb artifact get toylib-p01-cifar10/resnet56_e40-checkpoints:latest --root toylib_projects/p01_cnn_mnist/outputs/resnet_resnet56_e40/checkpoints/15600
```

## Known limits

- `userdata.get` (Colab secrets) [fails from the extension](https://github.com/googlecolab/colab-vscode/wiki/Known-Issues-and-Workarounds),
  which is why the notebook prompts for the W&B key.
- The batch must divide evenly across TPU cores. 128 and 400 work with 1 or 8 cores.
