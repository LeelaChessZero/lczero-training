# Running "new" training pipeline

Note that the code is still in active development, so things change a lot.
This document was last updated on 2025-10-28.

## Building

The new training pipeline is located in `src/` (Python part) and `csrc/` (C++
part).

* Python code uses `uv`. Install it as described in the
  [uv installation guide](https://docs.astral.sh/uv/#installation).
* Many steps are run via `just`. `just` is just a glorified shell script runner.
  So either look into [`Justfile`](../justfile) or install `just` as described
  in the [just installation guide](https://github.com/casey/just#installation).
* You'll need a recent protobuf compiler (`protoc`).
* You'll need a C++ compiler. In this example we use `clang`.

```bash
cd <repo-root>
uv python install 3.12
uv venv
uv sync
uv pip install meson ruff
git submodule update --init --recursive
CXX=clang++ CC=clang uv run meson setup build/release\
   --buildtype=release --native-file=native.ini
uv run meson configure build/release \
   -Dcpp_args='-Wno-error=deprecated-declarations'
just build
cd src/lczero_training
ln -sfT ../../build/release/_lczero_training.cpython-*-x86_64-linux-gnu.so _lczero_training.so
just build-proto
```

## Training a model

To train a model you need:

* Training data
* A configuration file
* Create a checkpoint.
* Run the pipeline.

### Training data

Unlike the old training pipeline, the new one doesn't need .tar files to be
unpacked. While it does support plain `.gz` chunk files, it's not efficient as
it stores each individual file name in memory. So instead, use `.tar` files,
the tool can index and seek inside them.

The tool watches a directory (and its subdirectories) for new files.

Terms used:

* **Chunk**/Game: A single training game, individual `.gz` file.
* **Chunk source**: A file (`.tar` or `.gz`) containing multiple chunks.
* **Frame**/Record/Position: A single training position inside a chunk.
* **Training tensor**: A single batch of inputs/outputs encoded in NN format for
  one training step.

Incoming data comes as chunks, but for the training we need frames from
different games.

### Note on RL vs SL training

The tool supports both supervised learning (SL) and reinforcement learning (RL).
Here is overview of the configuration differences:

| RL Training                                                                                                                                                                                                                                                                       | SL Training                                                                                                                                                   |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `shuffling_chunk_pool` has relatively small`chunk_pool_size`, which would be used as a sliding window.                                                                                                                                                                            | `chunk_pool_size` should be larger than all data, so that all data is used for training.                                                                      |
| `training.schedule.chunks_per_network` is non zero — that's how many new chunks to wait for before starting a new epoch.                                                                                                                                                          | `chunks_per_network` is zero. Once an epoch is done, it starts a new one immediately.                                                                         |
| RL currently uses "hanse sampling", where currently the entire chunk is loaded and rescored to use just one position from it. The reservoir `shuffling_frame_sampler` is not used in this case. This is currently slow (until we implement caching), so it limits the throughput. | For SL, it better to use two stage sampling: `shuffling_chunk_pool` in non-hanse mode, and then `shuffling_frame_sampler` to shuffle positions within chunks. |

### Creating a checkpoint

To create a fresh checkpoint, you'll need `model` and `training` sections of the
configuration file to be filled.

Then run:

```bash
uv run training-init --config <your_config>.textproto --lczero_model <model>.pb.gz
```

The `--lczero_model` parameter is optional. If not given, the network is
initialized with random weights.

### Training

Run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run tui --config <your_config>.textproto --logfile train.log
```

Notes:

* There are log files both in `tui` and in the configuration file. They
are slightly different, TUI log usually is more useful.
* In the tool, you can press `q` to quit, and `Ctrl+p` to get a command palette.
There, you have one useful command: "Start training immediately".
* For multi-GPU training, ensure your batch size is divisible by the number of GPUs.
* The overfit utility (`uv run overfit`) does not support multi-GPU. Set `CUDA_VISIBLE_DEVICES` to use only one GPU when running overfit.
* Also note that TUI is 100% vibe coded, so you'll see lots of mocks in the UI.
:-P

## Tools

The repository consists of set of tools, mostly written in Python, but some are
in C++. To run a Python tool, use `uv run <tool>`. C++ tools are binaries in
`build/release`. Most tools need a configuration file as a parameter (see
below).

### Python tools

| Tool                   | Description                                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------------------------------ |
| lc0-daemon             | The main training daemon. It acts as a JSONL server, so it's not usable directly from command line yet.      |
| lc0-tui                | A terminal user interface that runs the training daemon. Here is what you have to run.                       |
| lc0-init               | Initializes a new training run/checkpoint.                                                                   |
| lc0-migrate-checkpoint | Migrates JAX/Orbax checkpoint after model/training configuration changes.                                    |
| lc0-overfit            | Runs an overfitting test: takes one batch from the data loader and repeatedly trains on it                   |
| lc0-eval               | Evals batches from the data loader on a given checkpoint, can dump inputs/outputs in various formats.        |
| lc0-leela2jax          |                                                                                                              |
| lc0-describe           |                                                                                                              |
| lc0-test-dataloader    |                                                                                                              |
| lc0-tune-lr            | Trains on exponentially increasing learning rate, and outputs losses into csv file. Useful for picking a LR. |
| lc0-backfill-metrics   | Loads older checkpoints computes metrics for them, and exports them to tensorboard.                          |
| lc0-train              | Trains a single epoch (doesn't save or export the model though). Used for benchmarking.                      |

### C++ tools

| Tool                         | Description                                               |
| ---------------------------- | --------------------------------------------------------- |
| rescore_chunk                | Runs rescorer on a single chunk                           |
| startpos_policy_distribution |                                                           |
| result_distribution          |                                                           |
| filter_chunks                |                                                           |
| dump_chunk                   | Dumps the content of a chunk file for debugging purposes. |

## Configuration

The configuration is a text protobuf file, with the following sections:

| Section     | Description                                    |
| ----------- | ---------------------------------------------- |
| data_loader | Configuration for the data loader (see below). |
| model       | Model architecture configuration.              |
| training    | Configuration for the training configuration.  |
| metrics     | Metrics to export into tensorboard.            |
| export      | Configuration for exporting the trained model. |

Also it has `log_filename` field (where to write the log) and `name` field (must
be `little-teapot`).

It's recommended to use existing configuration files as a starting point.

### Data loader configuration

Data loader is a pipeline that consists of pluggable stages. Here are stages
that are currently implemented:

| Stage type                | Description                                                                                      | Input        | Output                |
| ------------------------- | ------------------------------------------------------------------------------------------------ | ------------ | --------------------- |
| `file_path_provider`      | Watches a directory for existing and new files.                                                  | None         | Filenames             |
| `chunk_source_reader`     | Reads and indexes chunk source files (`.tar` or `.gz`).                                          | Filenames    | ChunkSources          |
| `chunk_source_splitter`   | Splits chunk sources into smaller chunk sources give the proportion (used for test/train split). | ChunkSources | multiple ChunkSources |
| `shuffling_chunk_pool`    | Accumulates chunk sources and outputs chunks in shuffled order                                   | ChunkSources | Chunks                |
| `simple_chunk_extractor`  | Unpacks chunks from chunk sources                                                                | ChunkSources | Chunks                |
| `chunk_rescorer`          | Rescores chunks                                                                                  | Chunks       | Chunks                |
| `chunk_unpacker`          | Extracts positions from chunks                                                                   | Chunks       | Frames                |
| `shuffling_frame_sampler` | Outputs frames in shuffled order                                                                 | Frames       | Frames                |
| `tensor_generator`        | Converts frames into training batches in numpy tensor format                                     | Frames       | Training Tensors      |