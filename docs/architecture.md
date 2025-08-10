# Architecture Overview

The document outlines the architecture of the new Leela Chess Zero training
system. The training process involves a Reinforcement Learning (RL) pipeline
where new training data is continuously generated.

The old script required fresh starts to train a new network on fresher data.
Furthermore, the use of TensorFlow involved a very long model compilation time
(approx. 1 hour), which dominated the actual training time for a single epoch.

The new script will be a single, long-running Python application that automates
the entire cycle:

1. Monitors for a sufficient amount of new training data.
2. Triggers and executes the training of a new network.
3. Exports the trained network for use.

The core training loop will be implemented in JAX. The data
[loading and preprocessing pipeline](loader.md) is a C++ code exposed to Python
via pybind11. This C++ library is internally multi-threaded but exposes a simple
API to Python (GetNextBatch(), GetStats()).

We'd like to have a fancy TUI dashboard to monitor the loading and training
process.

## TrainingDaemon

The main (in terms of importance) class of the training code is
`TrainingDaemon`, which:

* Is a jsonl server operating through stdin/stdout, implementing the protocol
  described in [JSONL IPC Protocol](jsonl.md).
  * Receives a command to start training with the location of the config file.
    * Other commands are possible in the future.
  * Sends periodic progress notifications.
* Owns the data loader.
* Waits for the data loader to ingest enough new chunks.
* Starts the training loop when enough data is available.
* Finalizes and uploads the trained network.

## Configuration

The configuration is a large nested dataclass structure, which covers:

* [Data loader](../src/lczero_training/config/data_loader_config.py) to be
  passed to the C++ data loader.
* Information for the training daemon, e.g. how many chunks to wait for
  before starting the training.
* Model definition, for model builder.
* Training parameters, such as batch size, number of epochs, etc.
* Export parameters, such as the path to export the trained model to.

From user perspective, the configuration is a YAML file, which is parsed
into the dataclass structure.

## Data Loader

The internals of the data loader are described in detail in [Data Loader](loader.md).

From python perspective, it has the following interface:

* Constructor takes a `DataLoaderConfig` dataclass.
* `GetNextBatch()` returns a tuple of buffer-protocol-compliant tensors.
  * Later it will have a parameter that specifies wether we need training, test
    or validation batch.
* `GetStats()` returns a DataClass (exact structure TBD) with the current
  statistics of the data loader.

## TUI

TUI is a separate frontend app, implemented as `TrainingTuiApp`, which runs
`TrainingDaemon` as a subprocess (and communicates through jsonl via
stdin/stdout).

* TUI is `Textual`-based app.
* Located in `src/lczero_training/tui/`.
* UI ideas are described in [TUI](tui.md).
* TUI will have a log pane which shows stderr of `TrainingDaemon`.
