# OS Support:

- Ubuntu 16.04/18.04
- Windows (with more effort)

# Training

The training pipeline resides in the `tf` directory.

## Installation

```bash
git clone --recurse-submodules https://github.com/LeelaChessZero/lczero-training.git
cd lczero-training
python3 -m pip install -r tf/requirements.txt
./init.sh
```

## Data preparation

In order to start a training session you need training data. You can find data at http://data.lczero.org/files/training_data/. This data is packed in .tar files each containing .gz files that we call "chunks". Use `tar -xf training-*.tar` to extract the chunks.

## Training pipeline

The pipeline is configured in a .yaml file which can be kept in tf/configs; see `tf/configs/example.yaml` for a commented example. Google's [machine learning glossary](https://developers.google.com/machine-learning/glossary/) may help you with unfamiliar terms in the configuration.

Now you can invoke training with the following command:

```bash
./train.py --cfg configs/config.yaml --output /tmp/mymodel.txt
```

This will initialize the pipeline and start training a new neural network. You can view progress by invoking tensorboard:
```bash
tensorboard --logdir leelalogs
```

If you now point your browser at localhost:6006 you'll see the training progress as the steps pass by. Have fun!

## Restoring models

The training pipeline will automatically restore from a previous model if it exists in your `training:path` as configured by your yaml config. For initializing from a raw `weights.txt` file you can use `training/tf/net_to_model.py`, this will create a checkpoint for you.

## Supervised training

Generating trainingdata from pgn files is currently broken and has low priority, feel free to create a PR.
