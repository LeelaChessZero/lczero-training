# WARNING: THIS BRANCH REQUIRES TENSORFLOW 2.13+ AND HAS ONLY BEEN TESTED WITH 2.14

# What's new here?

I've added a few features and modules to the training pipeline. Please direct any questions to the Leela Discord server.


# BT3 Improvements
Building of the progress we made with ffn sizes, optimizers, and smolgen, BT3 brings several new improvements. The first is an improved embedding which adds a dense layer from a full representation of the board state (64 * 12 values) to a list of 64 embedding vectors, followed by an ffn. Arcturai's positional encoding has been removed since it seems to slightly degrade the performance of smolgen-equipped transformers. Previously, most of the attention heads in the first attention layer did not encode any useful information. Now they have some information to work on. The policy accuracy gain is a minor 0.3%, though it comes at negligible latency increase.

Support for layer normalization without centering and adding biases has been added as well. This results in negligible accuracy drop and should increase throughput by a couple percent, especially for smaller models.

There are also a significant number of auxiliary heads inspired by work on the Katago Go engine. One of the new ones is a "policy-value" head, which for each move predicts the value of that move weighted by the proportion of playouts of that node. It can be thought of as predicting the value given that the move was found to be the best, and may allow us to find checkmates and decisive lines faster.


# Quality of life
There are three quality of life improvements: a progress bar, new metrics, and pure attention code

Progress bar: A simple progress bar implemented in the Python `rich` module displays the current steps (including part-steps if the batches are split) and the expected time to completion.

Pure attention: The pipeline no longer contains any code from the original ResNet architecture. This makes for clearer yamls and code. The protobuf has been updated to support smolgen, input gating, and the square relu activation function.

## More metrics

I've added train value accuracy and train policy accuracy for the sake of completeness and to help detect overfitting. The speed difference is negligible. There are also three new losses metrics to evaluate policy. The cross entropy we are currently using is probably still the best for training, though we could try instead to turn the task into a classification problem, effectively using one-hot vectors at the targets' best moves, though this would run the risk of overfitting.

Thresholded policy accuracies: the thresholded policy accuracy @x% is the percent of moves for which the net has policy at least x% at the move the target thinks is best.

Reducible policy loss is the amount of policy loss we can reduce, i.e., the policy loss minus the entropy of the policy target.

The search policy loss is designed to loosely describe how long it would take to find the best move in the average position. It is implemented as the average of the multiplicative inverses of the network's policies at the targets' top moves, or one over the harmonic mean of those values. This is not too accurate since the search algorithm will often give up on moves the network does not like unless they provide returns that the network can immediately recognize.


# Architectural improvements
There are a few architectural improvements I've introduced. I only list the apparently useful ones here. For reference, doubling the model size (i.e., 40% larger embeddings or 100% more layers) seems to add 1.5% policy accuracy at 8h/384/512dff.

Note that the large expansion ratios of 4x in the models I report here are not as useful in larger models. A 1536dff outperforms a 4096dff at 10x 8h/1024.

The main improvement is smolgen, which adds 2% policy accuracy to a 10x 8h/384/512dff model.

I've also allowed for the model to train with sparsity so that we can increase throughput on the Ada and Hopper generations of Nvidia GPUs.


## Smolgen

Smolgen is the best improvement by far. It adds around 2% policy accuracy to a 10x 8h/384/512dff model. The motivation is simple: how can we encode global information into self-attention? The encoder architecture has two stages: self-attention and a dense feedforward layer. Self-attention only picks out information shaired between pairs of squares, while the feedforward layer looks at only one square. The only way global information enters the picture is through the softmax, but this cannot be expected to squeeze any significant information out. 

Of course repeated application of self-attention is sufficient with large enough embedding sizes and layers, but chess is fundamentally different from image recognition and NLP. The encoder architecture effectively partitions inputs into nodes and allows them at each layer to spread information and then do some postprocessing with the results. This works in image recognition since it makes sense to compare image patches and the image can be represented well by patch embeddings at each patch. In NLP, the lexical tokens are very suited for this spread of information since the simple structures of grammar allows self-attention (with distance embeddings of course so that tokens can interact locally at first).

Compared to these problems, chess is a nightmare. What it means that there are two rooks on the same file depends greatly on whether there are pieces between them. Even while the transformer architecture provides large gains against ResNets, which are stuck processing local information, it is still not suited for a problem which requires processing not at the between-square level bet on the global level. 

The first solution was logit gating. Arcturai observed that adding embeddings which represent squares which can be reached through knight, bishop, or rook moves vastly improved the architecture. My first attempt at improving upon this was logit gating. Because the board structure is fixed, it makes sense to add an additive offset to the attention logits so that heads can better focus on what is important. A head focusing on diagonals could have its gating emphasize square-pairs which lie on a same diagonal. I achieved further improvements applying multiplicative factors to the attention weights.

This solution works well, but still has its shortcomings. In particular, it is static. We'd like our offsets to change with the input. If pawns are blocking out a diagonal, we would like to greatly reduce the information transfer between pieces on that diagonal. This leads to fullgen, which dynamically generates additional attention logits from the board state. Because it should focus on spatial information and a 13-hot vector is sufficient to completely describe the board state, the representation is generated by applying a dense layer to compress each square into a representation of size 32 (of course, the embeddings already contain processed information which will be useful in the computation).

This is then flattened and put through two dense layers with hidden sizes 256 and hx256 and swish nonlinearities (barely better than relu). Finally, a dense layer (256 -> 4096) is applied to hx64x64, where h is the number of heads. This is added to the attention logits which are computed regularly. This last dense layer is extremely parameter intensive, so it is shared across all layers. This works well in practice.

Smolgen adds about a percent policy accuracy for +10% latency, which is well worth the cost. Increasing the number of heads so that each has size 16 adds ~0.5% pol acc to a 10x 8h/384/512dff since smolgen can do a lot of heavy lifting, but it is not clear whether this is worth the latency cost.


# Training

The training pipeline resides in `tf`, this requires tensorflow running on linux (Ubuntu 16.04 in this case). (It can be made to work on windows too, but it takes more effort.)

## Installation

Install the requirements under `tf/requirements.txt`. And call `./init.sh` to compile the protobuf files.

## Data preparation

In order to start a training session you first need to download training data from https://storage.lczero.org/files/training_data/. Several chunks/games are packed into a tar file, and each tar file contains an hour worth of chunks. Preparing data requires the following steps:

```
wget https://storage.lczero.org/files/training_data/training-run1--20200711-2017.tar
tar -xzf training-run1--20200711-2017.tar
```

## Training pipeline

Now that the data is in the right format one can configure a training pipeline. This configuration is achieved through a yaml file, see `training/tf/configs/example.yaml`:

The configuration is pretty self explanatory, if you're new to training I suggest looking at the [machine learning glossary](https://developers.google.com/machine-learning/glossary/) by google. Now you can invoke training with the following command:

```bash
./train.py --cfg configs/example.yaml --output /tmp/mymodel.txt
```

This will initialize the pipeline and start training a new neural network. You can view progress by invoking tensorboard:

```bash
tensorboard --logdir leelalogs
```

If you now point your browser at localhost:6006 you'll see the trainingprogress as the trainingsteps pass by. Have fun!

## Restoring models

The training pipeline will automatically restore from a previous model if it exists in your `training:path` as configured by your yaml config. For initializing from a raw `weights.txt` file you can use `training/tf/net_to_model.py`, this will create a checkpoint for you.

## Supervised training

Generating trainingdata from pgn files is currently broken and has low priority, feel free to create a PR.
