import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import bask
from skopt.space import Real, Integer
from skopt.utils import create_result, normalize_dimensions
from tune.plots import plot_objective
import matplotlib.pyplot as plt
import warnings
import time
import math
import tensorflow_addons as tfa

# Disable gpu as its vastly slower than cpu for this small network.
tf.config.set_visible_devices([], 'GPU')


def plot_chart(opt, parameters):
    iteration = len(opt.Xi)
    plt.style.use('dark_background')
    space = normalize_dimensions(opt.space.dimensions)
    fig, ax = plt.subplots(nrows=space.n_dims,
                           ncols=space.n_dims,
                           figsize=(3 * space.n_dims, 3 * space.n_dims))
    fig.patch.set_facecolor('#36393f')
    for i in range(space.n_dims):
        for j in range(space.n_dims):
            ax[i, j].set_facecolor('#36393f')
    warnings.filterwarnings(action='ignore')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    space = normalize_dimensions(opt.space.dimensions)
    result_object = create_result(Xi=opt.Xi,
                                  yi=opt.yi,
                                  space=space,
                                  models=[opt.gp])
    plot_objective(result_object,
                   levels=20,
                   size=3,
                   n_points=200,
                   n_samples=30,
                   dimensions=parameters,
                   alpha=0.25,
                   fig=fig,
                   ax=ax)
    try:
        plt.savefig(f"1s-{timestr}-{iteration}.png",
                    pad_inches=0.1,
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="#36393f")
    except:
        pass
    plt.close()


def train_model(args):
    layer_size1 = args[0]
    layer_size2 = args[1]
    l2reg = tf.keras.regularizers.L1L2(l1=float(args[2]), l2=float(args[3]))
    layer1_dropout = args[4]
    layer2_dropout = args[5]
    base_lr = args[6]

    FIELDS = 83
    RECORD_SIZE = (FIELDS + 2) * 4

    x_train = tf.data.FixedLengthRecordDataset(["Data/td.dat"],
                                               RECORD_SIZE,
                                               compression_type="GZIP")
    x_test = tf.data.FixedLengthRecordDataset(["Test/td.dat"],
                                              RECORD_SIZE,
                                              compression_type="GZIP")

    per_type_count = 3

    def extractor(raw):
        inputs = tf.io.decode_raw(tf.strings.substr(raw, 0, RECORD_SIZE - 8),
                                  tf.float32)
        inputs = tf.concat(
            [
                inputs[:, 0:0 + per_type_count],
                inputs[:, 40:40 + per_type_count],
                inputs[:, 60:60 + per_type_count],
                inputs[:, 80:81],
                ##inputs[:, 81:82],
                inputs[:, 82:83] / 50.0,
            ],
            1)
        ##target = tf.io.decode_raw(tf.strings.substr(raw, RECORD_SIZE-4, 4), tf.float32)
        ##target = tf.math.maximum(target, -tf.ones_like(target))
        target = tf.io.decode_raw(tf.strings.substr(raw, RECORD_SIZE - 8, 4),
                                  tf.float32)
        return (inputs, target)

    x_train = x_train.batch(64, drop_remainder=True).map(extractor).prefetch(4)
    x_test = x_test.batch(1024, drop_remainder=True).map(extractor).prefetch(4)

    ##for x in x_train:
    ##    (a, b)=x
    ##    tf.print(tf.math.reduce_std(a, 0), summarize=-1)
    ##    tf.print(tf.math.reduce_mean(a, 0), summarize=-1)
    ##tf.print("_______________________________________________________________________________")
    ##for x in x_test:
    ##    (a, b)=x
    ##    tf.print(tf.math.reduce_std(a, 0), summarize=-1)
    ##    tf.print(tf.math.reduce_mean(a, 0), summarize=-1)

    model = keras.Sequential([
        layers.InputLayer(input_shape=(FIELDS - 81 + 3 * per_type_count)),
        ##layers.BatchNormalization(scale=False),
        layers.Dense(layer_size1,
                     activation="swish",
                     name="layer1",
                     kernel_regularizer=l2reg),
        layers.Dropout(rate=layer1_dropout),
        layers.Dense(layer_size2,
                     activation="swish",
                     name="layer2",
                     kernel_regularizer=l2reg),
        layers.Dropout(rate=layer2_dropout),
        layers.Dense(1,
                     activation="sigmoid",
                     name="layer3",
                     kernel_regularizer=l2reg),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_lr,
                                           amsgrad=True),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.FalsePositives(thresholds=[0.95]),
            keras.metrics.FalseNegatives(thresholds=[0.95]),
            keras.metrics.Precision(thresholds=[0.95]),
            keras.metrics.Recall(thresholds=[0.95]),
            keras.metrics.RecallAtPrecision(precision=0.994),
        ],
    )

    reduceOnPlateau = tf.keras.callbacks.ReduceLROnPlateau()
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     patience=15,
                                                     restore_best_weights=True)
    history = model.fit(
        x_train,
        epochs=100,
        validation_data=x_test,
        callbacks=[reduceOnPlateau, earlystopping],
    )
    return model, history


def opt_train_model(args):
    full_args = [32, 8] + args
    _, history = train_model(full_args)
    return math.log(history.history['val_loss'][-1])


##parameters=["l1reg","l2reg","layer1_drop","layer2_drop", "lr"]
##opt = bask.optimizer.Optimizer([
##    Real(0.00000001,0.1, 'log-uniform'),
##    Real(0.00000001,0.1, 'log-uniform'),
##    (0.0, 0.6),
##    (0.0, 0.6),
##    Real(0.000001,0.01, 'log-uniform'),
##    ], n_initial_points=30, n_points=1000, acq_func="mes",gp_kwargs = dict(
##        normalize_y=True,
##        warp_inputs=True,
##    ))
##
##Xs = []
##ys =[]
##if len(Xs) > 0:
##    opt.tell(Xs, ys, progress=True, gp_burnin=100)
##for i in range(300):
##    print(opt.run(opt_train_model))
##    if opt.gp.chain_ is not None:
##        print(opt.optimum_intervals())
##        if i % 10 == 0:
##            plot_chart(opt, parameters)

#model, _ = train_model([94, 75, 10.0**-7, 10.0**-8, 0.54, 0.36, 19.0**-4])
model, _ = train_model([32, 8, 10.0**-7, 10.0**-8, 0.25, 0.15, 6 * 10.0**-3])
tf.print(model.weights, summarize=-1)

model.save("trained_model")
model.save_weights("ckpt")
