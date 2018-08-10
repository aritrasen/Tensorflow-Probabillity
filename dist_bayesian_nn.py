"""Trains a deep Bayesian neural net to classify MNIST digits."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports
from absl import flags
import matplotlib
matplotlib.use("Agg")
from matplotlib import figure  # pylint: disable=g-import-not-at-top
from matplotlib.backends import backend_agg
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import horovod.tensorflow as hvd

from tensorflow.contrib.learn.python.learn.datasets import mnist

# TODO(b/78137893): Integration tests currently fail with seaborn imports.
try:
  import seaborn as sns  # pylint: disable=g-import-not-at-top
  HAS_SEABORN = True
except ImportError:
  HAS_SEABORN = False

tfd = tf.contrib.distributions

IMAGE_SHAPE = [28, 28]

flags.DEFINE_float("learning_rate",
                   default=0.01,
                   help="Initial learning rate.")
flags.DEFINE_integer("max_steps",
                     default=6000,
                     help="Number of training steps to run.")
flags.DEFINE_list("layer_sizes",
                  default=["128", "128"],
                  help="Comma-separated list denoting hidden units per layer.")
flags.DEFINE_string("activation",
                    default="relu",
                    help="Activation function for all hidden layers.")
flags.DEFINE_integer("batch_size",
                     default=128,
                     help="Batch size.")
flags.DEFINE_string("data_dir",
                    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                                         "bayesian_neural_network/data"),
                    help="Directory where data is stored (if using real data).")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                         "bayesian_neural_network/"),
    help="Directory to put the model's fit.")
flags.DEFINE_integer("viz_steps",
                     default=400,
                     help="Frequency at which save visualizations.")
flags.DEFINE_integer("num_monte_carlo",
                     default=50,
                     help="Network draws to compute predictive probabilities.")
flags.DEFINE_bool("fake_data",
                  default=None,
                  help="If true, uses fake data. Defaults to real data.")

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

def plot_weight_posteriors(names, qm_vals, qs_vals, fname):
  """Save a PNG plot with histograms of weight means and stddevs.
  Args:
    names: A Python `iterable` of `str` variable names.
    qm_vals: A Python `iterable`, the same length as `names`,
      whose elements are Numpy `array`s, of any shape, containing
      posterior means of weight varibles.
    qs_vals: A Python `iterable`, the same length as `names`,
      whose elements are Numpy `array`s, of any shape, containing
      posterior standard deviations of weight varibles.
    fname: Python `str` filename to save the plot to.
  """
  fig = figure.Figure(figsize=(6, 3))
  canvas = backend_agg.FigureCanvasAgg(fig)

  ax = fig.add_subplot(1, 2, 1)
  for n, qm in zip(names, qm_vals):
    sns.distplot(qm.flatten(), ax=ax, label=n)
  ax.set_title("weight means")
  ax.set_xlim([-1.5, 1.5])
  ax.set_ylim([0, 4.])
  ax.legend()

  ax = fig.add_subplot(1, 2, 2)
  for n, qs in zip(names, qs_vals):
    sns.distplot(qs.flatten(), ax=ax)
  ax.set_title("weight stddevs")
  ax.set_xlim([0, 1.])
  ax.set_ylim([0, 25.])

  fig.tight_layout()
  canvas.print_figure(fname, format="png")
  print("saved {}".format(fname))


def plot_heldout_prediction(input_vals, probs,
                            fname, n=10, title=""):
  """Save a PNG plot visualizing posterior uncertainty on heldout data.
  Args:
    input_vals: A `float`-like Numpy `array` of shape
      `[num_heldout] + IMAGE_SHAPE`, containing heldout input images.
    probs: A `float`-like Numpy array of shape `[num_monte_carlo,
      num_heldout, num_classes]` containing Monte Carlo samples of
      class probabilities for each heldout sample.
    fname: Python `str` filename to save the plot to.
    n: Python `int` number of datapoints to vizualize.
    title: Python `str` title for the plot.
  """
  fig = figure.Figure(figsize=(9, 3*n))
  canvas = backend_agg.FigureCanvasAgg(fig)
  for i in range(n):
    ax = fig.add_subplot(n, 3, 3*i + 1)
    ax.imshow(input_vals[i, :].reshape(IMAGE_SHAPE), interpolation="None")

    ax = fig.add_subplot(n, 3, 3*i + 2)
    for prob_sample in probs:
      sns.barplot(np.arange(10), prob_sample[i, :], alpha=0.1, ax=ax)
      ax.set_ylim([0, 1])
    ax.set_title("posterior samples")

    ax = fig.add_subplot(n, 3, 3*i + 3)
    sns.barplot(np.arange(10), np.mean(probs[:, i, :], axis=0), ax=ax)
    ax.set_ylim([0, 1])
    ax.set_title("predictive probs")
  fig.suptitle(title)
  fig.tight_layout()

  canvas.print_figure(fname, format="png")
  print("saved {}".format(fname))


def build_input_pipeline(mnist_data, batch_size, heldout_size):
  """Build an Iterator switching between train and heldout data."""
  # Build an iterator over training batches.
  training_dataset = tf.data.Dataset.from_tensor_slices(
      (mnist_data.train.images, np.int32(mnist_data.train.labels)))
  training_batches = training_dataset.repeat().batch(batch_size)
  training_iterator = training_batches.make_one_shot_iterator()

  # Build a iterator over the heldout set with batch_size=heldout_size,
  # i.e., return the entire heldout set as a constant.
  heldout_dataset = tf.data.Dataset.from_tensor_slices(
      (mnist_data.validation.images,
       np.int32(mnist_data.validation.labels)))
  heldout_frozen = (heldout_dataset.take(heldout_size).
                    repeat().batch(heldout_size))
  heldout_iterator = heldout_frozen.make_one_shot_iterator()

  # Combine these into a feedable iterator that can switch between training
  # and validation inputs.
  handle = tf.placeholder(tf.string, shape=[])
  feedable_iterator = tf.data.Iterator.from_string_handle(
      handle, training_batches.output_types, training_batches.output_shapes)
  images, labels = feedable_iterator.get_next()

  return images, labels, handle, training_iterator, heldout_iterator


def build_fake_data(num_examples=10):
  """Build fake MNIST-style data for unit testing."""

  class Dummy(object):
    pass

  num_examples = 10
  mnist_data = Dummy()
  mnist_data.train = Dummy()
  mnist_data.train.images = np.float32(np.random.randn(
      num_examples, np.prod(IMAGE_SHAPE)))
  mnist_data.train.labels = np.int32(np.random.permutation(
      np.arange(num_examples)))
  mnist_data.train.num_examples = num_examples
  mnist_data.validation = Dummy()
  mnist_data.validation.images = np.float32(np.random.randn(
      num_examples, np.prod(IMAGE_SHAPE)))
  mnist_data.validation.labels = np.int32(np.random.permutation(
      np.arange(num_examples)))
  mnist_data.validation.num_examples = num_examples
  return mnist_data


class DSHandleHook(tf.train.SessionRunHook):
    def __init__(self, train_str, valid_str):
        self.train_str = train_str
        self.valid_str = valid_str
        self.train_handle = None
        self.valid_handle = None

    def after_create_session(self, session, coord):
        del coord
        if self.train_str is not None:
            self.train_handle, self.valid_handle = session.run([self.train_str,
                                                                self.valid_str])
print('session run ds string-handle done....')

def main(argv):
  del argv  # unused
  hvd.init()
  FLAGS.layer_sizes = [int(units) for units in FLAGS.layer_sizes]
  FLAGS.activation = getattr(tf.nn, FLAGS.activation)
  if tf.gfile.Exists(FLAGS.model_dir+ str(hvd.rank())):
    tf.logging.warning(
        "Warning: deleting old log directory at {}".format(FLAGS.model_dir))
    tf.gfile.DeleteRecursively(FLAGS.model_dir)
  tf.gfile.MakeDirs(FLAGS.model_dir)

  if FLAGS.fake_data:
    mnist_data = build_fake_data()
  else:
    mnist_data = mnist.read_data_sets(FLAGS.data_dir+str(hvd.rank()))
  
  with tf.Graph().as_default():
    (images, labels, handle,
     training_iterator, heldout_iterator) = build_input_pipeline(
         mnist_data, FLAGS.batch_size, mnist_data.validation.num_examples)

    # Build a Bayesian neural net. We use the Flipout Monte Carlo estimator for
    # each layer: this enables lower variance stochastic gradients than naive
    # reparameterization.
    with tf.name_scope("bayesian_neural_net", values=[images]):
      neural_net = tf.keras.Sequential()
      for units in FLAGS.layer_sizes:
        layer = tfp.layers.DenseFlipout(
            units,
            activation=FLAGS.activation)
        neural_net.add(layer)
      neural_net.add(tfp.layers.DenseFlipout(10))
      logits = neural_net(images)
      labels_distribution = tfd.Categorical(logits=logits)

    # Compute the -ELBO as the loss, averaged over the batch size.
    neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(labels))
    kl = sum(neural_net.losses) / mnist_data.train.num_examples
    elbo_loss = neg_log_likelihood + kl

    # Build metrics for evaluation. Predictions are formed from a single forward
    # pass of the probabilistic layers. They are cheap but noisy predictions.
    predictions = tf.argmax(logits, axis=1)
    accuracy, accuracy_update_op = tf.metrics.accuracy(
        labels=labels, predictions=predictions)

    # Extract weight posterior statistics for later visualization.
    names = []
    qmeans = []
    qstds = []
    for i, layer in enumerate(neural_net.layers):
      q = layer.kernel_posterior
      names.append("Layer {}".format(i))
      qmeans.append(q.mean())
      qstds.append(q.stddev())

    with tf.name_scope("train"):
      opt = tf.train.AdamOptimizer(learning_rate=(FLAGS.learning_rate*hvd.size()))
      opt = hvd.DistributedOptimizer(opt)
      global_step = tf.train.get_or_create_global_step()
      train_op = opt.minimize(elbo_loss, global_step=global_step)
      

      # Run the training loop.
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      config.gpu_options.visible_device_list = str(hvd.local_rank())

      checkpoint_dir = './checkpoints16' if hvd.rank() == 0 else None


      train_str_handle = training_iterator.string_handle()
      heldout_str_handle = heldout_iterator.string_handle()

      ds_handle_hook = DSHandleHook(train_str_handle, heldout_str_handle)
      hooks = [
        # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
        # from rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights
        # or restored from a checkpoint.
        hvd.BroadcastGlobalVariablesHook(0),

        # Horovod: adjust number of steps based on number of GPUs.
        tf.train.StopAtStepHook(last_step=FLAGS.max_steps // hvd.size()),


        tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': elbo_loss},
        every_n_iter=100),
        tf.train.LoggingTensorHook(tensors={'step': global_step, 'accuracy': accuracy},
        every_n_iter=100),

        ds_handle_hook
      ]

      with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           hooks=hooks,
                                           config=config) as mon_sess:
        
        while not mon_sess.should_stop():
          
          _ = mon_sess.run([train_op, accuracy_update_op],
                     feed_dict={handle: ds_handle_hook.train_handle})
          step = mon_sess.run(global_step)
          if (step+1) % FLAGS.viz_steps == 0:
          # Compute log prob of heldout set by averaging draws from the model:
          # p(heldout | train) = int_model p(heldout|model) p(model|train)
          #                   ~= 1/n * sum_{i=1}^n p(heldout | model_i)
          # where model_i is a draw from the posterior p(model|train).
             probs = np.asarray([mon_sess.run((labels_distribution.probs),
                                       feed_dict={handle: ds_handle_hook.valid_handle})
                              for _ in range(FLAGS.num_monte_carlo)])
             mean_probs = np.mean(probs, axis=0)

             image_vals, label_vals = mon_sess.run((images, labels),
                                            feed_dict={handle: ds_handle_hook.valid_handle})
             heldout_lp = np.mean(np.log(mean_probs[np.arange(mean_probs.shape[0]),
                                                 label_vals.flatten()]))
             print(" ... Held-out nats: {:.3f}".format(heldout_lp))

             qm_vals, qs_vals = mon_sess.run((qmeans, qstds))
          
        

          

if __name__ == "__main__":
  tf.app.run()