import argparse
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pickle
import os
import cifar10data
# import bay_dropout
# import standard_dropout
import warnings
import model
import config
import sys


# from copy import deepcopy


def run(args):

    dirmake = "result" + args.trial + "/"
    if not os.path.exists(dirmake):
        os.makedirs(dirmake)

    args.activation = getattr(tf.nn, args.activation)

    print("=" * 20 + " Print out your input " + "=" * 20)
    file = open(dirmake + "Settings.text", "w")
    for arg in vars(args):
        print(arg + ":", getattr(args, arg))
        file.write(arg + ":" + str(getattr(args, arg)) + "\n")
    file.close()
    exit()

    if args.data.lower() in ("cifar", "cifar10"):
        [
            args.X_train, args.Y_train,
            args.X_dev, args.Y_dev,
            args.X_test, args.Y_test, args.class_name
        ] = cifar10data.extract_data()
    else:
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        args.X_train, args.Y_train = mnist.train.images, mnist.train.labels
        args.X_dev, args.Y_dev = mnist.validation.images, mnist.validation.labels
        args.X_test, args.Y_test = mnist.test.images, mnist.test.labels

    nnetmodel = getattr(model, args.model.lower())
    rt_res = nnetmodel(args)

    # whether use dropout will be discussed later in the future.

    if args.model.lower() == "snn":
        with open(dirmake + "plot_snn", "wb") as out:
            pickle.dump([
                rt_res.plot.niter, rt_res.plot.runtime,
                rt_res.plot.loss, rt_res.plot.devAcc], out
            )
    else:
        with open(dirmake + "plot_bnn", "wb") as out:
            pickle.dump(
                [
                    rt_res.plot.niter, rt_res.plot.runtime, rt_res.plot.loss,
                    rt_res.plot.devAccMean, rt_res.plot.devAccUp,
                    rt_res.plot.devAccDown
                ], out
            )

        # Save posterior distributions of weights:
        # with open(dirmake + "post_bnn", "wb") as out:
        #     pickle.dump([
        #         rt_res.posterior.mean,
        #         rt_res.posterior.std,
        #         rt_res.posterior.samples], out
        #     )

    with open(dirmake + args.model + "acc", "wb") as out:
        pickle.dump(rt_res.acc, out)


if __name__ == "__main__":

    args = config.get_base_parser().parse_args()
    # orig_stdout = sys.stdout
    # out2file = open('out.txt', 'w')
    # sys.stdout = out2file
    run(args)
    # sys.stdout = orig_stdout
    # out2file.close()
