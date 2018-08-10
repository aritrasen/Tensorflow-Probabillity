# standard neural net

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time
import nnet


def snn(args):

    tf.reset_default_graph()

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    N = args.X_train.shape[0]
    dim = list(args.X_train.shape[1:])
    K = args.Y_train.shape[1]  # num of class

    X = tf.placeholder(tf.float32, [None] + dim)
    y = tf.placeholder(tf.float32, [None, K])

    tfd = tf.contrib.distributions

    neural_net = nnet.convnet(activation=args.activation,
                              inshape=args.inshape, numclass=K, isBay=False)

    logits = neural_net(X)
    labels_distribution = tfd.Categorical(logits=logits)
    pred = tf.nn.softmax(logits, name="pred")

    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    )
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(cost)

    # begin training

    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    class Dummy():
        pass

    res_return = Dummy()
    res_return.plot = Dummy()
    res_return.plot.niter = []
    res_return.plot.runtime = []
    res_return.plot.loss = []
    res_return.plot.devAcc = []

    with tf.Session() as sess:
        print("=" * 21 + "Optimization Start" + "=" * 21)
        start_time, algstart = time.time(), time.time()
        sess.run([init_global, init_local])
        niter = 0

        for epoch in range(args.training_epochs):

            # total_batch = int(N / args.batch_size)
            # Loop over all batches
            perm = np.random.permutation(N)

            for i in range(0, N, args.batch_size):
                batch_x = args.X_train[perm[i:i + args.batch_size]]
                batch_y = args.Y_train[perm[i:i + args.batch_size]]
                _, cost_val, acc_val = sess.run(
                    [optimizer, cost, accuracy],
                    feed_dict={X: batch_x, y: batch_y}
                )
                niter += 1

                if niter % 100 == 0:
                    end_time = time.time()
                    # eval on dev set
                    acc_val_dev = accuracy.eval(feed_dict={X: args.X_dev,
                                                           y: args.Y_dev})

                    # save
                    timediff = end_time - start_time
                    res_return.plot.niter.append(niter)
                    res_return.plot.runtime.append(timediff)
                    res_return.plot.loss.append(cost_val)
                    res_return.plot.devAcc.append(acc_val_dev)

                    print(
                        "Step: {:>3d} RunTime: {:.3f} "
                        "Loss: {:.3f} Acc: {:.3f} DevAcc: {:.3f}".format(
                            niter, timediff,
                            cost_val, acc_val, acc_val_dev
                        )
                    )
                    start_time = time.time()

        end_time = time.time()
        print("=" * 21 + "Optimization Finish" + "=" * 21)
        acc_val_test, probs = sess.run(
            [accuracy, labels_distribution.probs],
            feed_dict={X: args.X_test, y: args.Y_test}
        )
        print("Step: {:>3d} RunTime: {:.3f} TestAcc:{:.3f}".format(
            niter, end_time - algstart, acc_val_test
        ))

# extract weights & bias

    res_return.probs = np.asarray(probs)
    res_return.acc = np.asarray(acc_val_test)

    return res_return


def bnn(args):

    # %% Model

    class Dummy():
        pass

    tf.reset_default_graph()

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    tfd = tf.contrib.distributions

    N = args.X_train.shape[0]
    dim = list(args.X_train.shape[1:])
    K = args.Y_train.shape[1]  # num of class

    X = tf.placeholder(tf.float32, [None] + dim)
    y = tf.placeholder(tf.float32, [None, K])

    neural_net = nnet.convnet(
        numclass=K, inshape=args.inshape, isBay=True,
        priorstd=args.priorstd, poststd=args.poststd
    )
    logits = neural_net(X)

    labels_distribution = tfd.Categorical(logits=logits)

    # %% Loss

    neg_log_likelihood = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    )
    kl = sum(neural_net.losses) / N
    elbo_loss = neg_log_likelihood + args.KLscale * kl

    # %% Metrics

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # %% Posterior

    names = []
    qmeans = []
    qstds = []
    Wsample = []

    for i, layer in enumerate(neural_net.layers):
        if hasattr(layer, "kernel_posterior"):
            q = layer.kernel_posterior
            names.append("Layer {}".format(i))
            qmeans.append(q.mean())
            qstds.append(q.stddev())
            Wsample.append(q.sample(args.num_monte_carlo))

    # %% Train

    optimizer = tf.train.AdamOptimizer(
        args.learning_rate).minimize(elbo_loss)

    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    res_return = Dummy()
    res_return.plot = Dummy()
    res_return.plot.niter = []
    res_return.plot.runtime = []
    res_return.plot.loss = []
    res_return.plot.devAccMean = []
    res_return.plot.devAccUp = []
    res_return.plot.devAccDown = []

    with tf.Session() as sess:
        print("=" * 21 + "Optimization Start" + "=" * 21)
        start_time, algstart = time.time(), time.time()
        sess.run([init_global, init_local])
        niter = 0

        for epoch in range(args.training_epochs):

            perm = np.random.permutation(N)
            for i in range(0, N, args.batch_size):
                batch_x = args.X_train[perm[i:i + args.batch_size]]
                batch_y = args.Y_train[perm[i:i + args.batch_size]]
                _, loss_val, acc_val = sess.run(
                    [optimizer, elbo_loss, accuracy],
                    feed_dict={X: batch_x, y: batch_y}
                )
                niter += 1

                if niter % 100 == 0:
                    end_time = time.time()
                    # eval on dev set
                    acc_val_dev = np.asarray([
                        sess.run(accuracy,
                            feed_dict={X: args.X_dev, y: args.Y_dev}) for _ in range(args.num_monte_carlo)
                    ])

                    # save
                    timediff = end_time - start_time
                    AccMean = np.mean(acc_val_dev)
                    AccStd = np.std(acc_val_dev)
                    timediff = end_time - start_time
                    res_return.plot.niter.append(niter)
                    res_return.plot.runtime.append(timediff)
                    res_return.plot.loss.append(loss_val)
                    res_return.plot.devAccMean.append(AccMean)
                    res_return.plot.devAccUp.append(AccMean + AccStd)
                    res_return.plot.devAccDown.append(AccMean - AccStd)

                    print(
                        "Step: {:>3d} RunTime: {:.3f} Loss: {:.3f} "
                        "AccDevM: {:.3f} AccDevU: {:.3f}".format(
                            niter, timediff,
                            loss_val, AccMean, AccMean + AccStd
                        )
                    )
                    start_time = time.time()

        end_time = time.time()
        print("=" * 21 + "Optimization Finish" + "=" * 21)

        tmp = [sess.run(
            [accuracy, labels_distribution.probs],
            feed_dict={X: args.X_test, y: args.Y_test}
        )for _ in range(args.num_monte_carlo)]
        [acc_val_test, probs] = list(zip(* tmp))
        acc_val_test = np.asarray(acc_val_test)

        print("Step: {:>3d} RunTime: {:.3f} TestAcc:{:.3f}".format(
            niter, end_time - algstart, np.mean(acc_val_test)
        ))

        # evaluate the posterior distributions for kernel
        qm_vals, qs_vals, W_postsam = sess.run((qmeans, qstds, Wsample))

    # Return result

    res_return.probs = np.asarray(probs)
    res_return.acc = np.asarray(acc_val_test)
    res_return.posterior = Dummy()
    res_return.posterior.mean = qm_vals
    res_return.posterior.std = qs_vals
    res_return.posterior.samples = W_postsam
    res_return.names = names

    return res_return
