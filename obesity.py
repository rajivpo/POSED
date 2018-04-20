"""
Description:

I performed 7 tests.

Validation loss is calculated as the MSE of the NN prediction on the validation set. I do a 5-fold CV with usually ~1000 epochs/fold.

Setup is basic 1HL NN with different number of HUs, standard Adam optimizer with dropout regularization. All inputs are normalized.

Test1: 1HL, 100hu, BS=200 avg valid loss 21.738
Test2: 1HL, 100hu, BS=200 again, avg valid loss 21.753 this time. Full output below:

*****ALL EPOCHS COMPLETED CV 1 w/train loss: 137.440933228 valid loss: 21.3916
2018-04-20 03:11:46.772191: I c:\tf_jenkins\home\workspace\release-win\device\gpu\os\windows\tensorflow\core\common_runtime\gpu\gpu_device.cc:977] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 950M, pci bus id: 0000:01:00.0)
*****ALL EPOCHS COMPLETED CV 2 w/train loss: 134.437559128 valid loss: 19.8944
2018-04-20 03:12:18.317299: I c:\tf_jenkins\home\workspace\release-win\device\gpu\os\windows\tensorflow\core\common_runtime\gpu\gpu_device.cc:977] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 950M, pci bus id: 0000:01:00.0)
*****ALL EPOCHS COMPLETED CV 3 w/train loss: 142.249828339 valid loss: 23.6187
2018-04-20 03:12:49.880466: I c:\tf_jenkins\home\workspace\release-win\device\gpu\os\windows\tensorflow\core\common_runtime\gpu\gpu_device.cc:977] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 950M, pci bus id: 0000:01:00.0)
*****ALL EPOCHS COMPLETED CV 4 w/train loss: 121.711626053 valid loss: 21.4488
2018-04-20 03:13:22.003184: I c:\tf_jenkins\home\workspace\release-win\device\gpu\os\windows\tensorflow\core\common_runtime\gpu\gpu_device.cc:977] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 950M, pci bus id: 0000:01:00.0)
*****ALL EPOCHS COMPLETED CV 5 w/train loss: 138.500242233 valid loss: 22.4111
^^^^^^^TRAINING COMPLETE, AVG TRAIN LOSS: 134.868037796 AVG VALID LOSS: 21.7529239655

Test3: Trying 1 HL, 50hu. Same result essentially.
Test4: Trying 2 HL, L1:60hu L2:30hu. Used 3000 epochs instead of 1000 and got a validation loss of ~28.37, even worse than expected.
Test5: Back to 1 HL, only 20hu this time. Results did get worse, around 24.06
Test6: Trying something weird: 1HL, 1HU. This completely failed, was getting valid loss as high as 700.
Test7: I decreased the batch size down to 50 and there was a very marginal improvement to 19.75 validation loss.

It seems I can't really get much better than 19% MSE. It seems the other things may be doing better than NN at stuff like this.

I've also outputted various weight matrix tests to csv files, if you want to look at them on GitHub. It doesn't
seem like there is any discernable pattern to me. I've calc'd the std dev of each of the variables and there'll
all very close to each other.

"""

import tensorflow as tf
import numpy as np
import csv
import pandas as pd
from random import shuffle

n_nodes_hl1 = 50
# n_nodes_hl2 = 30
# n_nodes_hl3 = 100

num_vars = 123
n_classes = 1
batch_size = 100
myPkeep = 0.7

x = tf.placeholder('float', [None, num_vars])
y = tf.placeholder('float')
pkeep = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([num_vars, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    # hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
    #                   'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    # hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
    #                   'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    l1 = tf.nn.dropout(l1, pkeep)

    # l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    # l2 = tf.nn.relu(l2)
    # l2 = tf.nn.dropout(l2, pkeep)

    # l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    # l3 = tf.nn.relu(l3)
    # l3 = tf.nn.dropout(l3, pkeep)

    output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']

    return output, hidden_1_layer['weights']

def train_neural_network(x):

    full_train_x, full_train_y, text_x, test_y = getData()
    prediction, weight_matrix = neural_network_model(x)
    cost = tf.reduce_mean(tf.square(y - prediction))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    num_epochs = 500

    whole_train_loss = 0
    whole_valid_loss = 0

    for cv_no in range(5):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            bkpt = int(len(full_train_x)/5)
            valid_x = full_train_x[cv_no*bkpt:(cv_no+1)*bkpt]
            valid_y = full_train_y[cv_no*bkpt:(cv_no+1)*bkpt]
            train_x = full_train_x[:cv_no*bkpt] + full_train_x[(cv_no+1)*bkpt:]
            train_y = full_train_y[:cv_no*bkpt] + full_train_y[(cv_no+1)*bkpt:]

            for epoch_no in range(num_epochs):

                # Shuffle examples
                c = list(zip(train_x, train_y))
                shuffle(c)
                train_x, train_y = zip(*c)

                train_loss = 0
                for batch_no in range(int(len(train_x)/batch_size)):
                    epoch_x = train_x[batch_no*batch_size:(batch_no+1)*batch_size]
                    epoch_y = train_y[batch_no*batch_size:(batch_no+1)*batch_size]
                    _, c, pred = sess.run([optimizer, cost, prediction], feed_dict = {x: epoch_x, y: epoch_y, pkeep: myPkeep})
                    train_loss += c

                _, valid_loss = sess.run([prediction, cost], feed_dict = {x: valid_x, y: valid_y, pkeep: 1.0})

                if (epoch_no % 50 == 0):
                    print('valid loss at epoch', epoch_no, 'is', valid_loss)

            whole_train_loss += train_loss
            whole_valid_loss += valid_loss
            weights = sess.run([weight_matrix])
            df = pd.DataFrame(np.transpose(weights[0]))
            df.to_csv("weightsTest" + str(cv_no+1) + ".csv")

            print('*****ALL EPOCHS COMPLETED CV', cv_no+1, 'w/train loss:', train_loss, 'valid loss:', valid_loss)

    print('^^^^^^^TRAINING COMPLETE, AVG TRAIN LOSS:', whole_train_loss/5, 'AVG VALID LOSS:', whole_valid_loss/5)

def getData():
    data_reader = csv.reader(open('ObesityFullTrainSet.csv'))
    data_reader_test = csv.reader(open('ObesityTestSet.csv'))

    data_train_x = []
    data_train_y = []

    test_x = []
    test_y = []

    var_loc = -1

    first = True
    for row in data_reader:
        if first:
            first = False
            continue
        data_train_y.append(float(row[var_loc]))
        del row[var_loc]
        for i in range(len(row)):
            row[i] = float(row[i])
        data_train_x.append(row)

    first = True
    for row in data_reader_test:
        if first:
            first = False
            continue
        test_y.append(float(row[var_loc]))
        del row[var_loc]
        for i in range(len(row)):
            row[i] = float(row[i])
        test_x.append(row)

    return norm(data_train_x), data_train_y, norm(test_x), test_y

def norm(data):
    for j in range(len(data[0])):
        lst = [row[j] for row in data]
        lst_min = min(lst)
        lst_max = max(lst)
        if (lst_max - lst_min) == 0:
            continue
        for row in data:
            row[j] = (row[j] - lst_min)/(lst_max - lst_min)
    return data

train_neural_network(x)