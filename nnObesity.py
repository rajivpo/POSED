import tensorflow as tf
import numpy as np
import csv
import pandas as pd
from random import shuffle, randint, random

# Fixed Params
num_vars = 123
n_classes = 1

x = tf.placeholder('float', [None, num_vars])
y = tf.placeholder('float')
# pkeep = tf.placeholder('float')

def neural_network_model(data, n_nodes_hl1):
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
    # l1 = tf.nn.dropout(l1, pkeep)

    # l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    # l2 = tf.nn.relu(l2)
    # l2 = tf.nn.dropout(l2, pkeep)

    # l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    # l3 = tf.nn.relu(l3)
    # l3 = tf.nn.dropout(l3, pkeep)

    output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']

    return output, hidden_1_layer['weights']

def train_neural_network(x, n_nodes_hl1, batch_size, l1_alpha, eta, test_no):

    full_train_x, full_train_y, text_x, test_y = getData()
    prediction, weight_matrix = neural_network_model(x, n_nodes_hl1)
    cost = tf.reduce_mean(tf.square(y - prediction))
    l1_regularizer = tf.contrib.layers.l1_regularizer(scale=l1_alpha, scope=None)

    weights = tf.trainable_variables() # all vars of graph
    regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)

    regularized_loss = cost + regularization_penalty
    optimizer = tf.train.GradientDescentOptimizer(eta).minimize(regularized_loss)
    
    num_epochs = 200

    whole_train_loss = 0
    whole_valid_loss = 0

    for cv_no in range(5):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            valid_x = []
            valid_y = []
            train_x = []
            train_y = []

            for i in range(len(full_train_x)):
                if i % 5 == cv_no:
                    valid_x.append(full_train_x[i])
                    valid_y.append(full_train_y[i])
                else:
                    train_x.append(full_train_x[i])
                    train_y.append(full_train_y[i])

            for epoch_no in range(num_epochs):

                # Shuffle examples
                c = list(zip(train_x, train_y))
                shuffle(c)
                train_x, train_y = zip(*c)

                train_loss = 0
                for batch_no in range(int(len(train_x)/batch_size)):
                    epoch_x = train_x[batch_no*batch_size:(batch_no+1)*batch_size]
                    epoch_y = train_y[batch_no*batch_size:(batch_no+1)*batch_size]
                    _, c, pred = sess.run([optimizer, cost, prediction], feed_dict = {x: epoch_x, y: epoch_y})

                _, valid_loss = sess.run([prediction, cost], feed_dict = {x: valid_x, y: valid_y})
                _, train_loss = sess.run([prediction, cost], feed_dict = {x: train_x, y: train_y})

            whole_train_loss += train_loss
            whole_valid_loss += valid_loss
            weights = sess.run([weight_matrix])
            df = pd.DataFrame(np.transpose(weights[0]))
            df.to_csv("weightsTest" + str(cv_no+1) + ".csv")

            print('Test', test_no, ', ALL EPOCHS COMPLETED CV', cv_no+1, 'w/train loss:', train_loss, 'valid loss:', valid_loss)

    print('TRAINING COMPLETE, AVG TRAIN LOSS:', whole_train_loss/5, 'AVG VALID LOSS:', whole_valid_loss/5)
    
    return whole_train_loss/5, whole_valid_loss/5

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
        if lst_max - lst_min == 0:
            continue
        for row in data:
            row[j] = (row[j] - lst_min)/(lst_max - lst_min)
    return data

# def normSTD(data, to_csv):
#     for j in range(len(data[0])):
#         lst = [row[j] for row in data]
#         lst_std = np.std(np.asarray(lst))
#         lst_mean = float(sum(lst))/len(lst)
#         if lst_std == 0:
#             continue
#         for row in data:
#             row[j] = (row[j] - lst_mean)/(lst_std)
    
#     if to_csv:
#         df = pd.DataFrame(data)
#         df.to_csv("normSTD.csv")

#     return data

def randomSearchForHPs(num_trials):
    best_n = 100
    best_b = 100
    best_alpha = 0.5
    best_eta = 0.5
    best_train_loss = 1000
    best_valid_loss = 1000
    for i in range(num_trials):
        print("Num nodes", best_n, ", Batch size", best_b, ", Alpha", best_alpha, ", Eta", best_eta)
        print("Best train loss", best_train_loss)
        print("Best valid loss", best_valid_loss)

        n_nodes_hl1 = randint(5, 200)
        batch_size = randint(5, 200)
        if (random() < 0.7):
            l1_alpha = 0.01* random()
        else:
            l1_alpha = random()
        eta = 0.1*random()
        print(i)
        print(n_nodes_hl1)
        print(batch_size)
        print(l1_alpha)
        print(eta)
        # train_loss, valid_loss = train_neural_network(x, 100, 100, 0.05, 0.05, 0)
        train_loss, valid_loss = train_neural_network(x, n_nodes_hl1, batch_size, l1_alpha, eta, i)
        if valid_loss < best_valid_loss:
            best_n = n_nodes_hl1
            best_b = batch_size
            best_alpha = l1_alpha
            best_eta = eta
            best_train_loss = train_loss
            best_valid_loss = valid_loss
    print("Num nodes", best_n, ", Batch size", best_b, ", Alpha", best_alpha, ", Eta", best_eta)
    print("Best train loss", best_train_loss)
    print("Best valid loss", best_valid_loss)

def trainWithHPs(x, n_nodes_hl1, batch_size, alpha, eta):
    train_loss, valid_loss = train_neural_network(x, n_nodes_hl1, batch_size, alpha, eta, 1)
    print("train loss", train_loss)
    print("valid loss", valid_loss)

trainWithHPs(x, 62, 192, 0.0051, 0.0332)
# randomSearchForHPs(30)