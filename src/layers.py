"""
Source Code for Homework 3 of ECBM E4040, Fall 2016, Columbia University

Instructor: Prof. Zoran Kostic

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""
import numpy 

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
import random
import timeit
import inspect
import sys

from utils import shared_dataset, load_data
from nn import LogisticRegression, HiddenLayer, LeNetConvPoolLayer, train_nn, drop, DropoutHiddenLayer,BNConvLayer,ConvLayer
from skimage import transform, exposure
import numpy as np
from theano.tensor.signal import pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils_4 import load_data_prob4


import math
import time
import BatchNormalization as BN



#Reference code can be found in http://deeplearning.net/tutorial/code/convolutional_mlp.py
# Code for Adam modified from https://gist.github.com/Newmu/acb738767acb4788bac3
def Adam(cost, params, grads, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    grads = grads
    k=0.
    i = theano.shared(numpy.asarray(k,dtype=theano.config.floatX))
    
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates



PIXELS = 32
imageSize = PIXELS * PIXELS
num_features = imageSize


def resotre_CNN( n_epochs=128, batch_size=500):
    

    rng = numpy.random.RandomState(23455)

    datasets = load_data_prob4()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.matrix('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

  
    layer0_input = x.reshape((batch_size, 3, 32, 32))
  
    layer0 = ConvLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(64, 3, 3, 3),
        subsamp=(1,1),
        bmode='half'
    )

    layer1 = ConvLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(64, 64, 3, 3),
        subsamp=(1,1),
        bmode='half'
    )


    maxpool1=pool.pool_2d(
        input=layer1.output,
        ds=(2,2),
        ignore_border=True
    )

    layer2 = ConvLayer(
        rng,
        input=maxpool1,
        image_shape=(batch_size, 64, 16, 16),
        filter_shape=(128, 64, 3, 3),
        subsamp=(1,1),
        bmode='half'
    )    

    layer3 = ConvLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, 128, 16, 16),
        filter_shape=(128, 128, 3, 3),
        subsamp=(1,1),
        bmode='half'
    )


    maxpool2=pool.pool_2d(
        input=layer3.output,
        ds=(2,2),
        ignore_border=True
    )

    layer4 = ConvLayer(
        rng,
        input=maxpool2,
        image_shape=(batch_size, 128, 8, 8),
        filter_shape=(256, 128, 3, 3),
        subsamp=(1,1),
        bmode='half'
    )
    
    upsampling11 = T.extra_ops.repeat(layer4.output,2,axis=2)
    
    upsampling12 = T.extra_ops.repeat(upsampling11,2,axis=3)    

  

    layer5 = ConvLayer(
        rng,
        input=upsampling12,
        image_shape=(batch_size, 256, 16, 16),
        filter_shape=(128, 256, 3, 3),
        subsamp=(1,1),
        bmode='half'
    )    

    layer6 = ConvLayer(
        rng,
        input=layer5.output,
        image_shape=(batch_size, 128, 16, 16),
        filter_shape=(128, 128, 3, 3),
        subsamp=(1,1),
        bmode='half'
    )

    sum1=layer3.output + layer6.output
    
    upsampling21 = T.extra_ops.repeat(sum1,2,axis=2)
    
    upsampling22 = T.extra_ops.repeat(upsampling21,2,axis=3)
    


    layer7 = ConvLayer(
        rng,
        input=upsampling22,
        image_shape=(batch_size, 128, 32, 32),
        filter_shape=(64, 128, 3, 3),
        subsamp=(1,1),
        bmode='half'
    )

    layer8 = ConvLayer(
        rng,
        input=layer7.output,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(64, 64, 3, 3),
        subsamp=(1,1),
        bmode='half'
    )

    sum2=layer1.output + layer8.output

    layer9 = ConvLayer(
        rng,
        input=sum2,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(3, 64, 3, 3),
        subsamp=(1,1),
        bmode='half'
    )

    # the cost we minimize during training is the NLL of the model
    cost = layer9.mse(y,batch_size)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [x],
        layer9.output

    )

    validate_model = theano.function(
        [x,index],
        layer9.mse(y, batch_size),
        givens={
            y: valid_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer9.params + layer8.params + layer7.params + layer6.params + layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    
    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.

#     updates = [
#         (param_i, param_i - learning_rate * grad_i)
#         for param_i, grad_i in zip(params, grads)
#     ]

    
    updates = Adam(cost,params, grads)

    train_model = theano.function(
        [x,index],
        cost,
        updates=updates,
        givens={
            y: train_set_x[batch_size * index: (index + 1) * batch_size]
        }
    )
    
    corrupted_input = T.matrix('corrupted_input')
    
    dropout_input = drop(corrupted_input, p=0.7)
    
    corrupt_input = theano.function(
        [index],
        dropout_input,
        givens={
            corrupted_input: train_set_x[index * batch_size: (index + 1) * batch_size],
        }        
    )
    
    corrupt_input_valid = theano.function(
        [index],
        dropout_input,
        givens={
            corrupted_input: valid_set_x[index * batch_size: (index + 1) * batch_size],
        }        
    )
    
    corrupt_input_test = theano.function(
        [index],
        dropout_input,
        givens={
            corrupted_input: test_set_x[index * batch_size: (index + 1) * batch_size],
        }        
    )
    
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 3000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.99  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    test_data_8, test_data_8_y =load_data_prob4(theano_shared=False)[2]
    


    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            
            x_corrupted = corrupt_input(minibatch_index)
            

            cost_ij = train_model(x_corrupted, minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = numpy.zeros(n_valid_batches)
                for k in range(n_valid_batches):
                    x_corrupted = corrupt_input_valid(k)
                    validation_losses[k] = validate_model(x_corrupted,k)
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, error (MSE) on validation set  %f ' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    x_corrupted = corrupt_input_test(0)
                    test_restored = test_model(x_corrupted)
                 
                    for k in xrange(8):
                        plt.imsave('image_corrupt_'+str(k)+'.png', x_corrupted[k].reshape((3,32,32)).transpose(1,2,0))
                        plt.imsave('image_restored_'+str(k)+'.png',test_restored[k].reshape((3,32,32)).transpose(1,2,0))
                        plt.imsave('image_original_'+str(k)+'.png',numpy.reshape(test_data_8[k],(3,32,32)).transpose(1,2,0))

                    
            
            if patience <= iter:
                done_looping = True
                break  
                    #test_score = numpy.mean(test_losses)
                    

    print(('The code for file ran for %.2fm' % ((end_time - start_time) / 60.)), sys.stderr)


