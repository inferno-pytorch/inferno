=====
Usage
=====


Inferno is a utility library built around [PyTorch](http://pytorch.org/), designed to help you train and even build complex pytorch models. And in this tutorial, we'll see how! If you're new to PyTorch, I highly recommended you work through the [Pytorch tutorials](http://pytorch.org/tutorials/) first.

Building a PyTorch Model
~~~~~~~~~~~~~~~~~~~~~~~~~~

Inferno's training machinery works with just about any valid [Pytorch module](http://pytorch.org/docs/master/nn.html#torch.nn.Module). However, to make things even easier, we also provide pre-configured layers that work out-of-the-box. Let's use them to build a convolutional neural network for Cifar-10.

.. code:: python

    import torch.nn as nn
    from inferno.extensions.layers.convolutional import ConvELU2D
    from inferno.extensions.layers.reshape import Flatten

`ConvELU2D` is a 2-dimensional convolutional layer with orthogonal weight initialization and [ELU](http://pytorch.org/docs/master/nn.html#torch.nn.ELU) activation. `Flatten` reshapes the 4 dimensional activation tensor to a matrix. Let's use the Sequential container to chain together a bunch of convolutional and pooling layers, followed by a linear and softmax layer. 


.. code:: python

    model = nn.Sequential(
        ConvELU2D(in_channels=3, out_channels=256, kernel_size=3),
        nn.MaxPool2d(kernel_size=2, stride=2),
        ConvELU2D(in_channels=256, out_channels=256, kernel_size=3),
        nn.MaxPool2d(kernel_size=2, stride=2),
        ConvELU2D(in_channels=256, out_channels=256, kernel_size=3),
        nn.MaxPool2d(kernel_size=2, stride=2),
        Flatten(),
        nn.Linear(in_features=(256 * 4 * 4), out_features=10),
        nn.Softmax()
    )

Models this size don't win competitions anymore, but it'll do for our purpose. 

Data Logistics 
**************************

With our model built, it's time to worry about the data generators. Or is it? 

.. code:: python

    from inferno.io.box.cifar import get_cifar10_loaders
    train_loader, validate_loader = get_cifar10_loaders('path/to/cifar10',
                                                        download=True,
                                                        train_batch_size=128,
                                                        test_batch_size=100)

CIFAR-10 works out-of-the-`box` (pun very much intended) with all the fancy data-augmentation and normalization. Of course, it's perfectly fine if you have your own [`DataLoader`](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader).


Preparing the Trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With our model and data loaders good to go, it's finally time to build the trainer. To start, let's initialize one. 

.. code:: python

    from inferno.trainers.basic import Trainer

    trainer = Trainer(model)
    # Tell trainer about the data loaders
    trainer.bind_loader('train', train_loader).bind_loader('validate', validate_loader)


Now to the things we could do with it. 

Setting up Checkpointing
***************************************
When training a model for days, it's usually a good idea to store the current training state to disk every once in a while. To set this up, we tell `trainer` where to store these *checkpoints* and how often. 

.. code:: python

    trainer.save_to_directory('path/to/save/directory').save_every((25, 'epochs'))

So we're saving once every 25 epochs. But what if an epoch takes forever, and you don't wish to wait that long? 

.. code:: python

    trainer.save_every((1000, 'iterations'))

In this setting, you're saving once every 1000 iterations (= batches). But we might also want to create a checkpoint when the validation score is the best. Easy as 1, 2,

.. code:: python

    trainer.save_at_best_validation_score()

Remember that a checkpoint contains the entire training state, and not just the model. Everything is included in the checkpoint file, including optimizer, criterion, and callbacks but __not the data loaders__. 

Setting up Validation
**************************
Let's say you wish to validate once every 2 epochs.

.. code:: python

    trainer.validate_every((2, 'epochs'))


To be able to validate, you'll need to specify a validation metric.

.. code:: python

    trainer.build_metric('CategoricalError')

Inferno looks for a metric `'CategoricalError'` in `inferno.extensions.metrics`. To specify your own metric, subclass `inferno.extensions.metrics.base.Metric` and implement the `forward` method. With that done, you could:

.. code:: python

    trainer.build_metric(MyMetric)

or 

.. code:: python

    trainer.build_metric(MyMetric, **my_metric_kwargs)


A metric might be way too expensive to evaluate every training iteration without slowing down the training. If this is the case and you'd like to evaluate the metric every (say) 10 *training* iterations:

.. code:: python

    trainer.evaluate_metric_every((10, 'iterations'))

However, while validating, the metric is evaluated once every iteration.

Setting up the Criterion and Optimizer
***************************************
With that out of the way, let's set up a training criterion and an optimizer. 

.. code:: python

    # set up the criterion
    trainer.build_criterion('CrossEntropyLoss')

The `trainer` looks for a `'CrossEntropyLoss'` in `torch.nn`, which it finds. But any of the following would have worked: 

.. code:: python

    trainer.build_criterion(nn.CrossEntropyLoss)

or 

.. code:: python

    trainer.build_criterion(nn.CrossEntropyLoss())

What this means is that if you have your own loss criterion that has the same API as any of the criteria found in `torch.nn`, you should be fine by just plugging it in. 

The same holds for the optimizer:

.. code:: python

    trainer.build_optimizer('Adam', weight_decay=0.0005)

Like for criteria, the `trainer` looks for a `'Adam'` in `torch.optim` (among other places), and initializes it with `model`'s parameters. Any keywords you might use for `torch.optim.Adam`, you could pass them to the `build_optimizer` method. 

Or alternatively, you could use:

.. code:: python

    from torch.optim import Adam

    trainer.build_optimizer(Adam, weight_decay=0.0005)


If you implemented your own optimizer (by subclassing `torch.optim.Optimizer`), you should be able to use it instead of `Adam`. Alternatively, if you already have an optimizer *instance*, you could do:

.. code:: python

    optimizer = MyOptimizer(model.parameters(), **optimizer_kwargs)
    trainer.build_optimizer(optimizer)


Setting up Training Duration
********************************
You probably don't want to train forever, in which case you must specify: 

.. code:: python

    trainer.set_max_num_epochs(100)

or 

.. code:: python

    trainer.set_max_num_iterations(10000)


If you like to train indefinitely (or until you're happy with the results), use:

.. code:: python

    trainer.set_max_num_iterations('inf')

In this case, you'll need to interrupt the training manually with a `KeyboardInterrupt`. 

Setting up Callbacks
*********************
Callbacks are pretty handy when it comes to interacting with the `Trainer`. More precisely: `Trainer` defines a number of events as 'triggers' for callbacks. Currently, these are: 

.. code:: python

    BEGIN_OF_FIT,
    END_OF_FIT,
    BEGIN_OF_TRAINING_RUN,
    END_OF_TRAINING_RUN,
    BEGIN_OF_EPOCH,
    END_OF_EPOCH,
    BEGIN_OF_TRAINING_ITERATION,
    END_OF_TRAINING_ITERATION,
    BEGIN_OF_VALIDATION_RUN,
    END_OF_VALIDATION_RUN,
    BEGIN_OF_VALIDATION_ITERATION,
    END_OF_VALIDATION_ITERATION,
    BEGIN_OF_SAVE,
    END_OF_SAVE


As an example, let's build a simple callback to interrupt the training on NaNs. We check at the end of every training iteration whether the training loss is NaN, and accordingly raise a `RuntimeError`. 

.. code:: python

    import numpy as np
    from inferno.trainers.callbacks.base import Callback

    class NaNDetector(Callback):
        def end_of_training_iteration(self, **_):
            # The callback object has the trainer as an attribute. 
            # The trainer populates its 'states' with torch tensors (NOT VARIABLES!)
            training_loss = self.trainer.get_state('training_loss')
            # Extract float from torch tensor
            training_loss = training_loss[0]
            if np.isnan(training_loss):
                raise RuntimeError("NaNs detected!")


With the callback defined, all we need to do is register it with the trainer:

.. code:: python

    trainer.register_callback(NaNDetector())


So the next time you get `RuntimeError: "NaNs detected!`, you know the drill. 

Using Tensorboard
**************************

Inferno supports logging scalars and images to Tensorboard out-of-the-box, though this requires you have at least [tensorflow-cpu](https://github.com/tensorflow/tensorflow) installed. Let's say you want to log scalars every iteration and images every 20 iterations:

.. code:: python

    from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger

    trainer.build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                           log_images_every=(20, 'iterations')),
                         log_directory='/path/to/log/directory')


After you've started training, use a bash shell to fire up tensorboard with:

.. code:: bash

    $ tensorboard --logdir=/path/to/log/directory --port=6007
    
and navigate to `localhost:6007` with your favorite browser.

Fine print: missing the `log_images_every` keyword argument to `TensorboardLogger` will result in images being logged every iteration. If you don't have a fast hard drive, this might actually slow down the training. To not log images, just use `log_images_every='never'`. 

Using GPUs
*************

To use just one GPU: 

.. code:: python

    trainer.cuda()


For multi-GPU data-parallel training, simply pass `trainer.cuda` a list of devices: 

.. code:: python

    trainer.cuda(devices=[0, 1, 2, 3])


__Pro-tip__: Say you only want to use GPUs 0, 3, 5 and 7 (your colleagues might love you for this). Before running your training script, simply: 

.. code:: bash

    $ export CUDA_VISIBLE_DEVICES=0,3,5,7
    $ python train.py

This maps device 0 to 0, 3 to 1, 5 to 2 and 7 to 3. 

One more thing
**************************


Once you have everything configured, use 

.. code:: python

    trainer.fit()

to commence training! This last step is kinda important. :wink:

Cherries:
~~~~~~~~~~~~~~~~~~~~~~


Building Complex Models with the Graph API
****************************************************



Work in Progress:


Parameter Initialization
**************************

Work in Progress:


Support
*************
Work in Progress:

