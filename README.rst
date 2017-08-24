
=======
Inferno
=======



.. image:: https://img.shields.io/pypi/v/inferno.svg
        :target: https://pypi.python.org/pypi/not_yet_inferno

.. image:: https://img.shields.io/travis/nasimrahaman/inferno.svg
        :target: https://travis-ci.org/nasimrahaman/inferno

.. image:: https://readthedocs.org/projects/pytorch-inferno/badge/?version=latest
        :target: http://pytorch-inferno.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status               

.. image:: https://pyup.io/repos/github/nasimrahaman/inferno/shield.svg
     :target: https://pyup.io/repos/github/nasimrahaman/inferno/
     :alt: Updates



.. image:: http://svgshare.com/i/2j7.svg





Inferno is a little library providing utilities and convenience functions/classes around 
`PyTorch <https://github.com/pytorch/pytorch>`_. 
It's a work-in-progress, but the first release is underway! 



* Free software: Apache Software License 2.0
* Documentation: https://pytorch-inferno.readthedocs.io (Work in progress).


Features
--------

Current features include: 
  *   a basic 
      `Trainer class <https://github.com/nasimrahaman/inferno/tree/master/docs#preparing-the-trainer>`_ 
      to encapsulate the training boilerplate (iteration/epoch loops, validation and checkpoint creation),
  *   a `graph API <https://github.com/nasimrahaman/inferno/blob/master/inferno/extensions/containers/graph.py>`_ for building models with complex architectures, powered by `networkx <https://github.com/networkx/networkx>`_. 
  *   `easy data-parallelism <https://github.com/nasimrahaman/inferno/tree/master/docs#using-gpus>`_ over multiple GPUs, 
  *   `a submodule <https://github.com/nasimrahaman/inferno/blob/master/inferno/extensions/initializers>`_ for `torch.nn.Module`-level parameter initialization,
  *   `a submodule <https://github.com/nasimrahaman/inferno/blob/master/inferno/io/transform>`_ for data preprocessing / transforms,
  *   `support <https://github.com/nasimrahaman/inferno/tree/master/docs#using-tensorboard>`_ for `Tensorboard <https://www.tensorflow.org/get_started/summaries_and_tensorboard>`_ (best with atleast `tensorflow-cpu <https://github.com/tensorflow/tensorflow>`_ installed)
  *   `a callback API <https://github.com/nasimrahaman/inferno/tree/master/docs#setting-up-callbacks>`_ to enable flexible interaction with the trainer,
  *   `various utility layers <https://github.com/nasimrahaman/inferno/tree/master/inferno/extensions/layers>`_ with more underway,
  *   `a submodule <https://github.com/nasimrahaman/inferno/blob/master/inferno/io/volumetric>`_ for volumetric datasets, and more!





.. code:: python

  import torch.nn as nn
  from inferno.io.box.cifar10 import get_cifar10_loaders
  from inferno.trainers.basic import Trainer
  from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
  from inferno.extensions.layers.convolutional import ConvELU2D
  from inferno.extensions.layers.reshape import Flatten

  # Fill these in:
  LOG_DIRECTORY = '...'
  SAVE_DIRECTORY = '...'
  DATASET_DIRECTORY = '...'
  DOWNLOAD_CIFAR = True
  USE_CUDA = True

  # Build torch model
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

  # Load loaders
  train_loader, validate_loader = get_cifar10_loaders(DATASET_DIRECTORY,
                                                      download=DOWNLOAD_CIFAR)

  # Build trainer
  trainer = Trainer(model) \
    .build_criterion('CrossEntropyLoss') \
    .build_metric('CategoricalError') \
    .build_optimizer('Adam') \
    .validate_every((2, 'epochs')) \
    .save_every((5, 'epochs')) \
    .save_to_directory(SAVE_DIRECTORY) \
    .set_max_num_epochs(10) \
    .build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                    log_images_every='never'), 
                  log_directory=LOG_DIRECTORY)

  # Bind loaders
  trainer \
      .bind_loader('train', train_loader) \
      .bind_loader('validate', validate_loader)

  if USE_CUDA:
    trainer.cuda()

  # Go!
  trainer.fit()




To visualize the training progress, navigate to `LOG_DIRECTORY` and fire up tensorboard with 

.. code:: python

  $ tensorboard --logdir=${PWD} --port=6007


and navigate to `localhost:6007` with your browser.



Future Features: 
------------------------
Planned features include: 
  *   a class to encapsulate Hogwild! training over multiple GPUs, 
  *   minimal shape inference with a dry-run,
  *   proper packaging and documentation,
  *   cutting-edge fresh-off-the-press implementations of what the future has in store. :)



Credits
---------

This package was partially created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

