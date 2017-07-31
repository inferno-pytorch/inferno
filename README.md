# inferno

Inferno is a little library providing utilities and convenience functions/classes around [PyTorch](https://github.com/pytorch/pytorch). It's a work-in-progress and the API might change (for the better!) without much of a warning, so hang in tight! 

## Current Features
Current features include: 
* a basic [Trainer class](https://github.com/nasimrahaman/inferno/blob/master/inferno/trainers/basic.py) to encapsulate the training boilerplate (iteration/epoch loops, validation and checkpoint creation),
* a [graph API](https://github.com/nasimrahaman/inferno/blob/master/inferno/extensions/layers/graph.py) for building models with complex architectures, powered by [networkx](https://github.com/networkx/networkx). 
* [easy data-parallelism](https://github.com/nasimrahaman/inferno/blob/master/tests/training/basic.py#L117) over multiple GPUs, 
* [a submodule](https://github.com/nasimrahaman/inferno/blob/master/inferno/extensions/initializers) for `torch.nn.Module`-level parameter initialization,
* [a submodule](https://github.com/nasimrahaman/inferno/blob/master/inferno/io/transform) for data preprocessing / transforms,
* [support](https://github.com/nasimrahaman/inferno/blob/master/inferno/trainers/callbacks/logging/tensorboard.py) for [Tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) (best with atleast [tensorflow-cpu](https://github.com/tensorflow/tensorflow) installed),
* [a callback API](https://github.com/nasimrahaman/inferno/tree/master/inferno/trainers/callbacks) to enable flexible interaction with the trainer,
* [various utility layers](https://github.com/nasimrahaman/inferno/tree/master/inferno/extensions/layers) with more underway,
* [a submodule](https://github.com/nasimrahaman/inferno/blob/master/inferno/io/volumetric) for volumetric datasets, and more!

## Show me the Code!
```python
import torch.nn as nn
from inferno.io.box.cifar10 import get_cifar10_loaders
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.extensions.layers.convolutional import ConvELU2D
from inferno.extensions.layers.reshape import Flatten

# Fill these in:
LOG_DIRECTORY = '...'
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
  .set_max_num_epochs(10) \
  .build_logger(TensorboardLogger(), log_directory=LOG_DIRECTORY)

# Bind loaders
trainer \
    .bind_loader('train', train_loader) \
    .bind_loader('validate', validate_loader)

if USE_CUDA:
  trainer.cuda()

# Go!
trainer.fit()
```

To visualize the training progress, navigate to `LOG_DIRECTORY` and fire up tensorboard with 

```
$ tensorboard --logdir=${PWD} --port=6007
```

and navigate to `localhost:6007` with your browser.

## Future Features: 
Planned features include: 
* a class to encapsulate Hogwild! training over multiple GPUs, 
* minimal shape inference with a dry-run,
* proper packaging and documentation,
* cutting-edge fresh-off-the-press implementations of what the future has in store. :)

## Contributing
Got an idea? Awesome! Start a discussion by opening an issue or contribute with a pull request.  

## Who's Who?
As of today, this library is maintained by [Nasim Rahaman](https://github.com/nasimrahaman) with sizeable contributions from [Maurice Weiler](https://github.com/mauriceweiler) and [Steffen Wolf](https://github.com/Steffen-Wolf) @
[Image Analysis and Learning Lab](https://hci.iwr.uni-heidelberg.de/mip),
[Heidelberg Collaboratory for Image Processing](https://hci.iwr.uni-heidelberg.de/). 
