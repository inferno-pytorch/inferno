# Inferno: A Short Tutorial

Inferno is a utility library built around [PyTorch](http://pytorch.org/), designed to help you train and even build complex pytorch models. And in this tutorial, we'll see how! If you're new to PyTorch, I highly recommended you work through the [Pytorch tutorials](http://pytorch.org/tutorials/) first.

## Building a PyTorch Model
Inferno's training machinery works with just about any valid [Pytorch module](http://pytorch.org/docs/master/nn.html#torch.nn.Module). However, to make things even easier, we also provide pre-configured layers that work out-of-the-box. Let's use them to build a convolutional neural network for Cifar-10.

```python
import torch.nn as nn
from inferno.extensions.layers.convolutional import ConvELU2D
from inferno.extensions.layers.reshape import Flatten
```
`ConvELU2D` is a 2-dimensional convolutional layer with orthogonal weight initialization and [ELU](http://pytorch.org/docs/master/nn.html#torch.nn.ELU) activation. `Flatten` reshapes the 4 dimensional activation tensor to a matrix. Let's use the Sequential container to chain together a bunch of convolutional and pooling layers, followed by a linear and softmax layer. 

```python
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
```
Models this size don't win competitions anymore, but it'll do for our purpose. 

## Data Logistics 

With our model built, it's time to worry about the data generators. Or is it? 
```python
from inferno.io.box.cifar10 import get_cifar10_loaders
train_loader, validate_loader = get_cifar10_loaders('path/to/cifar10', 
                                                    download=True, 
                                                    train_batch_size=128, 
                                                    test_batch_size=100)
```
CIFAR-10 works out-of-the-`box` (pun very much intended) with all the fancy data-augmentation and normalization. Of course, it's perfectly fine if you have your own [`DataLoader`](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader).


## Preparing the Trainer

With our model and data loaders good to go, it's finally time to build the trainer. To start, let's initialize a one. 

```python
from inferno.trainers.basic import Trainer

trainer = Trainer(model)
```

Now to the things we could do with it. 

### Setting up Checkpointing
When training a model for days, it's usually a good idea to store the current training state to disk every once in a while. To set this up, we tell `trainer` where to store these *checkpoints* and how often. 
```python
trainer.save_to_directory('path/to/save/directory').save_every((25, 'epochs'))
```
So we're saving once every 25 epochs. But what if an epoch takes forever, and you don't wish to wait that long? 
```python
trainer.save_every((1000, 'iterations'))
```
In this setting, you're saving once every 1000 iterations (= batches). But we might also want to create a checkpoint when the validation score is the best. Easy as 1, 2,
```python
trainer.save_at_best_validation_score()
```
Remember that a checkpoint contains the entire training state, and not just the model. Everything is included in the checkpoint file, including optimizer, criterion, and callbacks but __not the data loaders__. 

### Setting up Validation
Let's say you wish to validate once every 2 epochs.

```python
trainer.validate_every((2, 'epochs'))
```

### Setting up the Criterion and Optimizer
With that out of the way, let's set up a training criterion and an optimizer. 

```python
# set up the criterion
trainer.build_criterion('CrossEntropyLoss')
```
The `trainer` looks for a `'CrossEntropyLoss'` in `torch.nn`, which it finds. But any of the following would have worked: 
```python
trainer.build_criterion(nn.CrossEntropyLoss)
```
or 
```python
trainer.build_criterion(nn.CrossEntropyLoss())
```
What this means is that if you have your own loss criterion that has the same API as any of the criteria found in `torch.nn`, you should be fine by just plugging it in. 

The same holds for the optimizer: 
```python
trainer.build_optimizer('Adam', weight_decay=0.0005)
```
Like for criteria, the `trainer` looks for a `'Adam'` in `torch.optim` (among other places), and initializes it with `model`'s parameters. Any keywords you might use for `torch.optim.Adam`, you could pass them to the `build_optimizer` method. 

Or alternatively, you could use:
```python
from torch.optim import Adam

trainer.build_optimizer(Adam, weight_decay=0.0005)
```

If you implemented your own optimizer (by subclassing `torch.optim.Optimizer`), you should be able to use it instead of `Adam`. Alternatively, if you already have an optimizer *instance*, you could do:

```python
optimizer = MyOptimizer(model.parameters(), **optimizer_kwargs)
trainer.build_optimizer(optimizer)
```

### Setting up Training Duration
You probably don't want to train forever, in which case you must specify: 
```python
trainer.set_max_num_epochs(100)
```
or 
```python
trainer.set_max_num_iterations(10000)
```
I usually like to train till I'm happy with the validation results - by setting `max_num_epochs` to a ridiculously large integer (yes, this is embarassing and will be fixed in the near future). 

### Setting up Callbacks
...

### Using Tensorboard
...

## Cherries

### Building Complex Models with the Graph API
...

### Parameter Initialization
...

## Support
...
