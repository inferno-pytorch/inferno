# Inferno: A Short Tutorial

Inferno is a utility library built around [PyTorch](http://pytorch.org/), designed to help you train and even build complex pytorch models. And in this tutorial, we'll see how! If you're new to PyTorch, I highly recommended you work through the [Pytorch tutorials](http://pytorch.org/tutorials/) first.

## Pytorch Recap
Before we dive in, let's do a quick recap of how PyTorch works. We'll start by building a simple neural network, for which we subclass `nn.Module`:

```python
import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self):
        # Initialize the superclass
        super(SimpleNN, self).__init__()
        self.linear1 = nn.Linear(in_features=1024, out_features=512)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=512, out_features=10)
        self.softmax = nn.Softmax()
    
    def forward(self, input):
        linear1 = self.linear1(input)
        relu = self.relu(linear1)
        linear2 = self.linear2(relu)
        softmax = self.softmax(linear2)
        return softmax
```

...

## How does Inferno help? 
...

### Training with Tensorboard Support
...

### Building Complex Models with the Graph API
...

### Cherries: Parameter Initialization and Callbacks
...

### Data Logistics
...

## Support
...
