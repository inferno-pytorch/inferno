# inferno

Inferno is a little library providing utilities and convenience functions/classes around [PyTorch](https://github.com/pytorch/pytorch). It's a work-in-progress, hang on tight! 

## Current Features
Current features include: 
* a basic [Trainer class](https://github.com/nasimrahaman/inferno/blob/master/inferno/trainers/basic.py) to encapsulate the training boilerplate (iteration/epoch loops, validation and checkpoint creation),
* [a submodule](https://github.com/nasimrahaman/inferno/blob/master/inferno/extensions/initializers) for `torch.nn.Module`-level parameter initialization,
* [a submodule](https://github.com/nasimrahaman/inferno/blob/master/inferno/io/transform) for data preprocessing / transforms,
* [support](https://github.com/nasimrahaman/inferno/blob/master/inferno/trainers/callbacks/logging/tensorboard.py) for [Tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) (best with atleast [tensorflow-cpu](https://github.com/tensorflow/tensorflow) installed),
* [a callback API](https://github.com/nasimrahaman/inferno/tree/master/inferno/trainers/callbacks) to enable flexible interaction with the trainer,
* [various utility layers](https://github.com/nasimrahaman/inferno/tree/master/inferno/extensions/layers) with more underway,
* [a submodule](https://github.com/nasimrahaman/inferno/blob/master/inferno/io/volumetric) for volumetric datasets, and more!

## Future Features: 
Planned features include: 
* a class to encapsulate Hogwild! training over multiple GPUs, 
* cutting-edge fresh-off-the-press implementations of what the future has in store. :)

## Contributing
Got an idea? Awesome! Start a discussion by opening an issue or contribute with a pull request.  

## Who's Who?
As of today, this library is maintained by Nasim Rahaman @
[Image Analysis and Learning Lab](https://hci.iwr.uni-heidelberg.de/mip),
[Heidelberg Collaboratory for Image Processing](https://hci.iwr.uni-heidelberg.de/). 
