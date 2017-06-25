# inferno

Inferno is a little library providing utilities and convenience functions/classes around [PyTorch](https://github.com/pytorch/pytorch). It's a work-in-progress, hang on tight! 

## Current Features
Current features include: 
* a basic [Trainer class](https://github.com/nasimrahaman/inferno/blob/master/inferno/trainers/basic.py) to encapsulate the training boilerplate (iteration/epoch loops, validation and checkpoint creation),
* [a class](https://github.com/nasimrahaman/inferno/blob/master/inferno/extensions/initializers/base.py#L4) defining API for `torch.nn.Module`-level parameter initialization,
* [a class](https://github.com/nasimrahaman/inferno/blob/master/inferno/io/transform/base.py#L5) defining API for data preprocessing / transforms,
* [classes](https://github.com/nasimrahaman/inferno/blob/master/inferno/io/volumetric/volume.py) for volumetric datasets, and more!

## Future Features: 
Planned features include: 
* a class to encapsulate Hogwild! training over multiple GPUs,
* a [callback API](https://github.com/nasimrahaman/inferno/blob/master/inferno/trainers/callbacks/base.py) compatible with Keras. 
* cutting-edge fresh-off-the-press implementations of what the future has in store. :)

## Contributing
Got an idea? Awesome! Start a discussion by opening an issue or contribute with a pull request.  

## Who's Who?
As of today, this library is maintained by Nasim Rahaman @
[Image Analysis and Learning Lab](https://hci.iwr.uni-heidelberg.de/mip),
[Heidelberg Collaboratory for Image Processing](https://hci.iwr.uni-heidelberg.de/). 
