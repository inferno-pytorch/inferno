"""
2D Residual Unet Example
================================

This example should illustrate how to a Residual Unet 
as a layer

"""
import torch
import inferno.extensions.layers.unet as unet


# a 2D data with 10 input channels
x = torch.rand(1, 10, 64,64)
x = torch.autograd.Variable(x)

# a unet with resiudal blocks
model = unet.ResBlockUnet(in_channels=10, ndim=2)

# pass x trough unet 
# by default the output number 
# of channels is twice the input channelss
out = model(x)

print(out.size())



"""
Chained 3D Residual Unets
================================

This example should illustrate how chain
multiple residual unets
"""



# 3D data with 5 input channels
x = torch.rand(1, 5, 32,32, 32)
x = torch.autograd.Variable(x)

# a unet with resiudal blocks
model_a = unet.ResBlockUnet(in_channels=5, depth=2, out_channels=12, ndim=3)

# if the last layer in the second unet
# shall be non-activated  we set 
# activated to False, this will only affect the
# very last convolution of the net
model_b = unet.ResBlockUnet(in_channels=12, depth=3, out_channels=2,  ndim=3,
                            activated=False)

# chain models
model = torch.nn.Sequential(model_a, model_b)

# and use the model
out = model(x)
print(out.size())
