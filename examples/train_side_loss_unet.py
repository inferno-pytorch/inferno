"""
Train UNet Example
================================

This example should illustrate how to use the trainer class
in conjunction with a unet, we use a toy dataset here

"""

import torch
import torch.nn as nn
from inferno.io.box.binary_blobs import get_binary_blob_loaders
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger

from inferno.extensions.layers.convolutional import ConvELU2D, Conv2D,ConvELU3D, Conv3D,ConvActivation
from inferno.extensions.layers.building_blocks import ResBlock
from inferno.extensions.layers.unet import ResBlockUNet
from inferno.utils.torch_utils import unwrap
import pylab



class MySideLossUNet(nn.Module):
    def __init__(self, in_channels, out_channels, depth=3):
        super(MySideLossUNet, self).__init__()

        self.depth = depth
        self.unet = ResBlockUNet(in_channels=in_channels, out_channels=in_channels*2,
                                 dim=2, unet_kwargs=dict(depth=depth),
                                 side_out_parts=['bottom', 'up'])

        # number of out channels
        self.n_channels_per_output = self.unet.n_channels_per_output

        # 1x1 conv to give the side outs of the unet 
        # the right number of channels 
        # and a Upsampling to give the right shape
        upscale_factor = 2**self.depth
        conv_and_scale = []
        for n_channels in self.n_channels_per_output:

            # conv blocks
            conv = Conv2D(in_channels=n_channels, out_channels=out_channels, kernel_size=1)
            if upscale_factor > 1:
                upsample = nn.Upsample(scale_factor=upscale_factor)
                conv_and_scale.append(nn.Sequential(conv, upsample))
            else:
                conv_and_scale.append(conv)

            upscale_factor //= 2

        self.conv_and_scale = nn.ModuleList(conv_and_scale)


        # combined number of channels after concat
        # concat side output predictions with main output of unet
        self.n_channels_combined = (self.depth + 1)* out_channels + in_channels*2

        self.final_block = nn.Sequential(
            ResBlock(dim=2,in_channels=self.n_channels_combined, out_channels=self.n_channels_combined),
            ResBlock(in_channels=self.n_channels_combined, out_channels=out_channels, 
                    dim=2, activated=False),
        )
   
    def forward(self, input):
        outs = self.unet(input)
        assert len(outs) == len(self.n_channels_per_output)

        # convert the unet output into the right number of
        preds = [None] * len(outs)
        for i,out in enumerate(outs):
            preds[i] = self.conv_and_scale[i](out)

        # this is the side output
        preds =  tuple(preds)

        # concat side output predictions with main output of unet
        combined = torch.cat(preds + (outs[-1],), 1)

        final_res = self.final_block(combined)

        # return everything
        return preds + (final_res,)


class MySideLoss(nn.Module):
    """Wrap a criterion. Collect regularization losses from model and combine with wrapped criterion.
    """

    def __init__(self):
        super(MySideLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduce=True)

        w = 1.0
        l = None


    def forward(self, predictions, target):
        w = 1.0
        l = None
        for p in predictions:
            ll = self.criterion(p, target)*w
            if l is None:
                l = ll
            else:
                l += ll
            w *= 2
        return l

            



# lil helper to make sure dirs exits
import os
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

# change directories to your needs
LOG_DIRECTORY = ensure_dir('log')
SAVE_DIRECTORY = ensure_dir('save')
DATASET_DIRECTORY = ensure_dir('dataset')


USE_CUDA = True

# Build a residual unet where the last layer is not activated
sl_unet = MySideLossUNet(in_channels=5, out_channels=2)

model = nn.Sequential(
    ResBlock(dim=2, in_channels=1, out_channels=5),
    sl_unet
)
train_loader, test_loader, validate_loader = get_binary_blob_loaders(
    train_batch_size=3,
    length=512, # <= size of the images
    gaussian_noise_sigma=1.5 # <= how noise are the images
)

# Build trainer
trainer = Trainer(model) \
.build_criterion(MySideLoss()) \
.build_optimizer('Adam') \
.validate_every((10, 'epochs')) \
.save_every((10, 'epochs')) \
.save_to_directory(SAVE_DIRECTORY) \
.set_max_num_epochs(40) \

# Bind loaders
trainer \
    .bind_loader('train', train_loader)\
    .bind_loader('validate', validate_loader)

if USE_CUDA:
    trainer.cuda()

# Go!
#trainer.fit()


# predict:
trainer.load(best=True)
trainer\
.bind_loader('train', train_loader)\
.bind_loader('validate', validate_loader)
trainer.eval_mode()

if USE_CUDA:
    trainer.cuda()

# look at an example
for x,y in test_loader:
    if USE_CUDA:
        x = x.cuda()

    ###############################
    #TODO show all side outs
    ##############################
    yy = trainer.apply_model(x)[-1]
    yy = nn.functional.softmax(yy,dim=1)
    yy = unwrap(yy, as_numpy=True, to_cpu=True)
    x  = unwrap(x,  as_numpy=True, to_cpu=True)
    y  = unwrap(y, as_numpy=True, to_cpu=True)

    batch_size = yy.shape[0]
    for b in range(batch_size):

        fig = pylab.figure()
        ax1 = fig.add_subplot(1,3,1)
        ax1.imshow(x[b,0,...])
        ax2 = fig.add_subplot(1,3,2)
        ax2.imshow(y[b,...])
        ax3 = fig.add_subplot(1,3,3)
        ax3.imshow(yy[b,1,...])

        pylab.show()

    break