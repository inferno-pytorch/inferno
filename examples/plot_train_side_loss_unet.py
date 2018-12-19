"""
Train Side Loss UNet Example
================================

In this example a UNet with side supervision
and auxiliary loss  implemented

"""

##############################################################################
# Imports needed for this example
import torch
import torch.nn as nn
from inferno.io.box.binary_blobs import get_binary_blob_loaders
from inferno.trainers.basic import Trainer

from inferno.extensions.layers.convolutional import  Conv2D
from inferno.extensions.models.res_unet import _ResBlock as ResBlock
from inferno.extensions.models import ResBlockUNet
from inferno.utils.torch_utils import unwrap
from inferno.utils.python_utils import ensure_dir
import pylab


##############################################################################
# To create a UNet with side loss we create a new nn.Module class
# which has a ResBlockUNet as member.
# The ResBlockUNet is configured such that the results of the
# bottom convolution and all the results of the up-stream
# convolutions are returned as (side)-output.
# a 1x1 convolutions is used to give the side outputs
# the right number of out_channels and UpSampling is
# used to resize all side-outputs to the full resolution
# of the input. These side `side-predictions` are
# returned by our MySideLossUNet.
# Furthermore, all  `side-predictions` are concatenated
# and feed trough another two residual blocks to make
# the final prediction.
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

##############################################################################
# We use a custom loss functions which applied CrossEntropyLoss
# to all side outputs.
# The side outputs are weighted in a quadratic fashion and added up
# into a single value
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



##############################################################################
# Training boilerplate (see :ref:`sphx_glr_auto_examples_trainer.py`)
LOG_DIRECTORY = ensure_dir('log')
SAVE_DIRECTORY = ensure_dir('save')
DATASET_DIRECTORY = ensure_dir('dataset')


USE_CUDA = torch.cuda.is_available()

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
trainer = Trainer(model)
trainer.build_criterion(MySideLoss())
trainer.build_optimizer('Adam')
trainer.validate_every((10, 'epochs'))
#trainer.save_every((10, 'epochs'))
#trainer.save_to_directory(SAVE_DIRECTORY)
trainer.set_max_num_epochs(40)

# Bind loaders
trainer \
    .bind_loader('train', train_loader)\
    .bind_loader('validate', validate_loader)

if USE_CUDA:
    trainer.cuda()

# Go!
trainer.fit()


##############################################################################
# Predict with the trained network
# and visualize the results

# predict:
#trainer.load(best=True)
trainer.bind_loader('train', train_loader)
trainer.bind_loader('validate', validate_loader)
trainer.eval_mode()

if USE_CUDA:
    trainer.cuda()

# look at an example
for img,target in test_loader:
    if USE_CUDA:
        img = img.cuda()

    # softmax on each of the prediction
    preds = trainer.apply_model(img)
    preds = [nn.functional.softmax(pred,dim=1)        for pred in preds]
    preds = [unwrap(pred, as_numpy=True, to_cpu=True) for pred in preds]
    img    = unwrap(img,  as_numpy=True, to_cpu=True)
    target  = unwrap(target, as_numpy=True, to_cpu=True)

    n_plots = len(preds) + 2
    batch_size = preds[0].shape[0]

    for b in range(batch_size):

        fig = pylab.figure()

        ax1 = fig.add_subplot(2,4,1)
        ax1.set_title('image')
        ax1.imshow(img[b,0,...])

        ax2 = fig.add_subplot(2,4,2)
        ax2.set_title('ground truth')
        ax2.imshow(target[b,...])

        for i,pred in enumerate(preds):
            axn = fig.add_subplot(2,4, 3+i)
            axn.imshow(pred[b,1,...])

            if i + 1 < len(preds):
                axn.set_title('side prediction %d'%i)
            else:
                axn.set_title('combined prediction')

        pylab.show()

    break
