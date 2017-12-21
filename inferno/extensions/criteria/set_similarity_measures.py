import torch.nn as nn
from ...utils.torch_utils import flatten_samples
from torch.autograd import Variable


class SorensenDiceLoss(nn.Module):
    """
    Computes a loss scalar, which when minimized maximizes the Sorensen-Dice similarity
    between the input and the target. For both inputs and targets it must be the case that
    `input_or_target.size(1) = num_channels`.
    """
    def __init__(self, weight=None, channelwise=True, eps=1e-6):
        """
        Parameters
        ----------
        weight : torch.FloatTensor or torch.cuda.FloatTensor
            Class weights. Applies only if `channelwise = True`.
        channelwise : bool
            Whether to apply the loss channelwise and sum the results (True)
            or to apply it on all channels jointly (False).
        """
        super(SorensenDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.channelwise = channelwise
        self.eps = eps

    def forward(self, input, target):
        if not self.channelwise:
            numerator = (input * target).sum()
            denominator = (input * input).sum() + (target * target).sum()
            loss = -2. * (numerator / denominator.clamp(min=self.eps))
        else:
            # TODO This should be compatible with Pytorch 0.2, but check
            # Flatten input and target to have the shape (C, N),
            # where N is the number of samples
            input = flatten_samples(input)
            target = flatten_samples(target)
            # Compute numerator and denominator (by summing over samples and
            # leaving the channels intact)
            numerator = (input * target).sum(-1)
            denominator = (input * input).sum(-1) + (target * target).sum(-1)
            channelwise_loss = -2 * (numerator / denominator.clamp(min=self.eps))
            if self.weight is not None:
                # With pytorch < 0.2, channelwise_loss.size = (C, 1).
                if channelwise_loss.dim() == 2:
                    channelwise_loss = channelwise_loss.squeeze(1)
                # Wrap weights in a variable
                weight = Variable(self.weight, requires_grad=False)
                # Apply weight
                channelwise_loss = weight * channelwise_loss
            # Sum over the channels to compute the total loss
            loss = channelwise_loss.sum()
        return loss
        
class Generalized_Dice_Loss(nn.Module):
    '''Taken from "Generalised Dice overlap as a deep learning loss
    function for highly unbalanced segmentations" by Sudre et al'''
    def __init__(self,eps=1e-6):
        self.eps = eps
        super(Generalized_Dice_Loss,self).__init__()

    def forward(self,input,target):
        '''input and target are respectively a tensor of the shape (N,*) with the batch_size N
        the output is the sum over the mean of each batch dimension'''
        batch_size = input.size(0)
        input = input.view(batch_size,-1)
        target = target.view(batch_size,-1)
        num_elements = input.size(1)

        numerator_1 = (input * target).sum(dim=1)
        denominator_1 = input.sum(dim=1) + target.sum(dim=1)
        weight_1 = 1 / ((input * input).sum(dim=1)).clamp(min=self.eps)
        first_addend = -weight_1 * (numerator_1 / denominator_1.clamp(min=self.eps))

        numerator_2 = ((1-input) * (1-target)).sum(dim=1)
        denominator_2 = 2 * num_elements - input.sum(dim=1) - target.sum(dim=1)
        weight_2 = 1 / ((input * input).sum(dim=1)).clamp(min=self.eps)
        second_addend = -weight_2 * (numerator_2 / denominator_2.clamp(min=self.eps))
        
        losses = first_addend + second_addend
        loss = 4*(losses.sum()/batch_size)
        return loss


class Generalized_squared_Dice_Loss(nn.Module):
    '''Taken from "Generalised Dice overlap as a deep learning loss
    function for highly unbalanced segmentations" by Sudre et al
    and adjusted to the original use in "V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation" by Milletari et al'''

    def __init__(self,eps=1e-6):
        self.eps = eps
        super(Generalized_squared_Dice_Loss, self).__init__()

    def forward(self, input, target):
        '''input and target are respectively a tensor of the shape (N,*) with the batch_size N
        the output is the sum over the mean of each batch dimension'''
        batch_size = input.size(0)
        input = input.view(batch_size, -1)
        target = target.view(batch_size, -1)
        num_elements = input.size(1)

        numerator_1 = (input * target).sum(dim=1)
        denominator_1 = (input*input).sum(dim=1) + (target*target).sum(dim=1)
        weight_1 = 1 / ((input * input).sum(dim=1)).clamp(min=self.eps)
        first_addend = -weight_1 * (numerator_1 / denominator_1.clamp(min=self.eps))

        numerator_2 = ((1 - (input*input)) * (1 - (target*target))).sum(dim=1)
        denominator_2 = 2 * num_elements - (input * input).sum(dim=1) - (target * target).sum(dim=1)
        weight_2 = 1 / ((input * input).sum(dim=1)).clamp(min=self.eps)
        second_addend = -weight_2 * (numerator_2 / denominator_2.clamp(min=self.eps))

        losses = first_addend + second_addend
        loss = 4*(losses.sum() / batch_size)
        return loss

class Squared_Dice_Loss(nn.Module):
    '''Taken from "Generalised Dice overlap as a deep learning loss
    function for highly unbalanced segmentations" by Sudre et al
    and adjusted to the original use in "V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation" by Milletari et al'''

    def __init__(self,eps=1e-6):
        self.eps = eps
        super(Squared_Dice_Loss, self).__init__()

    def forward(self, input, target):
        '''input and target are respectively a tensor of the shape (N,*) with the batch_size N
        the output is the mean over the loss of each batch dimension'''
        batch_size = input.size(0)
        input = input.view(batch_size, -1)
        target = target.view(batch_size, -1)
        num_elements = input.size(1)

        numerator_1 = (input * target).sum(dim=1)
        denominator_1 = (input*input).sum(dim=1) + (target*target).sum(dim=1)
        first_addend = -(numerator_1 / denominator_1.clamp(min=self.eps))

        numerator_2 = ((1 - (input*input)) * (1 - (target*target))).sum(dim=1)
        denominator_2 = 2 * num_elements - (input * input).sum(dim=1) - (target * target).sum(dim=1)
        second_addend = -(numerator_2 / denominator_2.clamp(min=self.eps))

        losses = first_addend + second_addend
        loss = 2*(losses.sum() / batch_size)
        return loss


class Dice_Loss(nn.Module):
    '''Taken from "Generalised Dice overlap as a deep learning loss
    function for highly unbalanced segmentations" by Sudre et al
    and adjusted to the original use in "V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation" by Milletari et al'''

    def __init__(self, eps=1e-6):
        self.eps = eps
        super(Dice_Loss, self).__init__()

    def forward(self, input, target):
        '''input and target are respectively a tensor of the shape (N,*) with the batch_size N
        the output is the mean over the loss of each batch dimension'''
        batch_size = input.size(0)
        input = input.view(batch_size, -1)
        target = target.view(batch_size, -1)
        num_elements = input.size(1)

        numerator_1 = (input * target).sum(dim=1)
        denominator_1 = input.sum(dim=1) + target.sum(dim=1)
        first_addend = -(numerator_1 / denominator_1.clamp(min=self.eps))

        numerator_2 = ((1 - input) * (1 - target)).sum(dim=1)
        denominator_2 = 2 * num_elements - input.sum(dim=1) - target.sum(dim=1)
        second_addend = -(numerator_2 / denominator_2.clamp(min=self.eps))

        losses = first_addend + second_addend
        loss = 2 * (losses.sum() / batch_size)
        return loss

class Original_Dice_Loss(nn.Module):
    '''Taken from "Generalised Dice overlap as a deep learning loss
    function for highly unbalanced segmentations" by Sudre et al.
    Now it is weighted such that the second addend has very little influence when there are ones'''

    def __init__(self, eps=1e-6):
        self.eps = eps
        super(Original_Dice_Loss, self).__init__()

    def forward(self, input, target):
        '''input and target are respectively a tensor of the shape (N,*) with the batch_size N
        the output is the mean over the loss of each batch dimension'''
        batch_size = input.size(0)
        input = input.view(batch_size, -1)
        target = target.view(batch_size, -1)
        num_elements = input.size(1)

        numerator_1 = (input * target).sum(dim=1)
        denominator_1 = (input * input).sum(dim=1) + (target * target).sum(dim=1)
        weight_1 = (1-1/(1+target.sum(dim=1)))
        first_addend = -weight_1 * (numerator_1 / denominator_1.clamp(min=self.eps))
        
        numerator_2 = ((1 - (input * input)) * (1 - (target * target))).sum(dim=1)
        denominator_2 = 2 * num_elements - (input * input).sum(dim=1) - (target * target).sum(dim=1)
        weight_2 = 1 / (1 + target.sum(dim=1))
        second_addend = -weight_2 * (numerator_2 / denominator_2.clamp(min=self.eps))

        losses = 2*(first_addend + second_addend)
        loss = losses.sum() / batch_size
        return loss



class Custom_Dice_Loss(nn.Module):
    '''Taken from "Generalised Dice overlap as a deep learning loss
    function for highly unbalanced segmentations" by Sudre et al.
    Now it is weighted such that the second addend has very little influence when there are ones'''

    def __init__(self, eps=1e-6):
        self.eps = eps
        super(Custom_Dice_Loss, self).__init__()

    def forward(self, input, target):
        '''input and target are respectively a tensor of the shape (N,*) with the batch_size N
        the output is the mean over the loss of each batch dimension'''
        batch_size = input.size(0)
        input = input.view(batch_size, -1)
        target = target.view(batch_size, -1)
        num_elements = input.size(1)

        numerator_1 = (input * target).sum(dim=1)
        denominator_1 = (input * input).sum(dim=1) + (target * target).sum(dim=1)
        weight_1 = input.sum(dim=1)/(input.sum(dim=1)+target.sum(dim=1)).clamp(min=self.eps)
        first_addend = -weight_1 * (numerator_1 / denominator_1.clamp(min=self.eps))

        numerator_2 = ((1 - (input * input)) * (1 - (target * target))).sum(dim=1)
        denominator_2 = 2 * num_elements - (input * input).sum(dim=1) - (target * target).sum(dim=1)
        weight_2 = target.sum(dim=1) / (input.sum(dim=1) + target.sum(dim=1)).clamp(min=self.eps)
        second_addend = -weight_2 * (numerator_2 / denominator_2.clamp(min=self.eps))

        losses = 4*(first_addend + second_addend)
        loss = losses.sum() / batch_size
        return loss

        
class TverskyLoss(nn.Module):
    """
    Computes a loss scalar according to Salehi et al., which generalizes the Dice loss.
    It has to parameters, alpha and beta, which weight the False Positives and False Negatives, respectively. 
    For alpha = beta = 0.5 TverslyLoss reduces to Dice Loss.
    In Salehis paper beta = 0.7, alpha = 1 - beta = 0.3 are optimal for very unbalanced data.
    """
    def __init__(self, alpha = 0.3, beta = 0.7, eps=1e-6):
        """
        Parameters
        ----------
        alpha: weight for the FPs
        beta:  weight for the FNs
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        
    def forward(self, input, target):
        '''input and target are respectively a tensor of the shape (N,*) with the batch_size N
        the output is the mean over the loss of each batch dimension'''
        
        batch_size = input.size(0)
        input = input.view(batch_size, -1)
        target = target.view(batch_size, -1)
        
        numerator = (input*target).sum(dim=1)
        denominator = (input*target).sum(dim=1) + self.alpha*((1.-target)*input).sum(dim=1) + self.beta*((1.-input)*target).sum(dim=1)
        
        losses = -numerator/denominator.clamp(min=self.eps)
        loss = losses.sum() / batch_size
        return loss
