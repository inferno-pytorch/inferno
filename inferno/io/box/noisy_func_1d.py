import torch.utils.data as data
import numpy


class NoisyFunc(data.Dataset):


    def __init__(self, f, size, signal_length, sigma, split, label_transform=None):
        # the f
        self.f = f

        # how many images are in the dataset
        self.size = size

        # blob related members
        self.signal_length             = signal_length

        # noise related members
        self.sigma = sigma

        # which split {'train', 'test', 'validate'}
        self.split              = split

        self.label_transform = label_transform

        # internal
        split_to_seed = dict(train=0, test=1, validate=2)
        self.master_seed  = split_to_seed[self.split]*self.size

        
    def __getitem__(self, index):
        numpy.random.seed(seed=self.master_seed + index)

        signal = numpy.zeros(self.signal_length, dtype=numpy.float32)
        gt = numpy.zeros(self.signal_length, dtype=numpy.int32)
        for i in range(self.signal_length):
            d,l = self.f(i)
            signal[i] = d
            gt[i] = l

        if self.label_transform is not None:
                gt = self.label_transform(gt)

        noisy_signal = signal.copy()

        # add gaussian noise
        noisy_signal += numpy.random.normal(scale=self.sigma,size=self.signal_length)
        
        
        # add singletons for channels
        return noisy_signal[None,...], gt[None,...]

    def __len__(self):
        return self.size


def get_noisy_func_loader(f, size, signal_length, sigma,train_batch_size=1, test_batch_size=1,
                            num_workers=1, label_transform=None):
    
    trainset = NoisyFunc(split='train',     f=f, size=size, signal_length=signal_length, sigma=sigma, label_transform=label_transform)
    testset  = NoisyFunc(split='test',      f=f, size=size, signal_length=signal_length, sigma=sigma, label_transform=label_transform)
    validset = NoisyFunc(split='validate',  f=f, size=size, signal_length=signal_length, sigma=sigma, label_transform=label_transform)


    trainloader = data.DataLoader(trainset, batch_size=train_batch_size,
                                            num_workers=num_workers)

    testloader = data.DataLoader(testset, batch_size=test_batch_size,
                                            num_workers=num_workers)

    validloader = data.DataLoader(validset, batch_size=test_batch_size,
                                            num_workers=num_workers)

    return trainloader, testloader, validloader



