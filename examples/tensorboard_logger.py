# general imports
import multiprocessing
import os
import numpy

# torch imports
import torch
from torch import nn 
import torch.utils.data as data
from torchvision import datasets

# inferno imports
from inferno.trainers.basic import Trainer


# access logger from any file
tb_logger = Trainer.tensorboard_summary_writer()


class FlatMNist(data.Dataset):

    def __init__(self):
        super().__init__()
        self.mnist = datasets.MNIST(root='.', download=True)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, i):
        img,l = self.mnist[i]
        one_hot = torch.zeros(10) 
        one_hot[l] = 1
        img = numpy.array(img).astype('float32') /255.0
        #img -= 0.485
        #img /= 0.229
        flat_mnist = img.reshape([784])
        return flat_mnist,one_hot, flat_mnist,l


class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        #self.mse = nn.MSELoss()
        self.rec_loss =  nn.BCELoss(reduction='sum')
        
    def forward(self, output, targets):
        rec, mu, logvar = output
        y_rec,y_labels = targets 

        as_img  = y_rec.view([-1, 1, 28, 28])
        as_img  = as_img.repeat([1,3,1,1])


        tb_logger.add_embedding(mu, metadata=y_labels, label_img=as_img)
   

        rec_loss = self.rec_loss(rec, y_rec)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        scaled_kld =  0.001*kld
        total = rec_loss + scaled_kld
        tb_logger.add_scalars('loss', {
            'rec_loss':rec_loss, 
            'kld':kld,
            'scaled_kld':scaled_kld,
            'total':total
        })


        return total

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784+10, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)
        self.relu = nn.ReLU()

    def encode(self, x, y):
        x = torch.cat([x,y], dim=1)
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, y):
        #z = torch.cat([z,y], dim=1)
        h3 = self.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x.view(-1, 784),y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar


# Fill these in:
out_dir = 'somedir'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

ds = FlatMNist()
train_loader = data.DataLoader(ds, batch_size=3000, 
    num_workers=multiprocessing.cpu_count())
model = VAE()
trainer = Trainer(model)
trainer.setup_tensorboard_summary_writer(
    log_directory=out_dir,
    add_scalars_every=(1, 'iteration'),
    add_embedding_every=(1, 'epoch')
)
trainer.cuda()
trainer.save_to_directory(out_dir) 
trainer.build_criterion(MyLoss()) 
trainer.build_optimizer('Adam',lr=0.01) 
trainer.save_every((1, 'epochs')) 
trainer.save_to_directory(out_dir) 
trainer.set_max_num_epochs(100000)      

# bind callbacks
trainer.bind_loader('train', train_loader, num_inputs=2, num_targets=2) 
trainer.fit()
