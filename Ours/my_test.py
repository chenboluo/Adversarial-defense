from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable




from resnet import *
import numpy as np
import time

from advertorch.attacks import LinfPGDAttack, CarliniWagnerL2Attack, DDNL2Attack, SinglePixelAttack, LocalSearchAttack, SpatialTransformAttack,L1PGDAttack
import torchvision.transforms.functional as VF
from torchattacks import PGD , AutoAttack, CW,EOTPGD
from advertorch.utils import predict_from_logits
from torchattacks import PGD , AutoAttack, CW,FGSM,OnePixel,Square,APGD,FAB



os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser(description='PyTorch CIFAR MART Defense')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=3.5e-3,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=5.0,
                    help='weight before kl (misclassified examples)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', default='TA4',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

args = parser.parse_args()
cnt = 0
    
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 10, 'pin_memory': True} if use_cuda else {}
torch.backends.cudnn.benchmark = True

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='../data_attack/', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=10)
testset = torchvision.datasets.CIFAR10(root='../data_attack/', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=10)





from torchvision import utils as vutils







class modelx(nn.Module):
    def __init__(self):
        super(modelx,self).__init__()
        self.model = ResNet18().cuda() 
        

     
    def forward(self,imgs):    
        imgs = imgs.cuda()
        pre1 = self.model(imgs)
        return pre1
        

       
model = modelx()
predmodel = torch.load('')  


model.load_state_dict(predmodel)




def main():


    natural_acc = []
    robust_acc = []
    cnt = 0
    logits = [0, 0, 0, 0, 0]
    for nat_data, labels in test_loader:
        nat_data, labels = nat_data.cuda(), labels.cuda()
        start_time = time.time()

        model.eval()

        adversary = LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255,nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
        adv = adversary.perturb(nat_data, labels)
        
        adversary = DDNL2Attack(model, nb_iter=40, gamma=0.05, init_norm=1.0, quantize=True, levels=256, clip_min=0.0,
                            clip_max=1.0, targeted=False, loss_fn=None)
        adv2 = adversary(nat_data, labels)

        attack = CW(model, c=1, kappa=0, steps=500, lr=0.01)
        adv3 = attack(nat_data, labels)
        
        adversary = SpatialTransformAttack(model, 10, clip_min=0.0, clip_max=1.0, max_iterations=10, search_steps=5, targeted=False)
        adv4 = adversary(nat_data, labels)

        
        
        
        data = [nat_data,adv,adv2,adv3,adv4]


        cnt = cnt + labels.shape[0]



        for j in range(5):
            imgs = data[j]
            model.eval()
            
            imgs = imgs.cuda()
            pre1 = model(imgs)
            pred = predict_from_logits( pre1 )
            for i in range(len(pred)):
                 if(  pred[i]==labels[i] ):
                     logits[j] = logits[j]+1

    for i in range(5):
        print(logits[i])
    print(cnt)





if __name__ == '__main__':
    main()

