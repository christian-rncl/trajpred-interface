from __future__ import print_function
import torch
from traphic import traphicNet
from social import highwayNet
from utils import ngsimDataset,maskedNLL,maskedMSE,maskedNLLTest
from torch.utils.data import DataLoader
import time
import math
import warnings

# ignite
from ignite.engine import Engine, Events

warnings.filterwarnings("ignore")

## Network Arguments
args = {}
args['dropout_prob'] = 0.5
args['use_cuda'] = False
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 50
args['grid_size'] = (13,3)
args['upp_grid_size'] = (7,3)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['num_lat_classes'] = 3
args['num_lon_classes'] = 2
args['train_flag'] = True
args['use_maneuvers'] = False
args['ours'] = False
args['model_path'] = 'trained_models/m_false/cslstm_b_pretrain2_NGSIM.tar'
args['nll_only'] = True
pretrainEpochs = 6
trainEpochs = 10
batch_size = 128
lr=1e-3


# Initialize network
if args['ours']:
    net = traphicNet(args)
else:
    net = highwayNet(args)
# net.load_state_dict(torch.load('trained_models/m_false/cslstm_b_pretrain2_NGSIM.tar'), strict=False)

# for i,child in enumerate(net.children()):
     # print(child)
     # if i < 10 or i in [12,13]:
     #     for params in child.parameters():
     #         params.requires_grad = False

if args['use_cuda']:
    net = net.cuda()


## Initialize optimizer
# pretrainEpochs = 80
# trainEpochs = 200
# batch_size = 128
optimizer = torch.optim.Adam(net.parameters(),lr=lr)
crossEnt = torch.nn.BCELoss()


## Initialize data loaders
trSet = ngsimDataset('data/TrainSet.mat')
valSet = ngsimDataset('data/ValSet.mat')
trDataloader = DataLoader(trSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=trSet.collate_fn)
valDataloader = DataLoader(valSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=valSet.collate_fn)

## Variables holding train and validation loss values:
train_loss = []
val_loss = []
prev_val_loss = math.inf

avg_tr_loss = 0
avg_tr_time = 0
avg_lat_acc = 0
avg_lon_acc = 0


def lstToCuda(lst):
    for item in lst:
        item.cuda()

def traphic_train(engine, batch):
    st_time = time.time()
    epoch = engine.state.epoch
    i = engine.state.iteration

    if epoch_num == 0:
        print('Pre-training with MSE loss')
    elif epoch_num == pretrainEpochs:
        print('Training with NLL loss')

    hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc, fut, op_mask = data

    # moving data to gpu during training causes a lot of overhead 
    if args['use_cuda']:
        lstToCuda([hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc, fut, op_mask])

    # Forward pass
    if args['use_maneuvers']:
        fut_pred, lat_pred, lon_pred = net(hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc)
        # Pre-train with MSE loss to speed up training
        if epoch < pretrainEpochs:
            l = maskedMSE(fut_pred, fut, op_mask)
        else:
        # Train with NLL loss
            l = maskedNLL(fut_pred, fut, op_mask) + crossEnt(lat_pred, lat_enc) + crossEnt(lon_pred, lon_enc)
            avg_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
            avg_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]

    else:
        fut_pred = net(hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc)
        if args['nll_only']:
            l = maskedNLL(fut_pred, fut, op_mask)
        else:
            if epoch < pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                l = maskedNLL(fut_pred, fut, op_mask)

    # Backprop and update weights
    optimizer.zero_grad()
    l.backward()
    a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
    optimizer.step()

    # Track average train loss and average train time:
    batch_time = time.time()-st_time
    avg_tr_loss += l.item()
    avg_tr_time += batch_time

    if i%100 == 99:
        eta = avg_tr_time/100*(len(trSet)/batch_size-i)
        print("Epoch no:",epoch,"| Epoch progress(%):",format(i/(len(trSet)/batch_size)*100,'0.2f'), "| Avg train loss:",format(avg_tr_loss/100,'0.4f'),"| Acc:",format(avg_lat_acc,'0.4f'),format(avg_lon_acc,'0.4f'), "| Validation loss prev epoch",format(prev_val_loss,'0.4f'), "| ETA(s):",int(eta))
        train_loss.append(avg_tr_loss/100)
        avg_tr_loss = 0
        avg_lat_acc = 0
        avg_lon_acc = 0
        avg_tr_time = 0

    return l.item()






