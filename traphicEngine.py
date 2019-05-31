from ignite.engine import Engine, Events
from utils import lstToCuda,maskedNLL,maskedMSE,maskedNLLTest
import time
import math
from IgniteEngine import IgniteEngine
import torch

class TraphicEngine(IgniteEngine):

    def __init__(self, net, optim, args):
        self.net = net
        self.args = args
        self.pretrainEpochs = args["pretrainEpochs"]
        self.trainEpochs = args["trainEpochs"]
        self.optim = optim

    def train_batch(self, engine, batch):
        avg_tr_loss = 0
        avg_tr_time = 0
        avg_lat_acc = 0
        avg_lon_acc = 0

        st_time = time.time()
        epoch = engine.state.epoch
        i = engine.state.iteration

        prev_val_loss = math.inf

        if epoch == 0:
            print('Pre-training with MSE loss')
        elif epoch == self.pretrainEpochs:
            print('Training with NLL loss')

        hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc, fut, op_mask = batch

        # moving data to gpu during training causes a lot of overhead 
        if self.args['use_cuda']:
            lstToCuda([hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc, fut, op_mask])

        # Forward pass
        if self.args['use_maneuvers']:
            fut_pred, lat_pred, lon_pred = self.net(hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc)
            # Pre-train with MSE loss to speed up training
            if epoch < self.pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
            # Train with NLL loss
                l = maskedNLL(fut_pred, fut, op_mask) + crossEnt(lat_pred, lat_enc) + crossEnt(lon_pred, lon_enc)
                avg_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
                avg_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]

        else:
            fut_pred = self.net(hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc)
            if self.args['nll_only']:
                l = maskedNLL(fut_pred, fut, op_mask)
            else:
                if epoch < self.pretrainEpochs:
                    l = maskedMSE(fut_pred, fut, op_mask)
                else:
                    l = maskedNLL(fut_pred, fut, op_mask)

        # Backprop and update weights
        self.optim.zero_grad()
        l.backward()
        a = torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
        self.optim.step()

        # Track average train loss and average train time:
        batch_time = time.time()-st_time
        avg_tr_loss += l.item()
        avg_tr_time += batch_time

        if i%100 == 99:
            # eta = avg_tr_time/100*(len(trSet)/batch_size-i)
            print("Epoch no:",epoch, "| Avg train loss:",format(avg_tr_loss/100,'0.4f'),"| Acc:",format(avg_lat_acc,'0.4f'),format(avg_lon_acc,'0.4f'), "| Validation loss prev epoch",format(prev_val_loss,'0.4f'), "| ETA(s):",int(eta))
            # train_loss.append(avg_tr_loss/100)

        return l.item()

    def getTrainer(self):
        return Engine(self.train_batch)


