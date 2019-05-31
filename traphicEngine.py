from ignite.engine import Engine, Events
from utils import lstToCuda,maskedNLL,maskedMSE,maskedNLLTest
import time
import math
from IgniteEngine import IgniteEngine
import torch
from ignite.contrib.handlers import ProgressBar

class TraphicEngine(IgniteEngine):
    """
    TODO:maneuver metrics, 
    """

    def __init__(self, net, optim, args):
        self.net = net
        self.args = args
        self.pretrainEpochs = args["pretrainEpochs"]
        self.trainEpochs = args["trainEpochs"]
        self.optim = optim

        # metrics to keep track of, consider making a metrics class
        # remember to 0 these out
        self.avg_lat_acc = 0
        self.avg_lon_acc = 0
        self.avg_trn_loss = 0

    def train_batch(self, engine, batch):

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
                self.avg_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
                self.avg_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]

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
        # self.avg_tr_loss += l.item()
        # avg_tr_time += batch_time

        # if i%100 == 99:
        #     # eta = avg_tr_time/100*(len(trSet)/batch_size-i)
        #     print("Epoch no:",epoch_num+1,"| Epoch progress(%):",format(i/(len(trSet)/batch_size)*100,'0.2f'), "| Avg train loss:",format(avg_tr_loss/100,'0.4f'),"| Acc:",format(avg_lat_acc,'0.4f'),format(avg_lon_acc,'0.4f'), "| Validation loss prev epoch",format(prev_val_loss,'0.4f'), "| ETA(s):",int(eta))
        #     train_loss.append(avg_tr_loss/100)

        return l.item()

    # def eval_batch(self, batch):
    #     net.train_flag = False

    #     hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc, fut, op_mask = data

    #     if self.args['use_cuda']:
    #         lstToCuda([hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc, fut, op_mask])

    #     model.eval()

    #     # Forward pass
    #     if args['use_maneuvers']:
    #         if epoch_num < pretrainEpochs:
    #             # During pre-training with MSE loss, validate with MSE for true maneuver class trajectory
    #             net.train_flag = True
    #             fut_pred, _ , _ = net(hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc)
    #             l = maskedMSE(fut_pred, fut, op_mask)
    #         else:
    #             # During training with NLL loss, validate with NLL over multi-modal distribution
    #             fut_pred, lat_pred, lon_pred = net(hist,upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc)
    #             l = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,avg_along_time = True)
    #             avg_val_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
    #             avg_val_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
    #     else:
    #         fut_pred = net(hist,upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc)
    #         if args['nll_only']:
    #             l = maskedNLL(fut_pred, fut, op_mask)
    #         else:
    #             if epoch_num < pretrainEpochs:
    #                 l = maskedMSE(fut_pred, fut, op_mask)
    #             else:
    #                 l = maskedNLL(fut_pred, fut, op_mask)

    #     avg_val_loss += l.item()
    #     val_batch_count += 1

    #     print(avg_val_loss/val_batch_count)

    #     # Print validation loss and update display variables
    #     print('Validation loss :',format(avg_val_loss/val_batch_count,'0.4f'),"| Val Acc:",format(avg_val_lat_acc/val_batch_count*100,'0.4f'),format(avg_val_lon_acc/val_batch_count*100,'0.4f'))
    #     val_loss.append(avg_val_loss/val_batch_count)
    #     prev_val_loss = avg_val_loss/val_batch_count

    def getTrainer(self):
        trainer = Engine(self.train_batch)
        metrics = {"avg_train_loss": self.avg_trn_loss, "avg_val_acc"}
        pbar = ProgressBar(persist=True, postfix=metrics)
        pbar.attach(trainer)
        return trainer


