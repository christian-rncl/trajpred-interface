from ignite.engine import Engine, Events
from utils import lstToCuda,maskedNLL,maskedMSE,maskedNLLTest
import time
import math
import torch
from ignite.contrib.handlers import ProgressBar

class TrajPredEngine:

    def __init__(self, net, optim, train_loader, val_loader, args):
        self.net = net
        self.args = args
        self.pretrainEpochs = args["pretrainEpochs"]
        self.trainEpochs = args["trainEpochs"]
        self.optim = optim
        self.train_loader = train_loader
        self.val_loader = val_loader

        ## training metrics to keep track of, consider making a metrics class
        # remember to 0 these out
        self.avg_trn_loss = 0

        self.metrics = {"Avg train loss": 0, "Avg val loss": 0 }
        ## validation metrics
        self.avg_val_loss = 0
        self.val_batch_count = 1

        # only if using maneuvers
        self.avg_lat_acc = 0
        self.avg_lon_acc = 0

        self.trainer = None
        self.evaluator = None

        self.makeTrainer()

    def getModelInput(self, batch) :
        raise NotImplementedError

    def getGT(self, batch):
        raise NotImplementedError

    def train_batch(self, engine, batch):

        epoch = engine.state.epoch
        i = engine.state.iteration

        fut, op_mask = self.getGT(batch)

        if epoch == 0:
            print('Pre-training with MSE loss')
        elif epoch == self.pretrainEpochs:
            print('Training with NLL loss')

        # moving data to gpu during training causes a lot of overhead 
        if self.args['use_cuda']:
            lstToCuda(self.getModelInput(batch))

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
            fut_pred = self.net(*self.getModelInput(batch))

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

        # Track average train loss:
        self.avg_trn_loss += l.item()
        self.metrics["Avg train loss"] += l.item() / 100.0

        return l.item()

    def eval_batch(self, engine, batch):
        self.net.train_flag = False
        self.net.eval()

        # hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc, fut, op_mask = batch
        fut, op_mask = self.getGT(batch)

        if self.args['use_cuda']:
            lstToCuda(self.getModelInput(batch))


        # Forward pass
        if self.args['use_maneuvers']:
            if epoch_num < pretrainEpochs:
                # During pre-training with MSE loss, validate with MSE for true maneuver class trajectory
                self.net.train_flag = True
                fut_pred, _ , _ = self.net(hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc)
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                # During training with NLL loss, validate with NLL over multi-modal distribution
                fut_pred, lat_pred, lon_pred = net(hist,upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc)
                l = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,avg_along_time = True)
                avg_val_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
                avg_val_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
        else:
            fut_pred = self.net(*self.getModelInput(batch))
            if self.args['nll_only']:
                l = maskedNLL(fut_pred, fut, op_mask)
            else:
                if epoch_num < pretrainEpochs:
                    l = maskedMSE(fut_pred, fut, op_mask)
                else:
                    l = maskedNLL(fut_pred, fut, op_mask)


        self.avg_val_loss += l.item()
        self.metrics["Avg val loss"] += l.item()/ self.val_batch_count

        self.val_batch_count += 1
        return fut_pred, fut

    def validate(self, engine):
        self.evaluator.run(self.val_loader)


    def zeroMetrics(self, engine):
        self.val_batch_count = 1

        self.metrics["Avg val loss"] = 0 
        self.metrics["Avg train loss"] = 0


    def zeroTrainLoss(self, engine):
        self.metrics["Avg train loss"] = 0

    def zeroValLoss(self, engine):
        self.metrics["Avg val loss"] = 0

    def makeTrainer(self):
        self.trainer = Engine(self.train_batch)
        self.evaluator = Engine(self.eval_batch)

        pbar = ProgressBar(persist=True, postfix=self.metrics)
        pbar.attach(self.trainer)

        ## attach hooks
        # evaluate after every batch
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.validate)
        self.trainer.add_event_handler(Events.ITERATION_COMPLETED, self.zeroMetrics)
        # zero out metrics for next epoch


    def start(self):
        self.trainer.run(self.train_loader, max_epochs=self.args["pretrainEpochs"] + self.args["trainEpochs"])
