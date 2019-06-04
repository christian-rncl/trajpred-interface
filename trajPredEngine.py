from ignite.engine import Engine, Events
from utils import lstToCuda,maskedNLL,maskedMSE,maskedNLLTest
import time
import math
import torch
from ignite.contrib.handlers import ProgressBar
import datetime

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

        self.save_name = None

    def netPred(self, batch):
        raise NotImplementedError

    def saveModel(self, engine):
        currentDT =  datetime.datetime.now()
        torch.save(self.net.state_dict(), "trained_models/{}_{}_{}_{}_{}_{}.tar".format(self.save_name,
        currentDT.hour, currentDT.minute, currentDT.second, currentDT.month, currentDT.year))


    def train_batch(self, engine, batch):
        self.net.train_flag = True
        epoch = engine.state.epoch

        _, _, _, _, _, _, _, fut, op_mask = batch
        fut_pred = self.netPred(batch)
        fut = fut.cuda()
        op_mask = op_mask.cuda()

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
        self.optim.step()

        # Track average train loss:
        self.avg_trn_loss += l.item()
        self.metrics["Avg train loss"] += l.item() / 100.0

        return l.item()

    def eval_batch(self, engine, batch):
        self.net.train_flag = False

        _, _, _, _, _, _, _, fut, op_mask = batch
        fut_pred = self.netPred(batch)
        fut = fut.cuda()
        op_mask = op_mask.cuda()

        # Forward pass
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
        self.trainer.add_event_handler(Events.COMPLETED, self.saveModel)
        # zero out metrics for next epoch


    def start(self):
        self.trainer.run(self.train_loader, max_epochs=self.args["pretrainEpochs"] + self.args["trainEpochs"])
