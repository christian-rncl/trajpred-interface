from trajPredEngine import TrajPredEngine
import torch

class SocialEngine(TrajPredEngine):
    """
    Implementation of abstractEngine for traphic
    TODO:maneuver metrics, too much duplicate code with socialEngine
    """

    def __init__(self, net, optim, train_loader, val_loader, args):
        super().__init__(net, optim, train_loader, val_loader, args)

    def getModelInput(self, batch) :
        hist, _, nbrs, _, mask, lat_enc, long_enc, _, _ = batch
        return hist, nbrs,  mask, lat_enc, long_enc

    def getGT(self, batch):
        _, _, _, _, _, _, _, fut, op_mask = batch
        return fut, op_mask




# from ignite.engine import Engine, Events
# from utils import lstToCuda,maskedNLL,maskedMSE,maskedNLLTest
# import time
# import math
# from abstractEngine import IgniteEngine
# import torch
# from ignite.contrib.handlers import ProgressBar

# class SocialEngine(IgniteEngine):
#     """
#     TODO:maneuver metrics, 
#     """

#     def __init__(self, net, optim, train_loader, val_loader, args):
#         self.net = net
#         self.args = args
#         self.pretrainEpochs = args["pretrainEpochs"]
#         self.trainEpochs = args["trainEpochs"]
#         self.optim = optim
#         self.train_loader = train_loader
#         self.val_loader = val_loader

#         ## training metrics to keep track of, consider making a metrics class
#         # remember to 0 these out
#         self.avg_trn_loss = 0

#         ## validation metrics
#         self.avg_val_loss = 0
#         self.val_batch_count = 1

#         # only if using maneuvers
#         self.avg_lat_acc = 0
#         self.avg_lon_acc = 0

#         self.trainer = None
#         self.evaluator = None

#         self.makeTrainer()

#     def train_batch(self, engine, batch):

#         epoch = engine.state.epoch
#         i = engine.state.iteration

#         prev_val_loss = math.inf

#         if epoch == 0:
#             print('Pre-training with MSE loss')
#         elif epoch == self.pretrainEpochs:
#             print('Training with NLL loss')


#         hist, _, nbrs, _, mask, lat_enc, lon_enc, fut, op_mask = batch

#         # moving data to gpu during training causes a lot of overhead 
#         if self.args['use_cuda']:
#             lstToCuda([hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc, fut, op_mask])

#         # Forward pass
#         if self.args['use_maneuvers']:
#             fut_pred, lat_pred, lon_pred = self.net(hist, nbrs, mask, lat_enc, lon_enc)
#             # Pre-train with MSE loss to speed up training
#             if epoch < self.pretrainEpochs:
#                 l = maskedMSE(fut_pred, fut, op_mask)
#             else:
#             # Train with NLL loss
#                 l = maskedNLL(fut_pred, fut, op_mask) + crossEnt(lat_pred, lat_enc) + crossEnt(lon_pred, lon_enc)
#                 self.avg_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
#                 self.avg_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]

#         else:
#             fut_pred = self.net(hist, nbrs, mask, lat_enc, lon_enc)
#             if self.args['nll_only']:
#                 l = maskedNLL(fut_pred, fut, op_mask)
#             else:
#                 if epoch < self.pretrainEpochs:
#                     l = maskedMSE(fut_pred, fut, op_mask)
#                 else:
#                     l = maskedNLL(fut_pred, fut, op_mask)

#         # Backprop and update weights
#         self.optim.zero_grad()
#         l.backward()
#         a = torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
#         self.optim.step()

#         # Track average train loss:
#         self.avg_trn_loss += l.item()

#         return l.item()

#     def eval_batch(self, engine, batch):
#         net.train_flag = False
#         model.eval()

#         hist, _, nbrs, _, mask, lat_enc, lon_enc, fut, op_mask = batch

#         if self.args['use_cuda']:
#             lstToCuda([hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc, fut, op_mask])


#         # Forward pass
#         if args['use_maneuvers']:
#             if epoch_num < pretrainEpochs:
#                 # During pre-training with MSE loss, validate with MSE for true maneuver class trajectory
#                 net.train_flag = True
#                 fut_pred, _ , _ = net(hist, nbrs, mask, lat_enc, lon_enc)
#                 l = maskedMSE(fut_pred, fut, op_mask)
#             else:
#                 # During training with NLL loss, validate with NLL over multi-modal distribution
#                 fut_pred, lat_pred, lon_pred = net(hist,upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc)
#                 l = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,avg_along_time = True)
#                 avg_val_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
#                 avg_val_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
#         else:
#             fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
#             if args['nll_only']:
#                 l = maskedNLL(fut_pred, fut, op_mask)
#             else:
#                 if epoch_num < pretrainEpochs:
#                     l = maskedMSE(fut_pred, fut, op_mask)
#                 else:
#                     l = maskedNLL(fut_pred, fut, op_mask)

#         self.avg_val_loss += l.item()

#         self.val_batch_count += 1
#         return fut_pred, fut

#     def validate(self, engine):
#         self.evaluator.run(self.val_loader)


#     def zeroMetrics(self, engine):
#         ## train metrics
#         self.avg_trn_loss = 0

#         ## validation metrics
#         self.avg_val_loss = 0
#         self.val_batch_count = 1

#         # only if using maneuvers
#         self.avg_lat_acc = 0
#         self.avg_lon_acc = 0


#     def makeTrainer(self):
#         self.trainer = Engine(self.train_batch)
#         self.evaluator = Engine(self.eval_batch)

#         if self.args['use_maneuvers']:
#             metrics = {}
#         else:
#             metrics = {"Avg train loss": self.avg_trn_loss / 100.0, "Avg val loss": self.avg_val_loss/self.val_batch_count }

#         pbar = ProgressBar(persist=True, postfix=metrics)
#         pbar.attach(self.trainer)

#         ## attach hooks
#         # evaluate after every batch
#         self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.validate)
#         # zero out metrics for next epoch
#         self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.zeroMetrics)


#     def start(self):
#         self.trainer.run(self.train_loader, max_epochs=self.args["pretrainEpochs"] + self.args["trainEpochs"])
