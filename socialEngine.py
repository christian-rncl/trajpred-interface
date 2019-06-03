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

