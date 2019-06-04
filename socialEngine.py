from trajPredEngine import TrajPredEngine
import torch

class SocialEngine(TrajPredEngine):
    """
    Implementation of abstractEngine for traphic
    TODO:maneuver metrics, too much duplicate code with socialEngine
    """

    def __init__(self, net, optim, train_loader, val_loader, args):
        super().__init__(net, optim, train_loader, val_loader, args)
        self.save_name = "social"

    def netPred(self, batch):
        hist, _, nbrs, _, mask, lat_enc, lon_enc, _, _ = batch

        if self.args['use_cuda']:
            hist = hist.cuda()
            nbrs = nbrs.cuda()
            mask = mask.cuda()
            lat_enc = lat_enc.cuda()
            lon_enc = lon_enc.cuda()

        fut_pred  = self.net(hist, nbrs, mask, lat_enc, lon_enc)
        return fut_pred


