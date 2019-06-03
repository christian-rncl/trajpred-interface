import torch
from traphic import traphicNet
from social import highwayNet
from utils import ngsimDataset
from torch.utils.data import DataLoader
import warnings
import math


from traphicEngine import TraphicEngine
from socialEngine import SocialEngine

# ignite

warnings.filterwarnings("ignore")

## Network Arguments
args = {}
args['dropout_prob'] = 0.5
args['use_cuda'] = True
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
args['ours'] = True
args['model_path'] = 'trained_models/m_false/cslstm_b_pretrain2_NGSIM.tar'
args['nll_only'] = True
args["pretrainEpochs"] = 6
args["trainEpochs"] = 10
args["bs"] = 128
batch_size = 128
lr=1e-3
verbose= True


# Initialize network
if args['ours']:
    net = traphicNet(args)
else:
    net = highwayNet(args)

# net.load_state_dict(torch.load('trained_models/m_false/cslstm_b_pretrain2_NGSIM.tar'), strict=False)

if args['use_cuda']:
    print("Using cuda")
    net = net.cuda()
    # net = net.to("cuda")

## Initialize optimizer
optim = torch.optim.Adam(net.parameters(),lr=lr)
crossEnt = torch.nn.BCELoss()

if verbose:
    print("*" * 3, "Using model: ", net)
    print("*" * 3, "Optim: ", optim)
    print("*" * 3, "Creating dataset and dataloaders...")

## Initialize data loaders
trSet = ngsimDataset('data/TrainSetTRAF.mat')
valSet = ngsimDataset('data/ValSetTRAF.mat')
trDataloader = DataLoader(trSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=trSet.collate_fn)
valDataloader = DataLoader(valSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=valSet.collate_fn)

## Variables holding train and validation loss values:
train_loss = []
val_loss = []
prev_val_loss = math.inf

if verbose:
    print("starting training...")

if args['ours']:
    if verbose:
        print("Training TRAPHIC")
    traphic = TraphicEngine(net, optim, trDataloader, valDataloader, args)
    traphic.start()
else:
    if verbose:
        print("Training conv social pooling")
    social = SocialEngine(net, optim, trDataloader, valDataloader, args)
    social.start()
