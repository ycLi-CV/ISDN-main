import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
net = model.Model(args, checkpoint)

x = torch.randn((1,3,24,30)).to('cuda:0')
torch.onnx.export(net, x,'test.onnx')
