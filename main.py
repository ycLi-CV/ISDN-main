import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#--pre_train ./experiment/test/model/model_best.pt --save_results --test_only --chop
def main():
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)

    if checkpoint.ok:
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)

        print('Total params: %.6fM' % (sum(p.numel() for p in _model.parameters())/1000000.0))
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, _model, _loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()
        checkpoint.done()

if __name__ == '__main__':
    main()
#python3 main.py --pre_train ./experiment/X4_8_2/model/model_best.pt --save_results --test_only --chop