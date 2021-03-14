import import_path
from xklib import space, set_seed
from xkcv_train import normal_train

Debug = True

def get_args():
    args = space()
    args.batchsize = 30 
    args.seed = 2019
    args.epochs = 25
    args.device = 'cuda:1'
    args.optimizer_name = 'sgd'
    args.optimizer_lr = 0.005
    args.optimizer_momentum = 0.2
    args.optimizer_weightdecay=0
    args.eval_interval = 20000
    args.loss_interval = 10000
    args.debug = True
    args.save_path = 'UserCaption.pth'
    args.load_path = None #'UserCaption.pth'

    " -------------- field related args----------------- "
    args.n_cap_len = 17
    args.n_filter_freq = 5
    args.n_word_dim = 800
    args.n_img_dim = 10048
    args.n_user_dim = 5
    args.n_hidden = 2000

    " -------------- Model Control Related ------------- "
    args.model_name = 'User_Caption'
    args.dataname = 'AVAPCap'
    args.dataloader = 'TorchDataloader'
    args.datapath = './data/AVA_PCap.json'
    args.pretrained_path = None
    args.imagedir = './data/images/'  #FIXME imgs? images?

    return args

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    model_name = 'User_Caption'
    print ('[MAIN] start train "User_Caption" model')
    model = normal_train(model_name, args, args.save_path, args.load_path)
    print ('[MAIN] end train')
