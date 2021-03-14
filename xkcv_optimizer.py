import torch

def interface_test(args, dataloader, model):
    # for get_optimizer() function
    assert(hasattr(args, 'optimizer_name')) 
    assert(hasattr(args, 'optimizer_lr'))
    assert(hasattr(args, 'optimizer_momentum'))

def get_instance(model, args) : # Get the optimizer for the 
    if args.optimizer_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=args.optimizer_lr, momentum=args.optimizer_momentum, weight_decay=args.optimizer_weightdecay)
    else :
        args.optimizer_name == 'adam'
        return torch.optim.Adam(model.parameters(), lr=args.optimizer_lr,  weight_decay=args.optimizer_weightdecay)
