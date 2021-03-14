import import_path
from xklib import *

def get_instance(name, args):
    adaptor = eval(name)(args)
    return adaptor

class Dataset2ModelAdaptor(object):
    def __init__(self, args):
        self.args = args
        pass
    def set_args(self, trainset, testset):
        pass
    def train_batch(self, trainbatch):
        pass
    def test_batch(self, testbatch):
        pass

class AVAPCap2User_Caption(Dataset2ModelAdaptor):
    def __init__(self, args):
        super(AVAPCap2User_Caption, self).__init__(args)
        pass

    def set_args(self, dataset, testset):
        print ("=========== Save Args=============")
        print ('user num:', dataset.n_user)
        print ('n_voc_size:', dataset.n_voc_size)
        print ('test num:', len(dataset))
        print ('train num:', len(testset))
        self.args.n_user = dataset.n_user
        self.args.n_voc_size = dataset.n_voc_size + 2
        self.args.dictionary = dataset.dictionary
        
    def train_batch(self, trainbatch):
        res = space()
        res.uid = trainbatch['uid']
        res.img_feat = trainbatch['img_feat']
        res.cap_seq = trainbatch['cap_seq']
        res.terms = trainbatch['terms']
        return res

    def test_batch(self, testbatch):
        test = []
        for item in testbatch:
            test.append([item['uid'], item['img_feat'], item['cap_seq'], item['terms']])
        return test

