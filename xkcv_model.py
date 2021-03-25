## python3 代码
import numpy as np
import torch
import os
import pandas as pd
import xkcv_optimizer
from torch import *
import torch.nn.init as init
import model
from nlp_score import score # 评分函数
from xklib import *
import pdb

####################################
# 
#      XXX xkcv_model protocol
#1. compared with nn.Module, the xkcv_model have the abillity to 
#   handle the loss. because a successful model contains 2 parts
#   the model parts(forward parts) and the loss parts
#
#      XXX Loss process protocol (always xkcv)
#1. Loss contains 2 kinds of : supervised and unsupervised
#2. for the supervised learning and inter_module unsupervised loss, the loss is calculated
#   by the father xkcv module, and in the function of the _step_loss() function
#3.    XXX(don't let the nn.module have loss) if you need loss, make it can be assessed by the 
#   driver module xvcvmodule, such as self.for_loss_XXX = mid_variates, the self.mid_variates
#   should only contains the information, not the loss itself
#
#      XXX Eval protocal
#1. if have two different state, use the forward() means  See. Cond_LSTM
# 
#      XXX args protocal
#1. The module should only get the values in __init__()
#2. The xkmodule can get args in the args = space() variates
# 
#      XXX input general protocal
#1. every batch appears in the first dim even the batch is 1, the dim will preserved
#      [[item_1]]
#
####################################



dict_path = './cache/xkmodel_save/'

def get_instance(name, args, path=None):
    model = eval(name)(args)
    if (path != None):
        path = dict_path + path
        if os.path.exists(path) :
            model._load_model(path)
    return model.to(args.device)

def save_xkmodel(model, path):
    if not os.path.exists(dict_path) : 
        os.mkdir(dict_path)
    model._save_model(dict_path+path)

# XXX (Driver Module, contains a lot nn.module and this is a driven model)
#     1. handle loss and train 
#     2. can use to eval and 

class xkcv_model(torch.nn.Module) :
    def __init__(self):
        super(xkcv_model, self).__init__()
        pass

    def _step_loss(self, input_batch):
        raise NotImplementedError()
        pass

    def apply_grad_clip(self):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                torch.nn.utils.clip_grad_norm_(p, self.clip_value, self.clip_type)
                #torch.nn.utils.clip_grad_value_(p, self.clip_value)


    def train_step(self, input_batch, batch=None, bid=None):
        """
            train_step，see it in [](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-optim/)
        """
        #XXX must add the self.train()
        self.train()
        self.optimizer.zero_grad()
        loss = self._step_loss(input_batch, batch, bid)
        if (torch.isnan(loss)):
            raise Exception("Loss become NaN, Exit Abnormal")

        self.apply_grad_clip()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval_test(self, test_dataset): 
        """
            the forward process and the result
            @test_dataset : should be a list of input, input formated see the concrete function
            @return       : should be a tuple of ([result_format], avg_score/avg_metric_str)
        """
        #XXX must add self.eval()
        raise NotImplementedError()
        
    def best_result(self):  
        raise NotImplementedError()

    def _save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def _load_model(self, path):
        self.load_state_dict(torch.load(path))

# XXX (需要注意，句子要加入 <SOS> 和 <EOS> 在句首和句尾
class User_Caption(xkcv_model):
    def __init__(self, args):
        super(User_Caption, self).__init__()
        self.clip_value = args.clip_value
        self.clip_type = args.clip_type
        self.dictionary = args.dictionary
        self.batch_size = args.batchsize
        self.device = torch.device(args.device)
        self.n_user = args.n_user
        self.n_user_dim = args.n_user_dim
        self.n_img_dim = args.n_img_dim   # [int] img feature dim
        self.n_word_dim = args.n_word_dim
        self.n_voc_size = args.n_voc_size
        self.n_cap_len  = args.n_cap_len
        self.n_hidden = args.n_hidden
        self.wei_user = torch.nn.Embedding(self.n_user, self.n_user_dim)
        # construct the link to the lstm
        self.wei_WI = torch.nn.Linear(self.n_img_dim+self.n_user_dim, self.n_word_dim)   # self.wei_WI 见图, image_feat transform matrix
        self.wei_WP = torch.nn.Linear(self.n_hidden, self.n_voc_size)  # self.wei_WP 预测softmax的地方，
        # === submodule ===
        self.cond_lstm = torch.nn.LSTM(self.n_word_dim, self.n_hidden, 1, True, False) # dropout 必须多层LSTM
        self.word_emb = torch.nn.Embedding(self.n_voc_size, self.n_word_dim)
        self.bn = torch.nn.BatchNorm1d(self.n_img_dim)
        self.aesthetic_layer = model.AestheticFeatureLayer()
        # XXX 初始化变量要reset_parameter中添加
        self.reset_parameter()
        self._best = xk_Maximize(['Bleu_4', 'Bleu_3', 'Bleu_2', 'Bleu_1', 'ROUGE_L'], [0.0]*5)

        def compare_cos(v_a, v_b):
            #pdb.set_trace()
            assert (isinstance(v_a, torch.Tensor))
            assert (isinstance(v_b, torch.Tensor))
            print  ('similarity is : ', torch.nn.CosineSimilarity(1)(v_a, v_b))

        self.compare = DelayArgProcessor(2, compare_cos)

        "-------------- frozen the backbone ---------------"
        for name, param in self.aesthetic_layer.named_parameters():
            param.requires_grad = False

        self.optimizer = xkcv_optimizer.get_instance(self, args) # XXX 一定要在 所有parameter之后

    def reset_parameter(self):
        #self.wei_user = init.normal_(self.wei_user, mean=0.0, std=1.0)
        pass

    def loss_function (self, raw_feature, high_feature):
        """ @ mutable
            
            this function is the loss of the high_feature, unsupervised loss. used by the father model
            to calculate loss if needed. if don't need please set the lossfunction to None

            @raw_feature: np.array()  .shape = (self.bs, self.dim)
            @return     : np.array()  .shape = (self.bs, self.fdim)
        """
        # TODO (try different loss function, the basic is to use tag-predict crossentropy like loss)
        ...

    def forward(self, input_batch):
        """
        @同step_loss()
        @return =   train_mode   ( prob, None     )
                    eval_mode    ( None, pred_seq )

            prob : .type=tensor  .shpae=[batch, n_vocab_size] 每个数字表示 n_voc的logits，n_voc是拓展之后的voc size
            pred_seq : [ words ]
        """
        import pdb
#        pdb.set_trace()
        n_batch = input_batch.img_feat.shape[0]

        _0, _1, _2, = self.aesthetic_layer(input_batch.img_feat.to(device=self.device))
        raw_img_feat = _2
        user_embedding = self.wei_user(input_batch.uid.to(dtype=torch.long, device=self.device)) # (n_user_dim,)
        if not self.training:
            #import pdb
            #pdb.set_trace()
            self.compare.process(user_embedding)
        """
            img_feat.shape = [n_batch, n_img_dim + n_user_dim]
        """
        img_feat = torch.cat([raw_img_feat, user_embedding], 1) 
        """
            h0.shape = [1, n_batch, n_hidden]
        """
        img_input = self.wei_WI(img_feat).unsqueeze(0) # FIXME 梯度爆炸了  #shapoe
        h0 = torch.zeros((1, n_batch, self.n_hidden)).to(self.device)
        c0 = torch.full_like(h0, 0).to(self.device)

        captions = []
        hiddens = []
        captions.append(self.n_voc_size-2)
        input_wid = torch.full((n_batch, 1), self.n_voc_size-2).long().to(device=self.device)              # size-3 <SOS> size-1 <EOS>
        output, (h0, c0) = self.cond_lstm(img_input, (h0,c0)) 
        idx = 0
        while (input_wid != self.n_voc_size-1).any() and len(hiddens) < self.n_cap_len:
            lstm_input = torch.transpose(self.word_emb(input_wid.to(device=self.device)), 0, 1) # n_step, n_batch, dim
            output, (h0, c0) = self.cond_lstm(lstm_input, (h0,c0)) 
            hiddens.append(output)
            if self.training:
                input_wid = (input_batch.cap_seq[:,[idx+1]])
            else :
                input_wid = self.wei_WP(output).cpu().numpy().argmax()
                captions.append(input_wid)
                input_wid = torch.full((1,1),input_wid).to(dtype=torch.long)
            idx += 1
        
        hiddens = torch.cat(hiddens, dim=0)
        return (hiddens, captions)
        

    def _step_loss(self, input_batch, epochid, batchid): # TODO 将这个拆分为 forward 和 loss_fn 两个
        """
        输入格式: 
        @ input_batch : 
            type(input_batch) = space @后为关键字
            @ key [tensor] uid      : 0-base  .shape = (n_batch,)   .dtype=long
            @ key [tensor] img_feat : .shape = (n_batch, 3, w, h) .dtype=float32
            @ key [tensor] cap_seq  : .shape = (n_batch, n_cap_len) .dtype=long
            ------------------------------------------------------
            <SOS>  voc_size
            <SOE>  voc_size+1
        """
        n_batch = input_batch.img_feat.shape[0]
        """ {DEBUG} """
        h, _ = self(input_batch)  #前向传播
        softmax = torch.nn.Softmax(dim=-1)
        loss = torch.tensor(0, dtype=torch.float32, device=self.device) # 初始化为0, 否则
        """
            NOTE:(DEBUG/FIXED)
                这里h[i]是第i个输入得到的结果。所以h[i]应该和cap_seq[i+1]比较结果
        """
        import pdb
        #pdb.set_trace()
        tot_num = 0
        for i, ht in enumerate(h[:-1]): # TODO(check) <EOS>怎么考虑,然后Loss计算要考虑后面的吗。
            indexes = (range(n_batch), input_batch.cap_seq[:,i+1])  # 同一个step中，所有的batch的gt对应的prob选取出来。
            masks = (input_batch.cap_seq[:,i] != (self.n_voc_size - 1)).float().to(device=self.device)
            tot_num += masks.sum()
            tmp = -torch.log(softmax((self.wei_WP(ht)))[indexes].squeeze())   # TODO(check) 所有都是加起来吗? , tmp.shape = (n_batch,)
            loss += (tmp * masks).sum()
        loss /= tot_num
        #self.compare.process(h0[0,0,:])
        return loss * 1.0
#TODO (将id->seq变成一个函数)

    def eval_test(self, test_dataset): 
        """
            output the eval information and store it at best result

        @ test_dataset: 
            type(test_datset) = list
            test_dataset = [item0=[
                                    userid=int, 
                                    imagefeat=tensor,
                                    cap_seq=tensor
                                  ]
                           ]
        """
        self.eval()
        BLEU_1 = 0.
        BLEU_2 = 0.
        BLEU_3 = 0.
        BLEU_4 = 0.
        ROUGE_L = 0.
        num_files = 0.
        once = Once(print)
        with torch.no_grad() : 
            for item in test_dataset:
                userid, imagefeat, cap_seq, terms = item
                assert (type(userid) == torch.Tensor and userid.shape == (1,))
                #assert (type(imagefeat) == torch.Tensor and imagefeat.shape == (1, 3, 224, 224) )
                assert (type(cap_seq) == torch.Tensor)
                " ========== 构造 forward 需要的数据 ==========="
                input_batch = space()
                input_batch.uid = userid
                input_batch.img_feat = imagefeat
                input_batch.cap_seq = cap_seq
                assert(not self.training)
                import pdb
                #pdb.set_trace()
                _, output = self(input_batch)
                # TODO (make use of output and gt and calculate the score)
                output_str = " ".join([ self.dictionary[i] for i in output if i < len(self.dictionary.token2id)][1:])
                gt_str = " ".join([ self.dictionary[i] for i in cap_seq.squeeze().numpy().tolist() if i < len(self.dictionary.token2id)][1:])
                print ('pred:', output_str)
                print ('gt  :', gt_str)
                from nlp_score import score
                ref = {1: [gt_str]}
                hypo = {1: [output_str]}
                score_map = score(ref, hypo)
                BLEU_1 += score_map['Bleu_1']
                BLEU_2 += score_map['Bleu_2']
                BLEU_3 += score_map['Bleu_3']
                BLEU_4 += score_map['Bleu_4']
                ROUGE_L += score_map['ROUGE_L']

        tot_len = len(test_dataset)
        self._best.process({'Bleu_1':BLEU_1/tot_len, 'Bleu_2':BLEU_2/tot_len, 'Bleu_3':BLEU_3/tot_len, 'Bleu_4':BLEU_4/tot_len, 'ROUGE_L':ROUGE_L/tot_len})
        #pdb.set_trace()
        return (None , ('\nBleu1:' + str(BLEU_1/len(test_dataset)) + # TODO add the result to this function
                '\nBleu2:' + str(BLEU_2/len(test_dataset)) + 
                '\nBleu3:'+ str(BLEU_3/len(test_dataset)) + 
                '\nBleu4:'+ str(BLEU_4/len(test_dataset)) +
                '\nRouge:'+ str(ROUGE_L/len(test_dataset))
               ))
                
    def best_result(self): 
        # return the best eval value to the caller
        return self._best.getdata()

