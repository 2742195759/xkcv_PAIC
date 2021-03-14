import import_path
from xklib import *
from torch.utils import data
import torch
import json
import sys
from tqdm import tqdm
from PIL import Image
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import TfidfModel
import numpy as np

""" 
    Dataset deal with the raw data, and make all the data computable, it don't associate with any models, only associate with dataset formated
    # DataAdaptor is associated with the model. 
    Dataloader is the connector between the models + torch.utils + data_adaptor , always be the orginal torch.utils because we can use the multi worker
"""

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        image = Image.open(path)
        image = image.convert('RGB')
        return image

class AVAPCap(data.Dataset):
    """ AVAPCap dataset
        @dataset
            [ uid_int, cap_seq_numpy, image_numpy]
                -> cap_seq_tensor : torch.long, shape
        @attribute
            self.split
            self.n_user
            self.dictionary
            self.data # 中间变量，使用[ uid_int, caq_seq_numpy, image_filename_str ]
            self.__len__() ： 返回数据总数
    """
    def __init__(self, dataname, datapath, imagedir, split, unk_count, max_len, transform=None):
        self.dataname = dataname
        self.top_k = 10
        assert (self.dataname.lower() == 'AVAPCap'.lower())
        self.datapath = datapath
        self.imagedir = imagedir
        self.transform = transform
        self.split = split
        self.raw_data  = json.load(open(self.datapath))
        self.preprocess(unk_count, max_len)
        self.global_information()
        self.split_data = [i for i in self.raw_data['images'] if i['split'] == self.split ]
        self.data = []
        self.n_cap_len = max_len
        for img in self.split_data:
            for idx,uid in enumerate(img['uids']): 
                self.data.append( [uid, img['sentences'][idx]['tokens'], img['filename']] )
                if self.split == 'val' : break # 如果是val那么就只选择一个caption作为预备caption
        self.data = self.tokenize()
    
    def _user_tf_idf(self, top_K=10):
        """
            input : self.raw_data
            return: user_terms    dict{uid: [word1, word2, ... word_K]}
        """
        " ============== 将相同User聚合成为文档，然后计算每个用户的TD-IDF数值，作为用户的显示特征信息 ================"
        
        user_terms = {}
        docs = {}
        for img in self.raw_data['images']:
            for idx, sent in enumerate(img['sentences']):
                user = img['uids'][idx]
                tokens = [ i for i,p in sent['tokens'] ]
                if user not in docs : docs[user] = []
                docs[user].extend(tokens)

        corpus = [ ds for u, ds in docs.items() ]
        dct = Dictionary(corpus)  # fit dictionary
        corpus = [dct.doc2bow(doc) for doc in corpus]  # convert corpus to BoW format
        model = TfidfModel(corpus)
        for user, doc in docs.items():
            bow = dct.doc2bow(doc)
            res = model[bow]
            res.sort(key=lambda x : x[1], reverse=True)
            res = res[:top_K]
            user_terms[user] = [ dct[a] for a,b in res ]
        return user_terms

    def preprocess(self, filter_freq, max_len):
        """
            input : self.raw_data
                    self.top_k
            update: self.raw_data
                    将其中的长句子按照长度 clip
                    将其中的低频词语变为   unk
        """
        " ================= 计算 单词频率 ==================="

        voc_freq = {}
        for img in self.raw_data['images']:
            for sent in img['sentences']:
                for token, pos in sent['tokens']:
                    token = token.lower()
                    if token not in voc_freq: voc_freq[token] = 0
                    voc_freq[token] += 1
        invalid_cnt = len([ 1 for k,v in voc_freq.items() if v < filter_freq ])
        tot_cnt     = len(voc_freq)
        print ('invalid rate :', invalid_cnt*1.0 / tot_cnt)
        print ('voc_size     :', tot_cnt - invalid_cnt)

        " =============== 替换tokens和raw ==================="
        #import pdb
        #pdb.set_trace()
        for img in self.raw_data['images']:
            for sent in img['sentences']:
                new_tokens = []
                for cnt_len, (token, pos) in enumerate(sent['tokens']):
                    if cnt_len >= max_len-2 : break
                    token = token.lower()
                    assert( token in voc_freq )
                    if voc_freq[token] < filter_freq : new_tokens.append(('unknow', pos))
                    else : new_tokens.append((token, pos))
                sent['tokens'] = new_tokens
                sent['raw'] = " ".join([ token for token, pos in sent['tokens'] ])
        self.user_terms = self._user_tf_idf(self.top_k)

    def global_information(self):
        """
            input: self.raw_data 
            output : self.user2id
                     self.n_user
                     self.dictionary
                     self.n_cap_len
                     self.n_voc_size
        """
        self.dictionary = None
        self.user2id = Hasher()
        all_docs = []
        user_names = []
        images = self.raw_data['images']
        for img in images:
            for sent in img['sentences'] : 
                tokens = [ i.lower() for i,t in sent['tokens'] ]  #XXX 注意需要使用lower()，减少变量个数
                all_docs.append(tokens)
            for uid in img['uids'] : 
                user_names.append(uid)

        self.user2id.feed(user_names)
        self.n_user = self.user2id.size()
        self.dictionary = Dictionary(all_docs)
        self.n_voc_size = len(self.dictionary)

    def tokenize(self):
        """
            input: self.data 
            output:
                将self.data 中的 item[1]变成tensor .shape=(self.n_cap_len)
                item[0] 变为用户编号。
                并且返回
        """
        def tokenize_step(tokens):
            n_voc_size = self.n_voc_size
            word_np = np.ones((self.n_cap_len), dtype=np.long) * (n_voc_size+1)
            wordid = [ self.dictionary.token2id[i] for i,t in tokens ]
            #raw_sent = [ i for i, t in sent['tokens'] ]
            wordid.insert(0, n_voc_size)
            if (len(wordid) >= self.n_cap_len) : 
                print ('[WARN] delete so long caption')
            word_np[0:len(wordid)] = np.array(wordid, dtype=np.long)
            return word_np
        return [ [self.user2id.name2id(a), tokenize_step(b), c] for a,b,c in self.data ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
            @ input : self.data
            @ output: {image: , uid:, cap_seq_np:}
        """
        filename = self.data[idx][2]
        image = self.transform(default_loader(self.imagedir+filename))
        user = self.user2id.id2name(self.data[idx][0])
        sample = {}
        sample['uid']   = self.data[idx][0]
        sample['cap_seq']   = self.data[idx][1]
        sample['img_feat'] = image
        terms = [ self.dictionary.token2id[item] for item in self.user_terms[user] ]
        if len(terms) < self.top_k : 
            terms.extend([self.n_voc_size-1]*(self.top_k-len(terms)))
        assert ( len(terms) == self.top_k )
        sample['terms'] = torch.Tensor(terms).long()
        return sample
