##############################
#   普通驱动函数
#      Main()
##############################

##############################
#
#      XXX Assumption 假设
# 1. 驱动函数中只有numpy 和 pandas 对象, 不包含具体的tensor数据
# 2. 驱动函数通过获取实例来获得model和dataloader
# 3. 驱动函数不使用学习率等，具体的训练过程在model函数中
# 4. 本普通驱动函数不考虑多重训练啥的，只负责简单的训练方法
#
##############################
##############################
#
#      XXX 命名规范
# 1. 类名驼峰，变量函数_分割，_*的函数作为私有函数，hook_*作为钩子函数()
# 2. 变量命名：
#    第一个字段作为类型符号，包含 [np_ df_ n_ str_ tnsr_ b_ m_ s_ f_]
#    第二个字段开始为名词
# 3. 类文档规范
#    class ClassName:
"""@ mutable | overridable       --------------------- 作为关键字栏目
                                 --------------------- 空行
    类描述放在这里               --------------------- 类描述文本
"""
#        def function_name(self):
"""@ 关键字

    函数功能描述

    @ 参数: 类型   描述
    @ 参数: 类型   描述
    @ ret : 返回值类型  描述
"""
#
##############################
##############################
#
#      XXX args对象必须参数
#     
#1. 优化器相关
#   optimizer_name 
#   optimizer_lr 
#   optimizer_momentum
# 
#2. 训练相关
#   device  
#   epochs
#   batchsize 
#   eval_interval
#   loss_interval
#
#3. 经典超参数类  hyper_*
#   例如 : hyper_alpha, hyper_lambda等
#
##############################

import os
import sys
import xkcv_model
import xkcv_dataloader
import xkcv_adaptor
import tqdm
from xklib import space 
if './utils/nlp-metrics' not in sys.path:
    sys.path.append('./utils/nlp-metrics')

def interface_test(model, dataset, args) : 
    assert(isinstance(args, space))

def normal_train(model_name, args, save=None, load=None):
    args.adaptor = xkcv_adaptor.get_instance(args.dataname + '2' + model_name, args)
    dataset = xkcv_dataloader.get_instance(args.dataloader, args) # XXX dataset should return some 
    model = xkcv_model.get_instance(model_name, args, load)
    interface_test(model, dataset, args)
    for epoch in range(args.epochs):
        tot_loss = 0.0
        dataset.shuffle()                         # XXX 每个epoch调用一次，shuffle
        tot = len(dataset)
        steps = dataset.get_batch_num()  # XXX dataset 要在 batch_id 过大时返回空
        for bid in tqdm.tqdm(range(steps)):
            batch = dataset.get_batch(bid)        # XXX dataset.get_batch()  应该返回一个 dict{str: numpy} , 列为对应的名词
            if not batch : 
                print ('[WARN] invalid batch, None')
                continue
            loss  = model.train_step(batch, epoch, bid)       # XXX type is float
            tot_loss += loss
            if ((bid+1) % args.eval_interval == 0): 
                print ('[epoch:{epoch}, step:{bid} / totstep:{tot}] eval = {eval_str}'.format(epoch=epoch, bid=bid, tot=steps, eval_str=model.eval_test(dataset.get_testset())))
            if ((bid+1) % args.loss_interval == 0): 
                print ('[epoch:{epoch}, step:{bid} / totstep:{tot}] loss = {loss_str}'.format(epoch=epoch, bid=bid, tot=steps ,loss_str=str(loss)))
        tot_loss /= tot
        print ('[epoch:{epoch}, step:{bid}] {eval_str}'.format(epoch=epoch, bid=bid, eval_str=model.eval_test(dataset.get_testset())))
        print ('[epoch:{epoch}]  tot loss:{tot_loss}'.format(epoch=epoch, tot_loss=tot_loss))
        print ('[BEST epoch:{epoch}] {eval_str}'.format(epoch=epoch, bid=bid, eval_str=model.best_result()))

        if save :
            xkcv_model.save_xkmodel(model, save)

    return model
