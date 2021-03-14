# use the normal multilabel methods to predict the label and 
# check the accuracy. if the accuracy is high then ok

# TODO 0.19, try to find other way to get the true result

from skmultilearn.adapt import MLkNN
from skmultilearn.adapt import MLARAM
from skmultilearn.adapt import MLTSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score
from PIL import Image
import numpy as np
import os
# 开始记录耗时
import datetime

class Timer():
    def start(self, msg=""):
        self.time = datetime.datetime.now()
        self.msg  = msg
    def restart(self, msg=''):
        end_time = datetime.datetime.now()
        time_cost = end_time - self.time
        print('[Timer]')
        print('       ' + self.msg)
        print('       ' + str(time_cost).split('.')[0])

        self.msg = msg
        self.time = datetime.datetime.now()
        
timer = Timer()
# copy from the loader function
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        image = Image.open(path)
        image = image.convert('RGB')
        return image

timer.start("loader model")
print ('[Model Single] RandomForest')
#classifier = MLkNN(k=20)
#classifier = MLARAM()
#classifier = MLTSVM()

#classifier = KNeighborsClassifier(30)  5%
#classifier = RandomForestClassifier(1)  24%
#classifier = SVC(kernel='rbf', probability=True)  #24%
classifier = MLPClassifier(hidden_layer_sizes=(200,100,50,25),max_iter=5000)

dirpath = '../data/ava_dataset/'
imagepath = dirpath + 'images/'
style_dir_path = dirpath + 'style_image_lists/'
train_id = style_dir_path + 'train.jpgl'
train_tag = style_dir_path + 'train.lab'
test_id = style_dir_path + 'test.jpgl'
test_tags = style_dir_path + 'test.multilab'

resnet18 = models.resnet18(pretrained=True)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

image_transforms = transforms.Compose([
    transforms.Resize((600,600)),
#    transforms.RandomSizedCrop(600),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
timer.restart('load trainset and fit')
print ('[Load Trainset]')
if not os.path.exists('../cache/ava_input_np.npy'): #XXX target' order is the same, so not this problem
    data = datasets.ImageFolder('../data/ava_style/train/', image_transforms)
    print (data.class_to_idx)
    train_loader = torch.utils.data.DataLoader(
        data, 
        batch_size=2, shuffle=True,
        num_workers=2, pin_memory=True
    )
    resnet18.eval()
    inputs = []
    targets = []
    i = 0
    print ('[Load]')
    for batchx, batchy in train_loader: # the output size is (batch_size, 1000)
        inputs.append(resnet18(batchx).detach().numpy())
        targets.extend(batchy.numpy())

    input_np = np.concatenate(inputs, axis=0)
    target_np = np.array(targets, dtype=np.int64)
    np.save('../cache/ava_input_np.npy', input_np)
    np.save('../cache/ava_target_np.npy', target_np)

input_np = np.load('../cache/ava_input_np.npy')
target_np = np.load('../cache/ava_target_np.npy')
classifier.fit(input_np, target_np) # XXX single
#classifier.fit(input_np, np.eye(14)[target_np]) # XXX Multi

### after the train process, start the eval and predict

timer.restart('load testset')
print ('[Loading Testset]')
# import pdb
# pdb.set_trace()
test_inputs = []
gts = []
if not os.path.exists('../cache/ava_test_inputs.npy'): #XXX target' order is the same, so not this problem
    for iid, tag_line in zip(open(test_id).readlines(), open(test_tags).readlines()):
        iid, tag_line = iid.strip(), tag_line.strip()
        imagefile = imagepath + iid + '.jpg'
        image = image_transforms(default_loader(imagefile)).unsqueeze(0)
        image_np = resnet18(image).detach().numpy()

        test_inputs.append(image_np)
        gts.append(np.array([int(i) for i in tag_line.split(' ')]))
    test_input_np = np.concatenate(test_inputs, axis=0)
    gts_np = np.array(gts, dtype=np.int64)

    np.save('../cache/ava_test_inputs.npy', test_input_np)
    np.save('../cache/ava_test_gts.npy', gts_np)
else : 
    test_input_np = np.load('../cache/ava_test_inputs.npy')
    gts_np = np.load('../cache/ava_test_gts.npy')


timer.restart('predict time')
print ('[Start Predict]')

#import pdb
#pdb.set_trace()

# metric for the function in multilabel classification function
preds_np = classifier.predict(test_input_np)
if hasattr(preds_np, 'toarray'):
    preds_np = preds_np.toarray()

hamm_metric = 0.0
jacc_metric = 0.0
macc_metric = 0.0
mrec_metric = 0.0
f1 = 0.0
cnt = 0

for gt, pred in zip(gts_np, preds_np):
    # gt_bool, pred_bool = gt.astype(np.bool), pred.astype(np.bool) 
    # hamm_metric = hamm_metric + (gt_bool^pred_bool).sum()
    # jacc_metric = jacc_metric + ((gt_bool & pred_bool).sum() * 1.0 / (gt_bool | pred_bool).sum() if (gt_bool | pred_bool).sum() > 0 else 1)
    # single_acc = ((gt_bool & pred_bool).sum() * 1.0 / pred_bool.sum()) if (pred_bool).sum() > 0 else 1
    # single_rec = ((gt_bool & pred_bool).sum() * 1.0 / gt_bool.sum()) if (gt_bool).sum() > 0 else 1
    # macc_metric = macc_metric + single_acc
    # mrec_metric = mrec_metric + single_rec
    macc_metric += (gt[int(pred)] == 1)

    cnt += 1
    if cnt <= 20: 
        print ("")
        print ("")
        print (gt, pred.astype(np.int32))

#f1 = f1 + 2.0 / (1.0 / single_acc + 1.0 / single_rec)
#print ('[Metric] hamm:        {h}'.format(h=hamm_metric*1.0/(len(gts_np)*14)))
#print ('[Metric] jacc:        {h}'.format(h=jacc_metric*1.0/len(gts_np)))
#print ('[Metric] macc:        {h}'.format(h=macc_metric*1.0/len(gts_np)))
#print ('[Metric] mrec:        {h}'.format(h=mrec_metric*1.0/len(gts_np)))
#print ('[Metric] f1:          {h}'.format(h=f1*1.0/len(gts_np)))
print ('[Metric] size=', len(gts_np))
print ('[Metric] acc = {h}'.format(h = macc_metric*1.0 / len(gts_np)))

timer.restart()
