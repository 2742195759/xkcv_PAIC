import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import glob

#pic_one = str(input("Input first image name\n"))
#pic_two = str(input("Input second image name\n"))


model = models.resnet101(pretrained=True)
model.eval()
layer = model._modules.get('avgpool')
def cos_sim(pic_one, pic_two):

    # Set model to evaluation mode
    scaler = transforms.Scale((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    def get_vector(image_name):
        # 1. Load the image with Pillow library
        img = Image.open(image_name)    # 2. Create a PyTorch Variable with the transformed image
        t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))    # 3. Create a vector of zeros that will hold our feature vector
        #    The 'avgpool' layer has an output size of 512
        my_embedding = torch.zeros(2048)    # 4. Define a function that will copy the output of a layer
        def copy_data(m, i, o):
            my_embedding.copy_(o.squeeze().data)    # 5. Attach that function to our selected layer
        h = layer.register_forward_hook(copy_data)    # 6. Run the model on our transformed image
        model(t_img)    # 7. Detach our copy function from the layer
        h.remove()    # 8. Return the feature vector
        return my_embedding

    pic_one_vector = get_vector(pic_one)
    pic_two_vector = get_vector(pic_two)

    # Using PyTorch Cosine Similarity
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_sim = cos(pic_one_vector.unsqueeze(0),
          pic_two_vector.unsqueeze(0))
    print('\nCosine similarity: {0}\n'.format(cos_sim))

pic = "../cache/input1.jpg"
imgs = glob.glob('../../User_Caption/data/img/*')
for i in range(10):
    print ("=============", imgs[i])
    cos_sim(pic, imgs[i])
