from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchvision import models
from trainObjPose import load_data

from sys import argv
def Axis2Rot(a):

    theta = np.linalg.norm(a)
    M= np.identity(3)
    w = np.asarray([[0, -a[2], a[1]], [a[2], 0, -a[0]] , [- a[1],a[0], 0]])
    if not theta == 0:
        M = M + (np.sin(theta) / theta) * w + ((1 - np.cos(theta)) / pow(theta,2)) * np.dot(w , w)
    return M

if __name__ == '__main__':

    # define path
    #model_path = "models/tool_yellow_v3.dat"
    dataset_path = "dataset/dataset_yellow.npy"

    model_path = "models/tool_yellow_v9.dat"
    #model_path = "models/model.dat"

    pathImage = "./example/tool_yellow_val"
    dataset_path = argv[1]
    saveNamePred = pathImage + "/" + "predictions.txt"
    saveNameLabel = pathImage + "/" + "labels.txt"

    # dataset
    phase = 'val'


    dataloaders = load_data(dataset_path)
    testloader = dataloaders[phase] # select validation dataset

    # load trained model
    model = torch.load(model_path)
    alexnet = models.alexnet(pretrained=True)
    in_feats = alexnet.classifier._modules['6'].in_features
    alexnet.classifier._modules['6'] = nn.Linear(in_feats, 9)
    alexnet.load_state_dict(model)
    alexnet.train(False) # remove dropout etc.

    use_gpu =  torch.cuda.is_available()

    if use_gpu:
        alexnet = alexnet.cuda()


    prediction_all = np.empty((0,9), dtype=float)
    label_all = np.empty((0,9), dtype=float)
    for data in testloader:

        images, labels = data

        if use_gpu:
            inputs = Variable(images.cuda())
        else:
            inputs = Variable(images)

        outputs = alexnet(inputs)
        out = outputs.data.cpu().numpy()

        for i in range(0, out.shape[0]):

            if out.shape[1] == 3:
                print(out[i])
                o = Axis2Rot(out[i])
            else:
                o = out[i]


            r = np.reshape(o, (3,3))
            u, s, vh = np.linalg.svd(r)
            R = np.dot(u, vh)
            print(R)
            prediction_all = np.append(prediction_all, np.reshape(R,(-1,9)), axis=0)



        lab = labels.numpy()


#        label_all = np.append(label_all, lab, axis=0)

 #       print(out)

    np.savetxt(saveNamePred, prediction_all, delimiter= " ", fmt = '%.4f')
 #   np.savetxt(saveNameLabel, label_all, delimiter= " ", fmt = '%.4f %.4f %.4f')