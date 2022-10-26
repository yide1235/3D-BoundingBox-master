from torch_lib.Dataset import *
from torch_lib.Model import Model, OrientationLoss


import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg
from torch.utils import data
from torch.utils.data.dataset import random_split
import matplotlib.pyplot as plt

import os

def main():

    # hyper parameters
    epochs = 100
    batch_size = 64
    alpha = 0.6
    w = 0.4

    print("Loading all detected objects in dataset...")

    train_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/training'
    dataset = Dataset(train_path)

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset_split, test_dataset_split = torch.utils.data.random_split(dataset, [train_size, test_size])



    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 6}

    generator = data.DataLoader(train_dataset_split, **params)

    generator_valid=data.DataLoader(test_dataset_split, **params)




    my_vgg = vgg.vgg19_bn(pretrained=True)
    model = Model(features=my_vgg.features).cuda()
    opt_SGD = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    conf_loss_func = nn.CrossEntropyLoss().cuda()
    dim_loss_func = nn.MSELoss().cuda()
    orient_loss_func = OrientationLoss




    # load any previous weights
    model_path = os.path.abspath(os.path.dirname(__file__)) + '/weights/'
    latest_model = None
    first_epoch = 0
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    else:
        try:
            latest_model = [x for x in sorted(os.listdir(model_path)) if x.endswith('.pkl')][-1]
        except:
            pass


    if latest_model is not None:
        checkpoint = torch.load(model_path + latest_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt_SGD.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print('Found previous checkpoint: %s at epoch %s'%(latest_model, first_epoch))
        print('Resuming training....')



    training_loss_total=[]
    validation_loss_total=[]



    total_num_batches = int(len(dataset) / batch_size)

    for epoch in range(first_epoch+1, epochs+1):
        curr_batch = 0
        passes = 0

        #training and validation loss for each epoch
        training_loss=[]
        validation_loss=[]


        #for training
        for local_batch, local_labels in generator:

            truth_orient = local_labels['Orientation'].float().cuda()
            truth_conf = local_labels['Confidence'].long().cuda()
            truth_dim = local_labels['Dimensions'].float().cuda()

            local_batch=local_batch.float().cuda()
            [orient, conf, dim] = model(local_batch)

            orient_loss = orient_loss_func(orient, truth_orient, truth_conf)
            dim_loss = dim_loss_func(dim, truth_dim)

            truth_conf = torch.max(truth_conf, dim=1)[1]
            conf_loss = conf_loss_func(conf, truth_conf)

            loss_theta = conf_loss + w * orient_loss
            loss = alpha * dim_loss + loss_theta

            

            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step()

            
           


            if passes % 10 == 0:
                print("--- epoch %s | batch %s/%s --- [loss: %s]" %(epoch, curr_batch, total_num_batches, loss.item()))
                passes = 0

            passes += 1
            curr_batch += 1
            training_loss.append(loss.item())



        training_loss=np.array(training_loss)
        print(type(training_loss[0]))
        training_ave=np.sum(training_loss)/len(training_loss)

        training_loss_total.append(training_ave)
        print('----------------------training loss for this epoch is: '+str(training_ave))


        model.eval()
        #validation for each epoch
        with torch.no_grad():
            for local_batch2, local_labels2 in generator_valid:

                truth_orient2 = local_labels2['Orientation'].float().cuda()
                truth_conf2 = local_labels2['Confidence'].long().cuda()
                truth_dim2 = local_labels2['Dimensions'].float().cuda()

                local_batch2=local_batch2.float().cuda()
                [orient2, conf2, dim2] = model(local_batch2)

                orient_loss2 = orient_loss_func(orient2, truth_orient2, truth_conf2)
                dim_loss2 = dim_loss_func(dim2, truth_dim2)

                truth_conf2 = torch.max(truth_conf2, dim=1)[1]
                conf_loss2 = conf_loss_func(conf2, truth_conf2)

                loss_theta2 = conf_loss2 + w * orient_loss2
                loss2 = alpha * dim_loss2 + loss_theta2

                

                validation_loss.append(loss2.item())



        validation_loss=np.array(validation_loss)




        validation_ave=np.sum(validation_loss)/len(validation_loss)
        print("validation loss for this epoch is: "+str(validation_ave))
        validation_loss_total.append(validation_ave)
        print('----------------------validation loss for this epoch is: '+str(validation_ave))


        # save after every 10 epochs
        if epoch % 10 == 0:
            name = model_path + 'epoch_%s.pkl' % epoch
            print("====================")
            print ("Done with epoch %s!" % epoch)
            print ("Saving weights as %s ..." % name)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt_SGD.state_dict(),
                    'loss': loss
                    }, name)
            print("====================")

    index=np.arange(epochs)
    for i in range(len(index)):
        index[i] += 1

    x=index

    y1=training_loss_total
    y2=validation_loss_total
    plt.plot(x,y1,label="training loss")
    plt.plot(x,y2,label="validation loss")

    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig('loss.png')
    plt.show()

if __name__=='__main__':
    main()
