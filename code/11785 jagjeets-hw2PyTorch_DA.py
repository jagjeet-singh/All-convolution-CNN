from __future__ import division
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pdb


transform = transforms.Compose([
     transforms.RandomSizedCrop(126),
     transforms.RandomHorizontalFlip(),
     #transforms.RandomVerticalFlip(),     
     transforms.ToTensor(),     
     #transforms.ToPILImage()
     #,transforms.Resize(126)
     #,transforms.RandomHorizontalFlip()
     #,transforms.RandomVerticalFlip()
     #transforms.ColorJitter(),
     #,transforms.ToTensor()
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))     
     ])

train_batch_size = 4
test_batch_size = 4

#trainset - abstract class representing the training Dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

#trainloader - combines train dataset and sampler, provides iterators
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                          shuffle=True, num_workers=2)

#testset - abstract class representing the test Dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

#testloader - combines test dataset and sampler, provides iterators
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                         shuffle=False, num_workers=2)

###############################################################################

# functions to show an image for an image passes as WxCxH

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

###############################################################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 3)    
	nn.init.xavier_normal(self.conv1.weight)
	self.conv2 = nn.Conv2d(96, 96, 3, 1, 1)
	nn.init.xavier_normal(self.conv2.weight)
        self.conv3 = nn.Conv2d(96, 96, 3, 2, 2)
        nn.init.xavier_normal(self.conv3.weight)
	self.conv4 = nn.Conv2d(96, 192, 3, 1, 1)
        nn.init.xavier_normal(self.conv4.weight)
	self.conv5 = nn.Conv2d(192, 192, 3)
        nn.init.xavier_normal(self.conv5.weight)
	self.conv6 = nn.Conv2d(192, 192, 3, 2, 2)
        nn.init.xavier_normal(self.conv6.weight)
	self.conv7 = nn.Conv2d(192, 192, 3)
        nn.init.xavier_normal(self.conv7.weight)
	self.conv8 = nn.Conv2d(192, 192, 1)
	nn.init.xavier_normal(self.conv8.weight)
	self.conv9 = nn.Conv2d(192, 10, 1)
	nn.init.xavier_normal(self.conv9.weight)
        self.glb_avg = nn.AvgPool2d(30)

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
	x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.glb_avg(x)
        x = x.view(-1, 10)
        return x

net = Net()

###############################################################################
criterion = nn.CrossEntropyLoss()

###############################################################################
lr1 = 0.25
lr2 = 0.1
lr3 = 0.05
lr4 = 0.001
max_epoch = 60
display_interval = 500

train_loss = np.zeros((max_epoch,1))
train_acc = np.zeros((max_epoch,1))
test_loss = np.zeros((max_epoch,1))
test_acc = np.zeros((max_epoch,1))
train_size = 50000
test_size = 10000
train_steps = train_size/train_batch_size
test_steps = test_size/test_batch_size


for epoch in range(max_epoch):
	
    if (epoch<200):
	lr = lr1
    elif (epoch<250):
	lr = lr2
    elif (epoch<300):
	lr = lr3
    else:
	lr = lr4
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    
    running_loss = 0.0
    running_loss_epoch = 0.0	
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs_data, labels_data = data

        inputs, labels = Variable(inputs_data), Variable(labels_data)

        optimizer.zero_grad()

        outputs = net(inputs)
	loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
	
	running_loss_epoch += loss.data[0] 
        
        # print statistics
        running_loss += loss.data[0]
        if i % 500 == 499:   
            print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0
	
	# Train accuracy
	_, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels_data).sum()
    train_loss[epoch] = running_loss_epoch / train_steps 
    train_acc[epoch] = correct/total*100

    running_loss_epoch = 0.0
    # Test accuracy and loss	
    correct = 0
    total = 0
    for data in testloader:
    	inputs_data, labels_data = data
	inputs, labels = Variable(inputs_data), Variable(labels_data)
	outputs = net(inputs)
	loss = criterion(outputs,labels)
	running_loss_epoch += loss.data[0]
    	_, predicted = torch.max(outputs.data, 1)
    	total += labels.size(0)
    	correct += (predicted == labels_data).sum()

    test_acc[epoch] = correct/total*100
    test_loss[epoch] = running_loss_epoch / test_steps 
    #np.savetxt('train_loss_60.txt',train_loss)
    #np.savetxt('train_acc_60.txt',train_acc)
    #np.savetxt('test_loss_60.txt',test_loss)
    #np.savetxt('test_acc_60.txt',test_acc)
    
print('#####Training Loss########')
print(train_loss)
print('#####Training Acc########')
print(train_acc)
print('#####Test Loss########')
print(test_loss)
print('#####Test Acc########')
print(test_acc)

print('Finished Training')


###############################################################################

correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Final Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

###############################################################################

# Plotting accuracies and losses

itern_axis_train = np.array(np.linspace(1,max_epoch,num=max_epoch))
itern_axis_test = np.array(np.linspace(1,max_epoch, num=max_epoch))

#Train and Test Accuracy
fig, ax=plt.subplots()
ax.plot(itern_axis_train,train_acc,'-b.', label='Train')
ax.plot(itern_axis_test,test_acc, '--r', label='Test')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title('Accuracy vs Epoch')
legend = ax.legend(loc='upper center', shadow=True)
plt.ylim((0,100))
#plt.savefig('Accuracy_epoch_60.png')
plt.show()

#Train and Test Cost
fig, ax=plt.subplots()
ax.plot(itern_axis_train,train_loss,'-b.', label='Train')
ax.plot(itern_axis_test,test_loss, '--r', label='Test')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss vs Epoch')
#ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.2f}"))
legend = ax.legend(loc='upper center', shadow=True)
plt.ylim((0,5))
#plt.savefig('Loss_epoch_60.png')
plt.show()

##############################################################################
