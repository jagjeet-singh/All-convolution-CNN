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


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 3)    
	nn.init.xavier_normal(self.conv1.weight)
        #nn.init.xavier_normal(self.conv1.bias)
	self.dropout1 = nn.Dropout2d(0.2)
        self.conv2 = nn.Conv2d(96, 96, 3, 2, 2)
        nn.init.xavier_normal(self.conv2.weight)
        #nn.init.xavier_normal(self.conv2.bias)
	self.dropout2 = nn.Dropout2d(0.5)
	self.conv3 = nn.Conv2d(96, 192, 3)
        nn.init.xavier_normal(self.conv3.weight)
        #nn.init.xavier_normal(self.conv3.bias)
	self.conv4 = nn.Conv2d(192, 192, 3, 2, 2)
        nn.init.xavier_normal(self.conv4.weight)
        #nn.init.xavier_normal(self.conv4.bias)
	self.dropout3 = nn.Dropout2d(0.5)
	self.conv5 = nn.Conv2d(192, 192, 3)
	nn.init.xavier_normal(self.conv5.weight)
        #nn.init.xavier_normal(self.conv5.bias)
        self.fc1 = nn.Linear(192*6*6,10)
	self.conv6 = nn.Conv2d(192, 192, 1)
	nn.init.xavier_normal(self.conv6.weight)
        #nn.init.xavier_normal(self.conv6.bias)
        self.conv7 = nn.Conv2d(192, 10, 1)
	nn.init.xavier_normal(self.conv7.weight)
        #nn.init.xavier_normal(self.conv7.bias)
        self.glb_avg = nn.AvgPool2d(6)

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
	#x = self.dropout1(x)
	x = F.relu(self.conv2(x))
        #x = self.dropout2(x)
        x = F.relu(self.conv3(x))
        #x = self.dropout(x)
        x = F.relu(self.conv4(x))
        #x = self.dropout3(x)
        x = F.relu(self.conv5(x))
        #x = self.dropout(x)
        x = F.relu(self.conv6(x))
	#x = x.view(-1, 192*6*6)
	#x = F.relu(self.conv6(x))
        #x = self.dropout(x)
	#x = F.relu(self.fc1(x))
        x = F.relu(self.conv7(x))
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
#pdb.set_trace()
train_size = 50000
test_size = 10000
train_steps = train_size/train_batch_size
test_steps = test_size/test_batch_size


for epoch in range(max_epoch):  # loop over the dataset multiple times
	
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
        if i % 500 == 499:    # print every 500 mini-batches
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
    	#pdb.set_trace()
	inputs, labels = Variable(inputs_data), Variable(labels_data)
	outputs = net(inputs)
	loss = criterion(outputs,labels)
	running_loss_epoch += loss.data[0]
    	_, predicted = torch.max(outputs.data, 1)
    	total += labels.size(0)
    	correct += (predicted == labels_data).sum()

    test_acc[epoch] = correct/total*100
    test_loss[epoch] = running_loss_epoch / test_steps 
    np.savetxt('train_loss_60.txt',train_loss)
    np.savetxt('train_acc_60.txt',train_acc)
    np.savetxt('test_loss_60.txt',test_loss)
    np.savetxt('test_acc_60.txt',test_acc)
    
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
plt.savefig('Accuracy_epoch_60.png')
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
plt.savefig('Loss_epoch_60.png')
plt.show()

###############################################################################

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
