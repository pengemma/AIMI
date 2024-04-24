from dataloader import read_bci_data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import os

def Dataset_Loader(data, label, Shuffle):
    Dataset = TensorDataset(torch.from_numpy(data), torch.from_numpy(label))
    return DataLoader(dataset = Dataset, batch_size = 256, shuffle = Shuffle)

class EEGNet(nn.Module):
    def __init__(self, activation):
        super(EEGNet, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16,eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25)
        )
        self.seperableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0),
            nn.Dropout(p=0.25)
        )
        self.classify = nn.Linear(736, 2, bias=True)

    def forward(self, input):
        output = self.firstconv(input)
        output = self.depthwiseConv(output)
        output = self.seperableConv(output)
        output = output.view(-1, 736)
        output = self.classify(output)
        return output

class DeepConvNet(nn.Module):
    def __init__(self, activation):
        super(DeepConvNet, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.Conv2d(25, 25, kernel_size=(2, 1), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(25, eps=1e-5, momentum=0.1),
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(50, eps=1e-5, momentum=0.1),
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(100, eps=1e-5, momentum=0.1),
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(200, eps=1e-5, momentum=0.1),
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.classify = nn.Linear(8600, 2, bias=True)

    def forward(self, input):
        output = self.conv0(input)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.view(-1, 8600)
        output = self.classify(output)
        return output

def train(model, loader_train, loss, optimizer, device):
    model.train()
    
    train_loss = 0
    train__accuracy = 0

    for batch_idx, (data, target) in enumerate(loader_train):
        data = data.to(device, dtype = torch.float32)
        target = target.to(device, dtype = torch.long)
        optimizer.zero_grad()
        predict = model.forward(data)
        Loss = loss(predict, target)
        train__accuracy += (torch.max(predict, 1)[1] == target).sum().item()
        train_loss += Loss.item()
        Loss.backward()
        optimizer.step()

    train__accuracy = 100. * train__accuracy / len(loader_train.dataset)
    train_loss /= len(loader_train.dataset)

    return train_loss, train__accuracy

def test(model, loader_test, device):
    model.eval()
    test_accuracy = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader_test):
            data = data.to(device, dtype = torch.float32)
            target = target.to(device, dtype = torch.long)
            predict = model.forward(data)
            test_accuracy += (torch.max(predict, 1)[1] == target).sum().item()

    test_accuracy = 100. * test_accuracy / len(loader_test.dataset)
    return test_accuracy

def show_result(model_name, acc):
    plt.figure(figsize=(10, 6))
    plt.title(f'Activation Function Comparision({model_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    for name, curve in acc.items():
        plt.plot(curve, label = name)
        print(name + ' Max Accuracy:' + str(max(curve)))
    plt.legend()
    plt.savefig(f'{model_name}.png')
    plt.show()

def show_loss_curve(model_name, loss_curve):
    plt.figure(figsize=(10, 6))
    plt.title(f'Loss Curve({model_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    for name, curve in loss_curve.items():
        plt.plot(curve, label = name)
    plt.legend()
    plt.savefig(f'{model_name}_loss_curve.png')
    plt.show()
    
if __name__ == '__main__':
    train_data, train_label, test_data, test_label = read_bci_data()
    train_dataloader = Dataset_Loader(train_data, train_label, True)
    test_dataloader = Dataset_Loader(test_data, test_label, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CNN_model = int(input("Choose CNN model ==> 1:EEGNet、2:DeepConvNet ==> "))

    if CNN_model == 1:
        models = {
                'EEGNet_ELU': EEGNet(nn.ELU()).to(device),
                'EEGNet_ReLU': EEGNet(nn.ReLU()).to(device),
                'EEGNet_LeakyReLU': EEGNet(nn.LeakyReLU()).to(device),
            }
    elif CNN_model == 2:
        models = {
                'DeepConvNet_ELU': DeepConvNet(nn.ELU()).to(device),
                'DeepConvNet_ReLU': DeepConvNet(nn.ReLU()).to(device),
                'DeepConvNet_LeakyReLU': DeepConvNet(nn.LeakyReLU()).to(device)
            }
    # elif CNN_model == 3:
    #     models = {
    #             'EEGNet_ELU_0.9': EEGNet(nn.ELU(alpha=0.9)).to(device),
    #             'EEGNet_ELU_0.8': EEGNet(nn.ELU(alpha=0.8)).to(device),
    #             'EEGNet_ELU_0.7': EEGNet(nn.ELU(alpha=0.7)).to(device),
    #             'EEGNet_ELU_0.6': EEGNet(nn.ELU(alpha=0.6)).to(device),
    #             'EEGNet_ELU_0.5': EEGNet(nn.ELU(alpha=0.5)).to(device),
    #             'EEGNet_ELU_0.4': EEGNet(nn.ELU(alpha=0.4)).to(device),
    #             'EEGNet_ELU_0.3': EEGNet(nn.ELU(alpha=0.3)).to(device),
    #             'EEGNet_ELU_0.2': EEGNet(nn.ELU(alpha=0.2)).to(device),
    #             'EEGNet_ELU_0.1': EEGNet(nn.ELU(alpha=0.1)).to(device),
    #         }
    # elif CNN_model == 4:
    #     models = {
    #             'DeepConvNet_ELU_0.9': DeepConvNet(nn.ELU(alpha=0.9)).to(device),
    #             'DeepConvNet_ELU_0.8': DeepConvNet(nn.ELU(alpha=0.8)).to(device),
    #             'DeepConvNet_ELU_0.7': DeepConvNet(nn.ELU(alpha=0.7)).to(device),
    #             'DeepConvNet_ELU_0.6': DeepConvNet(nn.ELU(alpha=0.6)).to(device),
    #             'DeepConvNet_ELU_0.5': DeepConvNet(nn.ELU(alpha=0.5)).to(device),
    #             'DeepConvNet_ELU_0.4': DeepConvNet(nn.ELU(alpha=0.4)).to(device),
    #             'DeepConvNet_ELU_0.3': DeepConvNet(nn.ELU(alpha=0.3)).to(device),
    #             'DeepConvNet_ELU_0.2': DeepConvNet(nn.ELU(alpha=0.2)).to(device),
    #             'DeepConvNet_ELU_0.1': DeepConvNet(nn.ELU(alpha=0.1)).to(device),
    #         }


    acc = {'Train_ELU': None, 'Train_ReLU': None, 'Train_LeakyReLU': None,
           'Test_ELU': None, 'Test_ReLU': None, 'Test_LeakyReLU': None}
    
    best_param = {'ELU': None, 'ReLU': None, 'LeakyReLU': None}

    loss_curve = {'ELU': None, 'ReLU': None, 'LeakyReLU': None}

    # elu_acc = {'Test_0.9': None, 'Test_0.8': None, 'Test_0.7': None, 'Test_0.6': None, 'Test_0.5': None, 'Test_0.3': None, 'Test_0.2': None, 'Test_0.1': None}

    for name, model in models.items():

        model_name, activation, num = name.split('_')
        epoch = 300
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.01)
        train_acc = []
        test_acc = []
        train_loss_curve = []
        best = 0

        with tqdm(range(epoch)) as pbar:
            for i in pbar:
                train_loss, train_accuracy = train(model, train_dataloader, loss, optimizer, device)
                test_accuracy = test(model, test_dataloader, device)
                train_acc.append(train_accuracy)
                test_acc.append(test_accuracy)
                train_loss_curve.append(train_loss)

                if test_accuracy > best:
                    best = test_accuracy
                    best_param[activation] = copy.deepcopy(model)

                pbar.set_description(f'{name} epcoh{i+1}  loss:{train_loss:.5f}  acc:{train_accuracy:.2f}%')

        acc['Train_' + activation] = train_acc
        acc['Test_' + activation] = test_acc
        # elu_acc['Test_' + num] = round(best, 2)
        loss_curve[activation] = train_loss_curve

    for name, param in best_param.items():
        torch.save(param, model_name+'_'+name+'_'+str(round(max(acc['Test_' + name]), 2))+'%.pth')

    show_result(model_name, acc)
    show_loss_curve(model_name, loss_curve)
    # print(elu_acc)

    print('=============================best accuracy=============================')
    for file in os.listdir('best/'):
        name, _ = file.rsplit('_', 1)
        model = torch.load('best/'+file)
        accuracy = test(model, test_dataloader, device)
        print(name + ' best accuracy:' + str(accuracy))