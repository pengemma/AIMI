from ResNet import ResNet18, ResNet50, ResNet152
from dataloader import ChestLoader, getData
from imbalanced import ImbalancedDatasetSampler

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import os
import matplotlib.pyplot as plt

def measurement(outputs, labels, smooth=1e-10):
    tp, tn, fp, fn = smooth, smooth, smooth, smooth
    labels = labels.cpu().numpy()
    outputs = outputs.detach().cpu().clone().numpy()
    for j in range(labels.shape[0]):
        if (int(outputs[j]) == 1 and int(labels[j]) == 1):
            tp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 0):
            tn += 1
        if (int(outputs[j]) == 1 and int(labels[j]) == 0):
            fp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 1):
            fn += 1
    return tp, tn, fp, fn

def evaluate(y, y_pred):
    cm = confusion_matrix(y, y_pred) #normalize='true'
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]).plot(cmap=plt.cm.Blues)
    plt.savefig(f'confusion.png')
    print(classification_report(y, y_pred, target_names=['0 - NORMAL', '1 - PNEUMONIA']))

def acc_fig(acc_file, net):
    # plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    columns = ["train_acc", "train_F1", "val_acc", "val_F1"]
    df = pd.read_csv(acc_file, usecols=columns)
    for phase in ["acc", "F1"]:
        df[[f'train_{phase}', f'val_{phase}']].plot()
        plt.savefig(f'{net} {phase} Curve.png')

def train(model, loader_train, loss, optimizer, device):
    model.train()
    predict_result = np.array([], dtype=int)
    
    train_loss = 0
    train_accuracy = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    
    for (data, target) in tqdm(loader_train):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        predict = model.forward(data)
        pred_labels = torch.max(predict, 1)[1]
        Loss = loss(predict, target)
        train_accuracy += (pred_labels == target).sum().item()
        train_loss += Loss.item()
        Loss.backward()
        optimizer.step()
        predict_result = np.concatenate((predict_result, pred_labels.detach().cpu().numpy()))

        sub_tp, sub_tn, sub_fp, sub_fn = measurement(pred_labels, target)
        tp += sub_tp
        tn += sub_tn
        fp += sub_fp
        fn += sub_fn

    F1_score = (2*tp) / (2*tp+fp+fn)
    train_accuracy = 100. * train_accuracy / len(loader_train.dataset)
    train_loss /= len(loader_train.dataset)

    print(f'↳ Loss: {train_loss}')
    print(f'↳ Training Acc.(%): {train_accuracy:.2f}%')

    return train_loss, train_accuracy, F1_score, predict_result

def valid(model, loader_valid, loss, device):
    model.eval()
    predict_result = np.array([], dtype=int)

    valid_loss = 0
    valid_accuracy = 0
    tp, tn, fp, fn = 0, 0, 0, 0

    with torch.no_grad():
        for (data, target) in tqdm(loader_valid):
            data = data.to(device)
            target = target.to(device)
            predict = model.forward(data)
            pred_labels = torch.max(predict, 1)[1]
            Loss = loss(predict, target)
            valid_accuracy += (pred_labels == target).sum().item()
            valid_loss += Loss.item()
            predict_result = np.concatenate((predict_result, pred_labels.detach().cpu().numpy()))

            sub_tp, sub_tn, sub_fp, sub_fn = measurement(pred_labels, target)
            tp += sub_tp
            tn += sub_tn
            fp += sub_fp
            fn += sub_fn

    recall = tp / (tp+fn)
    precision = tp / (tp+fp)
    F1_score = (2*tp) / (2*tp+fp+fn)
    valid_accuracy = 100. * valid_accuracy / len(loader_valid.dataset)
    valid_loss /= len(loader_valid.dataset)

    print (f'↳ Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {F1_score:.4f}')
    print (f'↳ Test Acc.(%): {valid_accuracy:.2f}%')

    return valid_loss, valid_accuracy, F1_score, predict_result


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Mode = int(input("Choose mode ==> 1:train、2:test、3:result ==> "))
    if Mode == 1 or Mode == 2:
        Choose = int(input("Choose ResNet model ==> 1:ResNet18、2:ResNet50、3:ResNet152 ==> "))
        if Choose == 1:
            model = ResNet18(num_classes = 2, channels = 1)
            name = 'resnet18'
            netnum = 18
        elif Choose == 2:
            model = ResNet50(num_classes = 2, channels = 1)
            name = 'resnet50'
            netnum = 50
        elif Choose == 3:
            model = ResNet152(num_classes = 2, channels = 1)
            name = 'resnet152'
            netnum = 152

    if Mode == 1:
        
        train_data = ChestLoader('', 'train')
        test_data = ChestLoader('', 'test')
        train_loader = DataLoader(dataset = train_data, batch_size = 64, shuffle = True) #sampler = ImbalancedDatasetSampler(train_data),
        test_loader = DataLoader(dataset = test_data, batch_size = 64, shuffle = False)

        model.to(device)

        epoch = 50

        loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([3.8896346, 1.346]))
        loss = loss.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.9)
        train_acc = []
        test_acc = []
        train_F1 = []
        test_F1 = []
        train_best = 0
        test_best = 0
        df = pd.DataFrame()

        for i in range(epoch):
            print(f'{name} Epoch: {i+1}')
            train_loss, train_accuracy, train_f1, train_pred = train(model, train_loader, loss, optimizer, device)
            test_loss, test_accuracy, test_f1, test_pred = valid(model, test_loader, loss, device)
            train_acc.append(train_accuracy)
            test_acc.append(test_accuracy)
            test_F1.append(test_f1)
            train_F1.append(train_f1)

            if train_accuracy > train_best:
                train_best = train_accuracy
                train_best_pred = train_pred
                train_best_param = copy.deepcopy(model)
                torch.save(train_best_param.state_dict(), name+'_train'+'_'+str(round(max(train_acc), 2))+'%.pth')
            if test_accuracy > test_best:
                test_best = test_accuracy
                test_best_pred = test_pred
                test_best_param = copy.deepcopy(model)
                torch.save(test_best_param.state_dict(), name+'_test'+'_'+str(round(max(test_acc), 2))+'%.pth')            

        df['train_acc'] = train_acc
        df['train_F1'] = train_F1
        df['test_acc'] = test_acc
        df['test_F1'] = test_F1
        df.to_csv(f'./{name}_acc_f1.csv', index=False)
    
    elif Mode == 2:

        print('=============================generate test result=============================')
        test_data = ChestLoader('', 'test')
        test_loader = DataLoader(dataset = test_data, batch_size = 128, shuffle = False)
        loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([3.8896346, 1.346])).to(device)
        for file in os.listdir('best/'):
            model.to(device)
            if name in file:
                model.load_state_dict(torch.load('best/' + file))
                valid(model, test_loader, loss, device)

    elif Mode == 3:
        
        print('=============================generate acc_fig and confusion_matrix=============================')
        test_data = ChestLoader('', 'test')
        test_loader = DataLoader(dataset = test_data, batch_size = 128, shuffle = False)

        net = ['resnet18', 'resnet50']
        for i in net:
            acc_fig(f'{i}_acc_f1.csv', i)

        model = ResNet18(num_classes = 2, channels = 1)
        model.to(device)
        loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([3.8896346, 1.346])).to(device)
        model.load_state_dict(torch.load('best/resnet18_test_91.03%.pth', map_location='cpu'))
        _, accuracy, f1, pred = valid(model, test_loader, loss, device)
        evaluate(getData("test")[1], pred.tolist())