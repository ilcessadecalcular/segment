import sys
import torch
from utils.distributed_utils import reduce_value,is_main_process
from loss_function import Binary_Loss,DiceLoss
from utils.metric import metric
from utils.padding import pad_tensor,pad_tensor_back


def train_one_epoch(model, optimizer, data_loader, device, epoch,criterion):
    model.train()
    mean_dice = torch.zeros(1).to(device)
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    for step, batch in enumerate(data_loader):
        if is_main_process():
            print(f"Batch: {step}/{len(data_loader)} epoch {epoch}")
            print(batch['source']['data'].shape)

        x = batch['source']['data']
        y = batch['label']['data']

        x = x.type(torch.FloatTensor).to(device)
        y = y.type(torch.FloatTensor).to(device)
        #print(x)
        predict = model(x)
        #predict = torch.sigmoid(predict)
        
        
        loss = criterion(predict, y)
        #loss.backward()
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)


        ys = y.detach()
        predicts = predict.detach()
        #logits = predicts
        logits = torch.sigmoid(predicts)
        labels = logits.clone()
        labels[labels > 0.5] = 1
        labels[labels <= 0.5] = 0
        
        smooth = 0.001
        intersection_sum = torch.sum(ys*labels)
        labels_sum = labels.sum()
        ys_sum = ys.sum()
        dice = 2*intersection_sum / (ys_sum + labels_sum +smooth)
        dice = reduce_value(dice, average=True)
        mean_dice = (mean_dice * step + dice) / (step + 1)
        
        if is_main_process():
            print("mean loss:" + str(mean_loss.item()))
            print('mean_dice'+str(mean_dice.item()))
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item(),mean_dice.item()

@torch.no_grad()
def evaluate(model, data_loader, device,epoch):
    model.eval()
    mean_dice = torch.zeros(1).to(device)
    for step, batch in enumerate(data_loader):
        #if is_main_process():
            #print('img',batch['source']['data'].shape)
        #    print('label',batch['label']['data'].shape)

        x = batch['source']['data']
        y = batch['label']['data']

        x = x.type(torch.FloatTensor).to(device)
        y = y.type(torch.FloatTensor).to(device)
        #print(x)
        #x, pad_front, pad_back, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(x,divide=16)
        #outputs = model(x)
        #outputs = pad_tensor_back(pad_front, pad_back, pad_left, pad_right, pad_top, pad_bottom)
        #print(torch.max(outputs,dim=1))
        
        # x1 = x[:,:,:300,:,:]
        # predict1 = model(x1)
        # #del x1
        # x2 = x[:,:,300:,:,:]
        # predict2 = model(x2)
        # #del x2
        # #del x
        #
        # predicts = torch.cat((predict1,predict2),2)
        # #del outputs1
        # #del outputs2

        predicts = model(x)

        # for metrics
        logits = torch.sigmoid(predicts)
        #logits = predicts
        labels = logits.clone()
        labels[labels > 0.5] = 1
        labels[labels <= 0.5] = 0
        #del outputs
        #logits = torch.argmax(outputs,dim=1)
        #labels = logits.clone()
        
        
        smooth = 0.001
        intersection_sum = torch.sum(y*labels)
        labels_sum = labels.sum()
        y_sum = y.sum()
        dice = 2*intersection_sum / (y_sum + labels_sum +smooth)
        dice = reduce_value(dice, average=True)
        #print('pt',labels.shape)
        #false_positive_rate, false_negtive_rate, dice = metric(y, labels)
        #mean_false_positive_rate = (mean_false_positive_rate*step + false_positive_rate) / (step+1)
        #mean_false_negtive_rate = (mean_false_negtive_rate * step + false_negtive_rate) / (step + 1)
        #del labels
        #del y
        mean_dice = (mean_dice * step + dice) / (step + 1)
        if is_main_process():
           print(f"Batch: {step}/{len(data_loader)} epoch {epoch}")
           print('intersection_sum'+str(intersection_sum))
           print('y_sum'+str(y_sum))
           print('labels_sum'+str(labels_sum))
           print('dice'+str(dice))
           print('mean_dice'+str(mean_dice))
        dice = torch.zeros(1).to(device)
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    #mean_false_positive_rate = reduce_value(mean_false_positive_rate, average=False)
    #mean_false_negtive_rate = reduce_value(mean_false_negtive_rate, average=False)
    
    print('mean_dice'+str(mean_dice))
    return mean_dice.item()

