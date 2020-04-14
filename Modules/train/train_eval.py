
from tqdm import tqdm

import torch
import torch.nn.functional as F

def train(model, device, trainloader, optimizer, i, train_losses, train_acc):
    model.train()
    pbar = tqdm(trainloader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(data)
        pred = F.log_softmax(pred, dim=-1) 
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        pred = pred.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        total = len(data)
        train_losses.append(loss.item())
        train_acc.append(correct/ total)
        pbar.set_description(desc=f'{i} loss : {loss.item()} acc : {correct/ total}')
        
def test(model, device, test_loader, test_losses, test_acc):
    model.eval()
    correct_agg = 0
    total_agg = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            loss = F.nll_loss(pred, target)
            pred = pred.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            total = len(data)
            total_agg += total
            correct_agg += correct
            test_losses.append(loss.item())
            test_acc.append(correct/ total)
        print('Acc : ', correct_agg / total_agg * 100.0)