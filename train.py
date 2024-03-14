import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
import time
import os
from dataset import dataset_train,dataset_test
from mlp import MultiMLP
import csv
import wandb
# 假设的模型和优化器
#
# wandb.init(
#     project="mlp_imple",
#
#
# )
# wandb.config = {
#       "learning_rate": 0.001,
#       "momentum":0.9,
#       "basicmodel": 'vgg16',
#       "batch_size": 800,
#
#       "num_epochs": 50  # 假设训练 30 个 epochs
#
#     }

#定义模型
layer_fixed = [16, 128, 1024, 2048, 4096]
model = MultiMLP(layer_fixed)
train_loader = DataLoader(dataset_train,batch_size=800,shuffle=True)
test_loader = DataLoader(dataset_test,batch_size=800,shuffle=False)
file = 'firsttry'
"""这里每次都要修改成训练的model"""
checkpoint_dir = "first"   #这里修改成训练的断点

optimizer = optim.SGD(model.parameters(), 0.001, momentum=0.8)
criterion = nn.L1Loss()  # 假设使用均方误差损失

# 记录文件和检查点路径
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(f'experiresult/{file}'):
    os.makedirs(f'experiresult/{file}')




def write_to_csv(file_path, epoch, loss):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, loss])
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def save_checkpoint(epoch, iteration, model, optimizer, loss, is_best=False):
    state = {
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    # filename = f"{checkpoint_dir}/checkpoint_epoch_{epoch}_iter_{iteration}.pth"
    # torch.save(state, filename)
    if is_best:
        torch.save(state, f"{checkpoint_dir}/checkpoint_best.pth")

def train(epoch):

    model.train()
    start_time = time.time()

    total_loss = 0  # 用于累积每个 epoch 的总损失
    pbar = tqdm.tqdm(total=len(train_loader), desc=f"Training Epoch {epoch}", leave=True,colour='white')
    for iteration, (data,labels) in enumerate(train_loader):
        data,labels = data,labels
        data,labels = data.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # if (iteration+1)%200 == 0:
        #     iter_loss = loss.item()
        #     write_to_csv(f'experiresult/{file}/train_log_iter.csv', epoch * len(train_loader) + iteration, iter_loss)
        #     """xiugai"""



        pbar.set_description(f"Training Epoch {epoch} [{iteration}/{len(train_loader)}]")
        pbar.update(1)  # 更新进度条
    pbar.close()
    average_loss = total_loss / len(train_loader)  # 计算平均损失
    epoch_time = time.time() - start_time
    write_to_csv(f'experiresult/{file}/train_log.csv', epoch, average_loss)
    # wandb.log({"average_loss_train":average_loss,'epoch_usedtime':epoch_time })
    # githubllllkk
    """这里要修改模型的路径"""
    print("epoch:",epoch,'\n',"loss:",average_loss,'\n',"epoch time used:",epoch_time)
def validate(epoch, best_loss):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(test_loader), desc=f"Training Epoch {epoch}", leave=True, colour='white')
        for iteration, (data, labels) in enumerate(test_loader):
            data, labels = data, labels
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    write_to_csv(f'experiresult/{file}/val_log.csv', epoch, avg_loss)
    # wandb.log({"average_loss_val": avg_loss})
    # 如果是最好的损失，保存为 best checkpoint
    if avg_loss < best_loss:
        best_loss = avg_loss
        save_checkpoint(epoch, 'best', model, optimizer, avg_loss, is_best=True)

        print("产生了新的最优结果，模型已经保存!")
    return best_loss

best_loss = float('inf')
num_epochs = 60  # 假设训练 30 个 epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if __name__ == "__main__":
    print(f'The model has {count_parameters(model):,} trainable parameters')
    print(device)
    for epoch in tqdm.trange(0, num_epochs):
        train(epoch)

        best_loss = validate(epoch, best_loss)
