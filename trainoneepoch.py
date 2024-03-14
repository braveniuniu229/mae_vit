import torch
import datetime
import os

def train_one_epoch(epoch, model, data_loader, loss_function, optimizer, device='cuda'):
    model.train()
    total_loss = 0.0
    for batch_idx, (data, targets) in enumerate(data_loader):
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    print(f'Epoch {epoch}, Average Loss: {average_loss}')

    return average_loss

best_loss = float('inf')

def save_checkpoint(epoch, model, optimizer, loss, best_loss, filename='checkpoint.pth.tar'):
    is_best = loss < best_loss
    if is_best:
        print(f'New best loss: {loss}, previous best: {best_loss}')
        best_loss = loss
        checkpoint = {
            'epoch': epoch + 1,  # next epoch
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
        }
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_dir = f'{timestamp}'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        filename = os.path.join(save_dir,filename)
        torch.save(checkpoint, filename)

def train_one_epoch_with_save(epoch, model, data_loader, loss_function, optimizer, device='cuda'):
    average_loss = train_one_epoch(epoch, model, data_loader, loss_function, optimizer, device)
    save_checkpoint(epoch, model, optimizer, average_loss, best_loss)
