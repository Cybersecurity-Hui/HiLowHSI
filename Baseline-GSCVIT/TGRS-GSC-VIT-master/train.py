import os
import numpy as np
from tqdm import tqdm
import torch
from utils.utils import grouper, sliding_window, count_sliding_window


def train(network, optimizer, criterion, train_loader, val_loader, epoch, saving_path, device, scheduler=None):
    best_acc = -0.1
    best_loss = float('inf')  # 初始化最佳损失值为无穷大
    losses = []

    for e in tqdm(range(1, epoch+1), desc=""):
        network.train()
        epoch_losses = []  # 用于记录每个epoch的所有batch的损失值
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = network(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            losses.append(loss.item())

        # 计算当前epoch的平均损失值
        current_mean_loss = np.mean(epoch_losses)
        
        if e % 5 == 0 or e == 1:
            mean_losses = np.mean(losses)
            train_info = "train at epoch {}/{}, loss={:.6f}"
            train_info = train_info.format(e, epoch, mean_losses)
            tqdm.write(train_info)
            losses = []
        else:
            losses = []

        val_acc = validation(network, val_loader, device)

        if scheduler is not None:
            scheduler.step()

        # 检查是否是最佳准确率和最小损失值
        is_best_acc = val_acc >= best_acc
        is_best_loss = current_mean_loss <= best_loss
        
        best_acc = max(val_acc, best_acc)
        best_loss = min(current_mean_loss, best_loss)
        
        # 保存检查点
        save_checkpoint(
            network, 
            is_best_acc=is_best_acc,
            is_best_loss=is_best_loss, 
            saving_path=saving_path, 
            epoch=e, 
            acc=best_acc,
            loss=current_mean_loss,
            best_loss=best_loss
        )


def validation(network, val_loader, device):
    num_correct = 0.
    total_num = 0.
    network.eval()
    for batch_idx, (images, targets) in enumerate(val_loader):
        images, targets = images.to(device), targets.to(device)
        outputs = network(images)
        _, outputs = torch.max(outputs, dim=1) 
        for output, target in zip(outputs, targets):
            num_correct = num_correct + (output.item() == target.item())
            total_num = total_num + 1
    overall_acc = num_correct / total_num
    return overall_acc


def test(network, model_dir, image, patch_size, n_classes, device):
    network.load_state_dict(torch.load(model_dir + "/model_best_acc.pth"))
    network.eval()

    patch_size = patch_size
    batch_size = 64
    window_size = (patch_size, patch_size)
    image_w, image_h = image.shape[:2]
    pad_size = patch_size // 2

    # pad the image
    image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')

    probs = np.zeros(image.shape[:2] + (n_classes, ))

    iterations = count_sliding_window(image, window_size=window_size) // batch_size
    for batch in tqdm(grouper(batch_size, sliding_window(image, window_size=window_size)),
                      total=iterations,
                      desc="inference on the HSI"):
        with torch.no_grad():
            data = [b[0] for b in batch]
            data = np.copy(data)
            data = data.transpose((0, 3, 1, 2))
            data = torch.from_numpy(data)
            data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            output = network(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu').numpy()

            for (x, y, w, h), out in zip(indices, output):
                probs[x + w // 2, y + h // 2] += out
    return probs[pad_size:image_w + pad_size, pad_size:image_h + pad_size, :]

def save_checkpoint(network, is_best_acc, is_best_loss, saving_path, **kwargs):
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path, exist_ok=True)

    if is_best_acc:
        tqdm.write("epoch = {epoch}: best OA = {acc:.4f}, loss = {loss:.6f}".format(**kwargs))
        torch.save(network.state_dict(), os.path.join(saving_path, 'model_best_acc.pth'))
    
    if is_best_loss:
        tqdm.write("epoch = {epoch}: current loss = {loss:.6f} (best loss = {best_loss:.6f})".format(**kwargs))
        torch.save(network.state_dict(), os.path.join(saving_path, 'model_best_loss.pth'))
    

