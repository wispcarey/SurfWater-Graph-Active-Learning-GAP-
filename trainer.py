import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import numpy as np
import my_cnn_sw
import argparse

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            x = self.transform(x)
        if self.targets is not None:
            y = self.targets[index]
            return x, y
        else:
            return x

    def __len__(self):
        return len(self.data)

def trainer(dic_path, num_epoch, batch_size, checkpoint_path=None,
            learning_rate=1e-2, save_checkpoint_path='my_deepwatermap.pt'):
    '''
        dic_path: dictionary, {'train':{'label','image'},'test':{'label','image'}}
        num_epoch: number of training epochs
        batch_size: the size of each batch in the training process
        checkpoint_path: load a checkpoint to continue training. None is to train from scratch
        learning_rate: learning rate parameter
        save_checkpoint_path: the path where checkpoint is saved to
    '''

    dic = np.load(dic_path, allow_pickle='TRUE').item()
    train_data = torch.from_numpy(dic['train']['image']).float().permute(0,3,1,2)
    train_labels = torch.from_numpy(dic['train']['onehot_label']).float()
    test_data = torch.from_numpy(dic['test']['image']).float().permute(0,3,1,2)
    test_labels = torch.from_numpy(dic['test']['onehot_label']).float()

    surfwater = MyDataset(train_data, train_labels)
    data_loader = DataLoader(surfwater, shuffle=True)
    surfwater_test = MyDataset(test_data, test_labels)
    data_loader_test = DataLoader(surfwater_test, batch_size=batch_size, shuffle=True)

    num_class = train_labels.shape[1]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')
    model = my_cnn_sw.WaterBoundaryCNN(input_num_c=6, num_class=num_class).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.01)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    loss_fun = nn.MSELoss().to(device)

    for epoch in range(num_epoch):

        model.train()
        for batchidx, (data, label) in enumerate(data_loader):
            data, label = data.to(device), label.to(device)

            x_pred = model(data)
            loss = loss_fun(x_pred, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for data, label in data_loader_test:
                data, label = data.to(device), label.to(device)

                x_pred = model(data)
                pred = x_pred.argmax(dim = 1).float()
                true_label = label.argmax(dim = 1).float()
                total_correct += torch.eq(pred, true_label).float().sum().item()
                total_num += torch.numel(pred)

            acc = total_correct / total_num
            if np.mod(epoch, 50) == 0:
                print(epoch, 'loss:', loss.item())
                print(epoch, 'acc:', acc)

        if not np.mod(epoch, 100) and epoch != 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, save_checkpoint_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dic', type=str, default='data/train_test_dic.npy',
                        help='path to the directory of the raw data')
    parser.add_argument('--num_epoch', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--checkpoint_path', type=int, default=None,
                        help='path to the data directory of the checkpoint')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='learning_rate')
    parser.add_argument('--save_checkpoint_path', type=str, default='data/my_deepwatermap.pt',
                        help='the path where the checkpoint is saved to')
    args = parser.parse_args()

    trainer(args.dic, args.num_epoch, args.batch_size, args.checkpoint_path,
            args.learning_rate, None)