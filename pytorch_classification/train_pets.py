import torch
import torchvision
import torch.nn as nn
from Test1_official_demo.model import LeNet
from Test2_alexnet.model import AlexNet
from Test3_vggnet.model import vgg
from Test4_googlenet.model import GoogLeNet
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
import sys
import os

class SplitDataset:
    def __init__(self, dataset, transform, split: str = 'train'):
        self.dataset = dataset
        self.split = split
        self.transform = transform

    def __getitem__(self, idx):
        d = self.dataset[idx][0]
        label = self.dataset[idx][1]
        trans = self.transform[self.split]
        return trans(d), label

    def __len__(self):
        return len(self.dataset)


def main():
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    # 第一次使用时要将download设置为True才会自动去下载数据集
    trainval_set = torchvision.datasets.OxfordIIITPet(root='./data', split="trainval",
                                                      download=True)
    # 70% 训练图片
    train_size = int(0.7 * len(trainval_set))
    test_size = len(trainval_set) - train_size
    train_dataset, validate_dataset = torch.utils.data.random_split(trainval_set, [train_size, test_size])
    train_dataset, validate_dataset = SplitDataset(train_dataset, data_transform, 'train'), SplitDataset(validate_dataset, data_transform, 'val')

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    # 30% 验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    val_data_iter = iter(validate_loader)
    val_image, val_label = val_data_iter.next()

    # AlexNet(num_classes=37, init_weights=True),
    nets = [vgg(model_name="vgg16", num_classes=37, init_weights=True),
            GoogLeNet(num_classes=37, aux_logits=True, init_weights=True)]
    for net in nets:
        # net = LeNet()
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        net.to(device)
        print(f'Start Training {net._get_name()}')
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        epochs = 100
        best_acc = 0.0
        train_steps = len(train_loader)
        for epoch in range(epochs):
            # train
            net.train()
            running_loss = 0.0
            train_bar = tqdm(train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar):
                images, labels = data
                optimizer.zero_grad()
                outputs = net(images.to(device))
                loss = loss_function(outputs, labels.to(device))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                         epochs,
                                                                         loss)

            # validate
            net.eval()
            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                val_bar = tqdm(validate_loader, file=sys.stdout)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = net(val_images.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            val_accurate = acc / val_num
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_accurate))

            save_path = f'./{net._get_name()}.pth'
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)

        print('Finished Training')


if __name__ == '__main__':
    main()
