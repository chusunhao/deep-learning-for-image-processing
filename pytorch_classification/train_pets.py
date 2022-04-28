import torch
import torchvision
import torch.nn as nn
from Test1_official_demo.model import LeNet
from Test2_alexnet.model import AlexNet
import torch.optim as optim
import torchvision.transforms as transforms


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(size=(224, 224)),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 70% 训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    trainval_set = torchvision.datasets.OxfordIIITPet(root='./data', split="trainval",
                                                      download=True, transform=transform)

    train_size = int(0.7 * len(trainval_set))
    test_size = len(trainval_set) - train_size
    train_set, val_set = torch.utils.data.random_split(trainval_set, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8,
                                               shuffle=True, num_workers=0)

    # 30% 验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=16,
                                             shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)
    val_image, val_label = val_data_iter.next()

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    nets = [AlexNet(num_classes=trainval_set.classes.__len__(), init_weights=True)]
    for net in nets:
        # net = LeNet()
        net.to(torch.device("cuda:0"))
        print(f'Start Training {net._get_name()}')
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        for epoch in range(5):  # loop over the dataset multiple times

            running_loss = 0.0
            for step, data in enumerate(train_loader, start=0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(inputs.cuda())
                loss = loss_function(outputs, labels.cuda())
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if step % 10 == 9:  # print every 500 mini-batches
                    with torch.no_grad():
                        outputs = net(val_image.cuda())  # [batch, 10]
                        predict_y = torch.max(outputs, dim=1)[1]
                        accuracy = torch.eq(predict_y, val_label.cuda()).sum().item() / val_label.size(0)

                        print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                              (epoch + 1, step + 1, running_loss / 500, accuracy))
                        running_loss = 0.0

        print('Finished Training')

        save_path = f'./{net._get_name()}.pth'
        torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()
