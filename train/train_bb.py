import os
import torch as th
from train.bb_dataset import BBDataSet

from net.simplenet2 import SimpleNet2


def train(csv_file, root_dir, epochs, batch_size=1, learning_rate=0.01):
    trainset = BBDataSet(csv_file, root_dir)
    trainloader = th.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    net = SimpleNet2(True)
    net.train()
    print(net.is_train)


    optimizer = th.optim.Adam(net.parameters(), lr=learning_rate)
    #criterion = th.nn.CrossEntropyLoss()
    criterion = th.nn.MSELoss()

    for epoch in range(epochs):
        running_loss = 0
        for i, data in enumerate(trainloader):
            inputs = data['image'].float()
            labels = data['bb'].float()

            optimizer.zero_grad()

            outputs = net(inputs)

            upsample = th.nn.Upsample(scale_factor=8)
            outputs = upsample(outputs)

            loss = criterion(outputs, labels)

            loss.backward()

            running_loss += loss.item()
            if i % 19 == 0:
                print('[%d, %5d loss: %.5f]' %
                      (epoch + 1, i + 1, running_loss))
                running_loss = 0.
                th.save(net.state_dict(), "../resource/trained_model/model.pth")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))

    csv_file = os.path.join(base, "../resource/data/bbdata.csv")
    root_dir = os.path.join(base, "../resource/image/")

    train(csv_file, root_dir, epochs=100000000, batch_size=10)
