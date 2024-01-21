import time
import os, glob
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torch.nn.init

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)  # reproducibility

if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
keep_prob = 0.7

first = time.perf_counter()
# preprocessing
trans = transforms.Compose([transforms.Resize((28, 28)),
                            transforms.ToTensor(),
                            transforms.Grayscale(1),
                            transforms.Normalize([0.5, ], [0.5, ])
                            ])

# dataset


test = torchvision.datasets.ImageFolder(root='./test', transform=trans)

# dataset loader


test_iter = torch.utils.data.DataLoader(dataset=test,
                                        batch_size=batch_size,
                                        shuffle=True)


# CNN Model


class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # L1 ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L3 ImgIn shape=(?, 7, 7, 64)
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=1 - keep_prob))

        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - keep_prob))
        # L5 Final FC 625 inputs -> 10 outputs
        self.fc2 = torch.nn.Linear(625, 26, bias=True)
        torch.nn.init.xavier_uniform(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)  # Flatten them for FC
        out = self.fc1(out)
        out = self.fc2(out)
        return out


model = CNN().to(device)

model.load_state_dict(torch.load('./p201710869.pth'))

# define cost/loss & optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)  # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Test model and check accuracy

with torch.no_grad():
    for i, (batch_xs, batch_ys) in enumerate(test_iter):
        X_test = Variable(batch_xs).to(device)
        Y_test = Variable(batch_ys).to(device)

        prediction = model(X_test)
        # correct_prediction = (torch.max(prediction.data, 1)[1] == Y_test.data)
        correct_prediction = torch.argmax(prediction, 1) == Y_test

        accuracy = correct_prediction.float().mean()

    print('Accuracy:', accuracy.item())
    second = time.perf_counter()
    count = len(glob.glob('./test/**', recursive=True))
    print(count)
    time_tot = ((second - first)/count)
    print('time:', time_tot)
