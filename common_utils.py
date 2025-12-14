import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb
import time
import os
from torch.profiler import profile, ProfilerActivity, record_function
import copy

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

label_to_text = {
    0: "Angry", 
    1: "Disgust", 
    2: "Fear", 
    3: "Happy", 
    4: "Sad", 
    5: "Surprise", 
    6: "Neutral"
}

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 1024, 3, padding=1)
        self.conv5 = nn.Conv2d(1024, 2048, 3, padding=1)
        self.conv6 = nn.Conv2d(2048, 4192, 3, padding=0)
        self.fc1 = nn.Linear(4192, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class QuantizedModelWrapper(nn.Module):
    def __init__(self, model, backend):
        super(QuantizedModelWrapper, self).__init__()
        self.model = model
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

def test(model: nn.Module, dataloader: DataLoader, max_samples=None) -> float:
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0
    n_inferences = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            if max_samples:
                n_inferences += images.shape[0]
                if n_inferences > max_samples:
                    break

    return total_loss / total, 100 * correct / total

def benchmark_model(model: nn.Module, dataloader: DataLoader, max_samples = None):
    total = 0
    total_time = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)
            start_time = time.time()
            outputs = model(images)    
            end_time = time.time()    
            total += labels.size(0)    
            total_time += end_time - start_time
            outputs[0]
            if (total > max_samples):
                break
    print("Total Images Processed : ", total)
    print("Total Time : ", total_time, "seconds")
    print("Average Time: ", total_time / total, "seconds")
    print("Average FPS: ", 1 / ( total_time / total))
    return {
        "Total Images": total,
        "tot_time": total_time,
        "mean_time": total_time / total,
        "mean_fps": 1 / ( total_time / total)
    }

def profile_model(model: nn.Module, dataloader: DataLoader, max_samples=None):
    total = 0
    total_time = 0
    model.eval()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                 profile_memory=False, record_shapes=False) as prof:
        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                start_time = time.time()
                with record_function("model_inference"):
                    outputs = model(images)    
                end_time = time.time()    
                total += labels.size(0)    
                total_time += end_time - start_time
                outputs[0]
                if (total > max_samples):
                    break
    return prof

class TrainingParams:
    def __init__(self):
        self.checkpoint = False
        self.save_state_dict = True
        self.dir_name = ""
        self.lr = .0001


def train(run: wandb.Run, model: nn.Module, params: TrainingParams,
          train_dataloader: DataLoader, test_dataloader: DataLoader, 
          num_epochs=10):
    if (params.checkpoint and not params.dir_name):
        print ("Error: When checkpointing is active, directory name must be provided")
        return
    if (params.dir_name):
        os.makedirs(params.dir_name, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), params.lr)
    train_loss, train_acc = test(model, train_dataloader)
    test_loss, test_acc = test(model, test_dataloader)
    print("Before Training")
    print(
        """
        Train Loss: {0}
        Train Acc: {1}
        Test Loss: {2}
        Test Acc: {3}    
        """.format(train_loss, train_acc, test_loss, test_acc)
    )
    if (run is not None):
        run.log({"train/loss": train_loss, "test/loss": test_loss, 
                "train/acc": train_acc, "test/acc": test_acc})
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        epoch_loss = 0.0
        model.train()
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % 1000 == 999:    # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
        train_loss, train_acc = test(model, train_dataloader)
        test_loss, test_acc = test(model, test_dataloader)
        print("Epoch {0} Finished".format(epoch + 1))
        print(
            """
            Train Loss: {0}
            Train Acc: {1}
            Test Loss: {2}
            Test Acc: {3}
            """.format(train_loss, train_acc, test_loss, test_acc)
        )

        if (run is not None):
            run.log({"train/loss": train_loss, "test/loss": test_loss, 
                    "train/acc": train_acc, "test/acc": test_acc})
        if (params.checkpoint): #Be careful this can and will just overwrite files
            if (params.save_state_dict):
                torch.save(model.state_dict(), os.path.join(params.dir_name, "checkpoint_{0}.pth".format(epoch + 1)))
            else:
                torch.save(model, os.path.join(params.dir_name, "checkpoint_{0}.pth".format(epoch + 1)))


    print('Finished Training')

def quantize_model(model, dataloader):
    backend = "fbgemm"
    quantized_model= QuantizedModelWrapper(copy.deepcopy(model), None)
    quantized_model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(quantized_model, inplace=True)
    test(quantized_model, dataloader, 500)
    torch.quantization.convert(quantized_model, inplace=True)
    return quantized_model


