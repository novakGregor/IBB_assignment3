from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import os
import copy


def train_model(model, criterion, optimizer, scheduler, train_dataloader, device, dataset_size, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc, loss_at_best = 0.0, 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch should have a training and validation phase, however our dataset is quite
        # small already so we will avoid further splitting and use training phase only

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize, because we are in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        scheduler.step()

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        time_so_far = time.time() - since
        print('Currently elapsed time: {:.0f}m {:.0f}s'.format(time_so_far // 60, time_so_far % 60))

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            loss_at_best = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Acc: {:4f}'.format(best_acc))
    print('Loss at Best Acc: {:4f}'.format(loss_at_best))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main(model_ft, epochs, model_save_path):
    data_dir = "data/perfectly_detected_ears/pytorch_format"

    data_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), data_transform["train"])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    dataset_size = len(train_dataset)

    class_names = train_dataset.classes
    print("Training dataset size:", dataset_size)
    print("Detected classes:", class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device to be used:", device)
    print()

    try:
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    except AttributeError:
        num_ftrs = model.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, train_dataloader, device, dataset_size,
                           num_epochs=epochs)
    print("Saving model to", model_save_path)
    torch.save(model_ft.state_dict(), model_save_path)


if __name__ == "__main__":
    resnet18 = "resnet18"
    resnet152 = "resnet152"
    densenet161 = "densenet161"
    variables = {
        resnet18: [models.resnet18, "models/resnet18_{}_epochs"],
        resnet152: [models.resnet152, "models/resnet152_{}_epochs"],
        densenet161: [models.densenet161, "models/densenet151_{}_epochs"]
    }

    # change this to use different model
    used_model = resnet152

    load_model, model_save_path = variables[used_model]

    for num_epochs in range(50, 101, 25):
        model = load_model(pretrained=True)
        format_save_path = model_save_path.format(num_epochs)
        main(model, num_epochs, format_save_path)
