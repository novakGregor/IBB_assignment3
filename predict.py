import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np


def line_plot(save_path, resnet18_data, resnet152_data, densenet161_data, ylim, title):
    plt.figure(figsize=(8, 6))
    x_axis = [25, 50, 75, 100]
    plt.xticks(x_axis)
    plt.ylim(0, ylim)
    plt.plot(x_axis, resnet18_data, label="resnet18")
    plt.scatter(x_axis, resnet18_data)
    plt.plot(x_axis, resnet152_data, label="resnet152")
    plt.scatter(x_axis, resnet152_data)
    plt.plot(x_axis, densenet161_data, label="densenet161")
    plt.scatter(x_axis, densenet161_data)
    plt.title(title)
    plt.xlabel("Number of training epochs")
    plt.ylabel("Value")
    for data in [resnet18_data, resnet152_data, densenet161_data]:
        for i, j in zip(x_axis, data):
            if j % 1 == 0:
                annotation = str(j)
            else:
                annotation = "{:.2f}".format(j)
            plt.text(i, j, annotation, fontsize=7)
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def run_prediction(model, model_path, test_data_dir, train_data_dir):
    rank1_score, rank5_score = 0, 0
    data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # load dataset
    dataset = datasets.ImageFolder(test_data_dir, data_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)
    train_dataset = datasets.ImageFolder(train_data_dir)
    print(dataset.classes)
    dataset_size = len(dataset)

    try:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
    except AttributeError:
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, len(train_dataset.classes))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        for data, target in dataloader:
            correct_class = dataset.classes[target.item()]
            prediction = model(data)

            # get best 5 predictions and best prediction
            best5_indices = np.argsort(prediction).tolist()[0][::-1][:5]
            best5_predicted = [train_dataset.classes[i] for i in best5_indices]
            # should be the same as best5_predicted[0]
            best_predicted = train_dataset.classes[np.argmax(prediction).item()]

            print("Correct class:", correct_class)
            print("Rank-1 predicted class:", best_predicted)
            print("Rank-5 predicted classes:", best5_predicted)

            # increment scores if needed
            if best_predicted == correct_class:
                rank1_score += 1
            if correct_class in best5_predicted:
                rank5_score += 1
            print("=====")

    rank1 = rank1_score / dataset_size
    rank5 = rank5_score / dataset_size
    print("Dataset size:", dataset_size)
    print("Number of correct predictions (Rank-1):", rank1_score)
    print("Number of correct predictions (Rank-5):", rank5_score)
    print("Rank-1 accuracy: {} %".format(rank1 * 100))
    print("Rank-5 accuracy: {} %".format(rank5 * 100))

    return rank1, rank5


if __name__ == '__main__':
    resnet18 = "resnet18"
    resnet152 = "resnet152"
    densenet161 = "densenet161"
    variables = {
        resnet18: [models.resnet18, "models/resnet18_{}_epochs", "plots/matrix_resnet18_{}_epochs"],
        resnet152: [models.resnet152, "models/resnet152_{}_epochs", "plots/matrix_resnet152_{}_epochs"],
        densenet161: [models.densenet161, "models/densenet161_{}_epochs", "plots/matrix_densenet161_{}_epochs"]
    }

    # used_dataset = "data/my_yolov5_detected_ears/yolov5_cropped_pytorch"
    used_dataset = "data/perfectly_detected_ears/pytorch_format/test"
    train_dataset = "data/perfectly_detected_ears/pytorch_format/train"

    # rank-1 and rank-5 results for all models
    results = {}

    # use all saved models with different amount of training epochs
    for used_model_name in [resnet18, resnet152, densenet161]:
        rank1_list, rank5_list = [], []
        for num_epochs in range(25, 101, 25):
            base_model, model_path, conf_matrix_path = variables[used_model_name]
            used_model = base_model()

            rank1, rank5 = run_prediction(used_model, model_path.format(num_epochs), used_dataset, train_dataset)
            rank1_list.append(rank1)
            rank5_list.append(rank5)

        print("Rank-1 scores:")
        print(rank1_list)
        print("\nRank-5 scores:")
        print(rank5_list)
        results[used_model_name] = (rank1_list, rank5_list)

    # retrieve lists from results dict
    resnet18_rank1, resnet18_rank5 = results[resnet18]
    resnet152_rank1, resnet152_rank5 = results[resnet152]
    densenet161_rank1, densenet161_rank5 = results[densenet161]

    # rank-1 and rank-5 plots
    line_plot("plots/perfectly_detected_rank1_ylim0.5.png", resnet18_rank1, resnet152_rank1, densenet161_rank1, 0.5,
              "Rank-1 comparison of trained models on perfectly detected ears")
    line_plot("plots/perfectly_detected_rank5_ylim0.5.png", resnet18_rank5, resnet152_rank5, densenet161_rank5, 0.5,
              "Rank-5 comparison of trained models on perfectly detected ears")
    line_plot("plots/perfectly_detected_rank1_ylim1.png", resnet18_rank1, resnet152_rank1, densenet161_rank1, 1,
              "Rank-1 comparison of trained models on perfectly detected ears")
    line_plot("plots/perfectly_detected_rank5_ylim1.png", resnet18_rank5, resnet152_rank5, densenet161_rank5, 1,
              "Rank-5 comparison of trained models on perfectly detected ears")


