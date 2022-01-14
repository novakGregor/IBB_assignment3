from collections import defaultdict
from shutil import copy2
import os


"""
Script for copying perfectly image dataset to folder format, expected by Pytorch, ie.:
    for training dataset:
        <path_to_dataset>/train/<class_id>/<image_name>
    for testing dataset:
        <path_to_dataset>/test/<class_id>/<image_name>
"""


# CSV file with class ids by image
path_to_ids = "data/perfectly_detected_ears/annotations/recognition/ids.csv"

# read CSV file and copy image filenames into dicts (key: class id; item: image filename)
train_images_by_class = defaultdict(list)
test_images_by_class = defaultdict(list)
with open(path_to_ids, "r") as f:
    for line in f:
        image_path, class_id = line.strip().split(",")
        if image_path.startswith("test"):
            test_images_by_class[class_id].append(image_path)
        else:
            train_images_by_class[class_id].append(image_path)

# where new dataset will be copied to
pytorch_root_path = "data/perfectly_detected_ears/pytorch_format"
pytorch_train_path = os.path.join(pytorch_root_path, "train")
pytorch_test_path = os.path.join(pytorch_root_path, "test")

# iterate through all class ids
for class_id in train_images_by_class.keys():
    print("Current class:", class_id)
    train_class_folder = os.path.join(pytorch_train_path, class_id)
    test_class_folder = os.path.join(pytorch_test_path, class_id)

    # Create folders for current class if necessary, but only if there exists at least one image
    # for that class - otherwise PyTorch can't read dirs into dataset
    if train_images_by_class[class_id] and not os.path.exists(train_class_folder):
        os.makedirs(train_class_folder)
    if test_images_by_class[class_id] and not os.path.exists(test_class_folder):
        os.makedirs(test_class_folder)

    # iterate through train and test images of current class
    for files, folder_path in zip((train_images_by_class[class_id], test_images_by_class[class_id]),
                                  (train_class_folder, test_class_folder)):
        # copy each image to new destination
        for file in files:
            filename = file.split("/")[-1]
            original_filepath = os.path.join("data/perfectly_detected_ears", file)
            new_filepath = os.path.join(folder_path, filename)

            print("    Original file:", original_filepath)
            print("    New file:", new_filepath)
            print()
            copy2(original_filepath, new_filepath)
