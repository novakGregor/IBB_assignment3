from collections import defaultdict
from shutil import copy2
import os


"""
Script for copying detected ears on test dataset from assignment 2 to folder format, expected by Pytorch, ie.:
    <path_to_images>/<class_id>/<image_name>
"""


path_to_ids = "data/perfectly_detected_ears/annotations/recognition/awe-translation.csv"

images_by_class = defaultdict(list)
classes_by_images = defaultdict(str)

with open(path_to_ids, "r") as f:
    next(iter(f))
    next(iter(f))
    for line in f:
        split_line = line.strip().split(",")
        image_file = split_line[2]
        if image_file.startswith("train"):
            continue
        class_name = split_line[0]
        image_num = image_file.split("/")[-1][:4]
        images_by_class[class_name].append(image_num)
        classes_by_images[image_num] = class_name

for image in os.listdir("data/my_yolov5_detected_ears"):
    image_num = image[:4]
    image_class = classes_by_images[image_num]

    path_to_class_dir = "data/yolov5_cropped_pytorch/{}".format(image_class)
    if not os.path.exists(path_to_class_dir):
        os.makedirs(path_to_class_dir)

    old_image_path = "data/my_yolov5_detected_ears/{}".format(image)
    new_image_path = os.path.join(path_to_class_dir, image)
    copy2(old_image_path, new_image_path)
