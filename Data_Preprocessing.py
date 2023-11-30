import os
import glob
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import cv2
import xml.etree.ElementTree as ET

import json

with open("config.json") as json_data_file:
    data = json.load(json_data_file)

image_size = data['data_preprocessing']['image_size']

def read_image_and_read_label(folder_path):
    xml_paths = folder_path
    txt_paths = folder_path + "/TXT"

    folder_out = folder_path + "/BUFFER"
    if not (os.path.isdir(folder_out)):
        os.mkdir(folder_out)

    txtFire_paths = glob.glob(txt_paths + "/*.txt")
    txtFire_paths = [path.replace("\\", "/") for path in txtFire_paths]

    for label_path in txtFire_paths:
        image_path = (
            xml_paths + label_path.split(txt_paths)[1].split(".txt")[0] + ".jpg"
        )

        image_name = label_path.split(txt_paths)[1].split(".txt")[0].split("/")[1]
        label_class = []

        with open(label_path) as f:
            for row in f.readlines():
                label_class.append(row.replace("\n", "").split(" "))

        image = cv2.imread(image_path, 0)
        image_binarization = binarization(image, 50, -0.2)
        for idx, label_class in enumerate(label_class):
            image_cropping, class_name = cropping(image_binarization, label_class)
            image_resizing = resizing(image_cropping, image_size)

            out_fname = class_name + "_" + image_name + "-" + str(idx + 1) + ".jpg"
            image_new_path = folder_out + "/" + out_fname
            cv2.imwrite(image_new_path, image_resizing)

    create_folder(folder_path)


def binarization(image, windows= 50, k = -0.2):
    # # Niblack
    image = image.astype(np.float32)
    mean = cv2.blur(image, (windows, windows))
    mean_square = cv2.blur(image**2, (windows, windows))
    std_dev = np.sqrt(mean_square - mean**2)
    threshold = mean + k * std_dev
    binary_image = np.zeros_like(image)
    binary_image[image >= threshold] = 255
    binary_image = binary_image.astype(np.uint8)
    return binary_image

def cropping(image, label_data):
    class_name = label_data[0]
    xmin = int(label_data[1])
    xmax = int(label_data[2])
    ymin = int(label_data[3])
    ymax = int(label_data[4])
    imag_cropping = image[ymin:ymax, xmin:xmax]
    return imag_cropping, class_name


def resizing(image, size):
    image_resize = cv2.resize(image, (size, size))
    return image_resize


def create_folder(folder_path):
    folderImage_new_path = folder_path + "/IMAGE"
    if not (os.path.isdir(folderImage_new_path)):
        os.mkdir(folderImage_new_path)

    directory = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "31",
        "32",
        "33",
        "34",
        "35",
        "36",
        "37",
        "38",
        "39",
        "40",
        "41",
    ]

    for i in range(42):
        path = os.path.join(folderImage_new_path + "/" + str(directory[i]))
        if not (os.path.isdir(path)):
            os.mkdir(path)

    change_location_image(folder_path)


def change_location_image(folder_path):
    folderImage_new_path = folder_path + "/IMAGE"
    img_folder = folder_path + "/BUFFER"

    image_files = [
        f for f in os.listdir(img_folder) if f.endswith(".jpg") or f.endswith(".png")
    ]
    for file_name in image_files:
        new_dir = (
            folderImage_new_path
            + "/"
            + str(int(file_name.split("_")[0].split("LN0")[1]) - 1)
        )
        old_path = os.path.join(img_folder, file_name)
        new_path = os.path.join(new_dir, file_name)
        shutil.move(old_path, new_path)

    os.rmdir(img_folder)

    image_to_csv(folder_path, image_size)


def image_to_csv(folder_path, size):
    img_folder = folder_path + "/IMAGE"

    csv_out = "./DATASET"
    if not (os.path.isdir(csv_out)):
        os.mkdir(csv_out)

    columnNames = list()

    for i in range(1, (size * size) + 1):
        pixel = "pixel" + str(i)
        columnNames.append(pixel)

    columnNames.append("class")

    train_dataB = pd.DataFrame(columns=columnNames)
    num_images = 0

    c = 1
    for folder in os.listdir(img_folder):
        dir = img_folder + "/" + str(folder)
        print("Folder:", c)
        c += 1
        for file in tqdm(os.listdir(dir)):
            img = Image.open(os.path.join(dir, file))
            img.load()
            imgdata = np.asarray(img, dtype="int32")
            dataB = []
            for y in range(size):
                for x in range(size):
                    dataB.append(imgdata[y][x])
            dataB.append(str(folder))
            train_dataB.loc[num_images] = dataB
            num_images += 1

    train_dataB.to_csv(csv_out + "/" + folder_path.split("/")[1] + ".csv", index=False)

    remove_folder(folder_path)


def remove_folder(folder_path):
    folder_path_remove = [folder_path + "/IMAGE", folder_path + "/TXT" ]
    for i in range(len(folder_path_remove)):
        shutil.rmtree(folder_path_remove[i])

if __name__ == "__main__":
    folder_path = "roboflowXML_data/windows_test"
    print("START Data_Preprocessing")
    read_image_and_read_label(folder_path)
    print("END Data_Preprocessing")
