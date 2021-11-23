import os
import glob

import numpy as np
from PIL import Image


def create_header(csv, features: int):
    pixel = 'pixel'
    for i in range(features):
        csv.write(pixel + str(i) + ",")
    csv.write("label\n")


def get_label_from_emotion(emotion: str):
    if emotion == "angry":
        return 0
    elif emotion == "disgust":
        return 1
    elif emotion == "fear":
        return 2
    elif emotion == "happy":
        return 3
    elif emotion == "neutral":
        return 4
    elif emotion == "sad":
        return 5
    elif emotion == "surprise":
        return 6

    # unsupported emotion: should not get here
    return -1


def convert(csv_name: str, folder: str) -> None:
    print('converting images to csv...')
    with open(csv_name, 'w') as csv:
        # create the header labels
        create_header(csv, 48*48)
        # get all images
        for file in glob.iglob(folder + "*/*.jpg", recursive=True):
            # get the emotion name (folder name); map to int label
            emotion: str = os.path.basename(os.path.dirname(file))
            label: int = get_label_from_emotion(emotion)

            with Image.open(file) as img:
                array: np.ndarray = np.asarray(img).flatten()
                # write each pixel to the csv
                for i in range(len(array)):
                    csv.write(str(array[i]) + ",")
                # add the label at the end
                csv.write(str(label) + "\n")

    print('done.')


if __name__ == "__main__":
    # csv_name = "train.csv"
    # convert(csv_name, "train/")
    csv_name = "test.csv"
    convert(csv_name, "test/")
