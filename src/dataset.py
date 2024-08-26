import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from src.config import *


def frame_extraction(video_path):
    frame_list = []
    cap = cv2.VideoCapture(video_path)

    # Skip frame
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frame = max(int(video_frame_count / sequence_length), 1)

    for frame_counter in range(sequence_length):
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frame)
        success, frame = cap.read()
        if not success:
            break

        # Resize frame
        resized = cv2.resize(frame, (image_height, image_width))
        nomalized = resized / 255

        frame_list.append(nomalized)

    cap.release()
    return frame_list


def create_dataset():
    features = []  # Contain frames of the videos
    labels = []  # Contain indexes of class which associated with the videos
    video_file_paths = []  # Contain paths of the videos

    for class_index, class_name in enumerate(class_list):
        print(f'Create data of class: {class_name}, please wait...')

        # Get list of video which nemed in class name
        file_list = os.listdir(os.path.join("./dataset", dataset_dir, class_name))
        for file_name in file_list:
            # Get full video path
            video_file_path = os.path.join("./dataset", dataset_dir, class_name, file_name)
            frames = frame_extraction(video_file_path)

            if len(frames) == sequence_length:
                features.append(frames)
                labels.append(class_index)
                video_file_paths.append(video_file_path)

    # Convert list to array
    features = np.asarray(features)
    labels = np.array(labels)

    return features, labels, video_file_paths


if __name__ == '__main__':
    # visualize
    plt.figure(figsize=(10, 10))

    # Get class name
    all_class_names = os.listdir("../dataset/{}".format(dataset_dir))
    print(all_class_names)

    # Generate list of 20 random activities
    random_range = random.sample(range(len(all_class_names)), 20)
    for counter, random_index in enumerate(random_range, 1):
        selected_class_name = all_class_names[random_index]
        video_files_name_list = os.listdir(f'../dataset/{dataset_dir}/{selected_class_name}')
        selected_video_file_name = random.choice(video_files_name_list)

        # Visualize
        cap = cv2.VideoCapture(f'../dataset/{dataset_dir}/{selected_class_name}/{selected_video_file_name}')
        _, frame = cap.read()
        cap.release()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.putText(frame, selected_class_name, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # Display frame
        plt.subplot(5, 4, counter)
        plt.imshow(frame)
        plt.axis('off')

    plt.savefig("../images/{}".format("activities"), bbox_inches="tight")
    plt.tight_layout()
    plt.show()
