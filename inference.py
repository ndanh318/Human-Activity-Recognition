import cv2
import numpy as np
from collections import deque

from keras.models import load_model

from src.config import *


def predict_video(video_path, output_file_path, sequence_length, model):
    video_reader = cv2.VideoCapture(video_path)

    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    frame_queue = deque(maxlen=sequence_length)

    predicted_class_name = ''

    while video_reader.isOpened():
        success, frame = video_reader.read()
        if not success:
            break

        resized_frame = cv2.resize(frame, (image_height, image_width))
        nomalized_frame = resized_frame / 255

        frame_queue.append(nomalized_frame)

        if len(frame_queue) == sequence_length:
            predicted_labels_probabilities = model.predict(np.expand_dims(frame_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = class_list[predicted_label]

        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        video_writer.write(frame)

    video_reader.release()
    video_writer.release()


if __name__ == '__main__':
    # paths
    input_video_path = 'video/Test Video.mp4'
    output_video_path_1 = './demo/ConvLSTM.mp4'
    output_video_path_2 = './demo/LRCN.mp4'

    # Load model
    convlstm_model = load_model("model/convlstm_model.h5")
    convlstm_model.summary()

    lrcn_model = load_model("model/LRCN_model.h5")
    lrcn_model.summary()

    predict_video(input_video_path, output_video_path_1, sequence_length, convlstm_model)
    predict_video(input_video_path, output_video_path_2, sequence_length, lrcn_model)
