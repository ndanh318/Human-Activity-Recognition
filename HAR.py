import os
import cv2
import random
import argparse
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

from src.config import *
from src.dataset import *
from src.model import *
from src.utils import *


def get_args():
    parser = argparse.ArgumentParser("Training ConvLSTM model")
    parser.add_argument("--dataset_dir", "-d", type=str, default="./dataset")
    parser.add_argument("--model", "-m", type=str, default="convlstm")
    parser.add_argument("--batch_size", "-b", type=int, default=64)
    parser.add_argument("--epochs", "-e", type=int, default=1000)
    parser.add_argument("--sequence_length", "-l", type=int, default=20)
    parser.add_argument("--image_saved", "-is", type=str, default="./images")
    parser.add_argument("--model_saved", "-s", type=str, default="./model")

    args = parser.parse_args()
    return args


def main(args):
    seed_constant = 27
    np.random.seed(seed_constant)
    random.seed(seed_constant)
    tf.random.set_seed(seed_constant)

    # preprocessing dataset
    features, labels, video_file_paths = create_dataset()
    one_hot_encoded_labels = to_categorical(labels)
    features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels,
                                                                                test_size=0.2, shuffle=True, random_state=42)

    # model
    if args.model == "convlstm":
        convlstm_model = create_convlstm_model()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min',restore_best_weights=True)
        convlstm_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])
        convlstm_model_training_history = convlstm_model.fit(features_train, labels_train,
                                                             epochs=args.epochs,
                                                             shuffle=True,
                                                             batch_size=args.batch_size,
                                                             validation_split=0.2,
                                                             callbacks=[early_stopping])

        # save model
        model_file_name = f"{args.model_saved}/convlstm_model.h5"
        convlstm_model.save(model_file_name)

        # visualize plot
        plot_metric(convlstm_model_training_history, 'loss', 'val_loss',
                    'Total Loss and Validation Loss', "ConvLSTM_loss")
        plot_metric(convlstm_model_training_history, 'accuracy', 'val_accuracy',
                    'Total Accuracy vs Total Validation Accuracy', "ConvLSTM_accuracy")

    elif args.model == "lrcn":
        LRCN_model = create_LRCN_model()
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)
        LRCN_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])
        LRCN_model_training_history = LRCN_model.fit(features_train, labels_train,
                                                     epochs=args.epochs,
                                                     batch_size=args.batch_size,
                                                     shuffle=True,
                                                     validation_split=0.2,
                                                     callbacks=[early_stopping_callback])

        # save model
        model_file_name = f"{args.model_saved}/LRCN_model.h5"
        LRCN_model.save(model_file_name)

        # visualize plot
        plot_metric(LRCN_model_training_history, 'loss', 'val_loss',
                    'Total Loss vs Total Validation Loss', "LRCN_loss")
        plot_metric(LRCN_model_training_history, 'accuracy', 'val_accuracy',
                    'Total Accuracy vs Total Validation Accuracy', "LRCN_accuracy")


if __name__ == '__main__':
    args = get_args()
    main(args)
