import os

# hyperparameters
image_width = 64
image_height = 64
sequence_length = 20
dataset_dir = 'UCF50'

# CLASSES
class_list = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]
# class_list = [name for name in os.listdir("./dataset/{}/".format(dataset_dir))]