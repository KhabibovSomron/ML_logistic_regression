import math
from random import choice
import os
from matplotlib import pyplot as plt
import cv2
import logging
import numpy as np

class Show:

    def __init__(self):
        pass

    def generate_image_path(self, image_folder):
        file = choice(os.listdir(image_folder))
        image_path = os.path.join(image_folder, file)

        img = cv2.imread(image_path)
        return img

    def render_images(self, classes, dataset_path):
        fig =  plt.figure(figsize=(10, 10))
        rows = math.ceil(len(classes) / 3)
        columns = 3
        i = 1
        for item in classes:
            image_dir_path = os.path.join(dataset_path, item)
            fig.add_subplot(rows, columns, i)
            plt.imshow(self.generate_image_path(image_dir_path))   
            i += 1

        plt.show()

    def show_classes_histogram(self, classes_images_counts, classes):
        plt.figure()
        plt.bar(classes, classes_images_counts)
        plt.show()
        logging.info("Histogram shown")

    
    def show_result_plot(self, test_scores, train_sizes):
        test_scores_mean = np.mean(test_scores)
        print(test_scores_mean)
        test_scores_std = np.std(test_scores)

        plt.figure()
        plt.title('Learning curve')
        plt.xlabel('Training data size')
        plt.ylabel('Accuracy')
        plt.grid()

        plt.plot(train_sizes, test_scores, 'o-', color='g', label='Testing score')
        plt.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.25,
            color='g'
        )

        plt.show()
        logging.info("Plot shown")