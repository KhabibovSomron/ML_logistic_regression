import os
import cv2
import logging
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class Data:

    def __init__(self):
        self.CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        self.DATASET_PATH = r"D:/lessons/Machine Learning/LR1/notMNIST_small/notMNIST_small"
        self.DATA_COLUMN_NAME = 'data'
        self.LABELS_COLUMN_NAME = 'labels'
        self.HASHED_DATA_COLUMN_NAME = 'data_bytes'
        self.BALANCE_BORDER = 0.85
        self.MAX_ITERATIONS_COUNT = 1000000
        self.TRAIN_SIZES = [50, 100, 1000, 10000, 50000]


    def get_class_data(self, folder_path):
        result_data = list()
        files = os.listdir(folder_path)
        for file in files:
            image_path = os.path.join(folder_path, file)
            img = cv2.imread(image_path)
            if img is not None:
                result_data.append(img.reshape(-1))

        return result_data


    def get_classes_images_counts(self, data_frame):
        classes_images_counts = list()
        for class_index in range(len(self.CLASSES)):
            labels = data_frame[self.LABELS_COLUMN_NAME]
            class_rows = data_frame[labels == class_index]
            class_count = len(class_rows)

            classes_images_counts.append(class_count)
            logging.info(f"Class {self.CLASSES[class_index]} contains {class_count} images")

        return classes_images_counts

    
    def check_classes_balance(self, data_frame):
        classes_images_counts = self.get_classes_images_counts(data_frame)

        max_images_count = max(classes_images_counts)
        avg_images_count = sum(classes_images_counts) / len(classes_images_counts)
        balance_percent = avg_images_count / max_images_count

    
        logging.info(f"Balance: {balance_percent:.3f}")
        if balance_percent > self.BALANCE_BORDER:
            logging.info("Classes are balanced")
        else:
            logging.info("Classes are not balanced")

        return classes_images_counts


    def create_data_frame(self):
        data = list()
        labels = list()
        for class_item in self.CLASSES:
            class_folder_path = os.path.join(self.DATASET_PATH, class_item)
            class_data = self.get_class_data(class_folder_path)

            data.extend(class_data)
            labels.extend([self.CLASSES.index(class_item) for _ in range(len(class_data))])

        data_frame = pandas.DataFrame({self.DATA_COLUMN_NAME: data, self.LABELS_COLUMN_NAME: labels})
        logging.info("Data frame is created")

        return data_frame


    def remove_duplicates(self, data):
        data_bytes = [item.tobytes() for item in data[self.DATA_COLUMN_NAME]]
        data[self.HASHED_DATA_COLUMN_NAME] = data_bytes
        data.sort_values(self.HASHED_DATA_COLUMN_NAME, inplace=True)
        data.drop_duplicates(subset=self.HASHED_DATA_COLUMN_NAME, keep='first', inplace=True)
        data.pop(self.HASHED_DATA_COLUMN_NAME)
        logging.info("Duplicates removed")

        return data

    def shuffle_data(self, data):
        data_shuffled = data.sample(frac=1, random_state=42)
        logging.info("Data shuffled")

        return data_shuffled

    def split_dataset_into_subsamples(self, data_frame):
        data = np.array(list(data_frame[self.DATA_COLUMN_NAME].values), np.float32)
        labels = np.array(list(data_frame[self.LABELS_COLUMN_NAME].values), np.float32)

        x_train, x_remaining, y_train, y_remaining = train_test_split(data, labels, train_size=0.8)
        x_valid, x_test, y_valid, y_test = train_test_split(x_remaining, y_remaining, test_size=0.5)
        logging.info("Data split")

        return x_train, y_train, x_test, y_test, x_valid, y_valid


    def get_logistic_regression(self, x_train, y_train, x_test, y_test):
        test_scores = list()
        for train_size in self.TRAIN_SIZES:
            logistic_regression = LogisticRegression(max_iter=self.MAX_ITERATIONS_COUNT)
            logistic_regression.fit(x_train[:train_size], y_train[:train_size])
            logging.info("Regression fit is completed")

            score = logistic_regression.score(x_test, y_test)
            logging.info("Score is calculated")
            test_scores.append(score)

        return test_scores