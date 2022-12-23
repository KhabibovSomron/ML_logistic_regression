import logging
import datetime
from data import Data
from show import Show

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
start_time = datetime.datetime.now()

data = Data()
show = Show()

show.render_images(data.CLASSES, data.DATASET_PATH)

data_frame = data.create_data_frame()
data_frame = data.remove_duplicates(data_frame)

classes_images_counts = data.check_classes_balance(data_frame)
show.show_classes_histogram(classes_images_counts, data.CLASSES)
data_frame = data.shuffle_data(data_frame)

x_train, y_train, x_test, y_test, *_ = data.split_dataset_into_subsamples(data_frame)

test_scores = data.get_logistic_regression(x_train, y_train, x_test, y_test)
show.show_result_plot(test_scores, data.TRAIN_SIZES)

end_time = datetime.datetime.now()
logging.info(end_time - start_time)
