import os, cv2, errno
from darkflow.net.build import TFNet
import numpy as np
import time

options = {
    'model': 'cfg/tiny-yolo-sound-overfit-test.cfg',
    'load': -1,
    'threshold': 0,
    'gpu': 0.7
}

tfnet = TFNet(options)

TEST_PATH = 'C:/Users/VESELINOVVLADISLAVSA/Documents/GitHub/sound-detection/data/overfit/'
PREDICTIONS_PATH = './predictions/'
LABELS = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

# Get the audio directory path, place yours here
test_path = os.path.dirname(TEST_PATH)

file_names = []
image_paths = []
xml_paths = []

slash = '\\' if os.name == 'nt' else '/'

for root, sub_dirs, files in os.walk(test_path):
    for file in files:
        if file.endswith('.jpg'):
            # file_names.append(file)
            image_paths.append(os.path.join(root, file))

            # folder_name = root.split('\\')[-1]

        elif file.endswith('.xml'):
            xml_paths.append(os.path.join(root, file))

image_paths = sorted(image_paths)
xml_paths = sorted(xml_paths)

# Create prediction directory  if necessary
prediction_path = os.path.dirname(os.path.realpath(__file__)) + '/../predictions/'

try:
    os.makedirs(prediction_path)
except OSError as exception:
    if (exception.errno != errno.EEXIST):
        raise

np.random.seed(1234)
colors = [tuple(255 * np.random.rand(3)) for label in range(len(LABELS))]

for image_path, xml_path in zip(image_paths, xml_paths):
    img = cv2.imread(image_path)
    predictions = tfnet.return_predict(img)
    
    for prediction in predictions:
        top_left = (prediction['topleft']['x'], prediction['topleft']['y'])
        bottom_right = (prediction['bottomright']['x'], prediction['bottomright']['y'])

        label = prediction['label']
        confidence = prediction['confidence']

        label_index = LABELS.index(label)
        img = cv2.rectangle(img, top_left, bottom_right, colors[label_index], 3)

        text = '{}: {:.0f}%'.format(label, confidence * 100)
        img = cv2.putText(img, text, top_left, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    file_name =  PREDICTIONS_PATH + image_path.split(slash)[-1]
    cv2.imwrite(file_name, img)
