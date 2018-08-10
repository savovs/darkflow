import os, cv2, errno
from darkflow.net.build import TFNet
from xml.etree import ElementTree
import numpy as np
import time

options = {
    'model': 'cfg/tiny-yolo-sound.cfg',
    'load': -1,
    'threshold': 0,
    'gpu': 0.7
}

tfnet = TFNet(options)

TEST_PATH = 'C:/Users/VESELINOVVLADISLAVSA/Documents/GitHub/sound-detection/data/test/'
AP_PATH = 'C:/Users/VESELINOVVLADISLAVSA/Documents/GitHub/mAP/'
XML_PATH = 'C:/Users/VESELINOVVLADISLAVSA/Documents/GitHub/sound-detection/data/test/annotations/'
PREDICTIONS_PATH = './predictions/'
LABELS = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']


# Txt files for average precision evaluation
# https://github.com/Cartucho/mAP
def write_ground_truth_txt(read_path, write_path):
    result_string = ''
    if os.path.exists(read_path):
        dom = ElementTree.parse(read_path)
        names = dom.findall('object/name')

        x_mins = dom.findall('object/bndbox/xmin')
        y_maxes = dom.findall('object/bndbox/ymax')
        x_maxes = dom.findall('object/bndbox/xmax')
        y_mins = dom.findall('object/bndbox/ymin')
        
        for index, name in enumerate(names):
            result_string += '{} {} {} {} {}\n'.format(name.text, x_mins[index].text, y_maxes[index].text, x_maxes[index].text, y_mins[index].text)

    else:
        print('file doesn\'t exist')

    with open(write_path, 'w') as eval_ground_truth_text_file:
        eval_ground_truth_text_file.write(result_string)
        eval_ground_truth_text_file.close()

# Get the audio directory path
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

    if len(predictions) > 0:
        name = image_path.split(slash)[-1]
        xml_name = name.replace('.jpg', '.xml')
        txt_name = name.replace('.jpg', '.txt')

        # Save image and ground-truth boxes in evaluation project
        eval_images_path =  os.path.join(AP_PATH, 'images', name)
        cv2.imwrite(eval_images_path, img)
        
        read_xml_path = os.path.join(XML_PATH, xml_name)
        write_xml_path = os.path.join(AP_PATH, 'ground-truth', txt_name)
        write_ground_truth_txt(read_xml_path, write_xml_path)

        predicted_string = ''
        for prediction in predictions:
            top_left = (prediction['topleft']['x'], prediction['topleft']['y'])
            bottom_right = (prediction['bottomright']['x'], prediction['bottomright']['y'])

            label = prediction['label']
            confidence = prediction['confidence']

            left = prediction['topleft']['x']
            top = prediction['topleft']['y']
            right = prediction['bottomright']['x']
            bottom = prediction['bottomright']['y']

            predicted_string += '{} {} {} {} {} {}\n'.format(label, confidence, left, top, right, bottom)

            label_index = LABELS.index(label)
            img = cv2.rectangle(img, top_left, bottom_right, colors[label_index], 3)

            text = '{}: {:.0f}%'.format(label, confidence * 100)

            text_coordinates = (left, bottom)
            actual_text_coordinates = (left, bottom + 20)

            prediction_correct = label in image_path
            color  = (255, 255, 255) if prediction_correct else (0, 0, 255)
            img = cv2.putText(img, text, text_coordinates, cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)

            if not prediction_correct:
                img = cv2.putText(img, 'Actual: ' + name, actual_text_coordinates, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
        
        with open(os.path.join(AP_PATH, 'predicted', txt_name), 'w') as eval_predicted_text_file:
            eval_predicted_text_file.write(predicted_string)
            eval_predicted_text_file.close()

        # Save image with labels in ./predictions
        out_path =  PREDICTIONS_PATH + name
        cv2.imwrite(out_path, img)

