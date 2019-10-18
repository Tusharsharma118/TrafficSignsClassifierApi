import numpy as np
from flask import Flask, request, jsonify, render_template

import tensorflow as tf
import skimage
import math

app = Flask(__name__)
model = tf.keras.models.load_model('final_model.h5')


def prepare_response(classes, error_text='An Error Occured'):
    if classes is not None:
        index = np.argmax(classes)
        if classes[0][index] > 0.6:
            return {
                'class': str(index),
                'class_desc': predictions[index],
                'confidence': math.floor(classes[0][index] * 100)
            }
        else:
            return {
                'class': 'None',
                'class_desc': 'Traffic Sign not present',
                'confidence': math.floor(classes[0][np.argmax(classes)] * 100)
                }
    else:
        return {
                'class': 'Error',
                'class_desc': error_text,
                'confidence': ''
                }


predictions = {
    0: 'Warning for a bad road surface',
    1: 'Warning for a speed bump',
    2: 'Warning for a slippery road surface',
    3: 'Warning for a curve to the left',
    4: 'Warning for a curve to the right',
    5: 'Warning for a double curve, first left then right',  # Merge Classes 5 & 6 later
    6: 'Warning for a double curve, first left then right',
    7: 'Watch out for children ahead',
    8: 'Watch out for  cyclists',
    9: 'Watch out for cattle on the road',
    10: 'Watch out for roadwork ahead',
    11: 'Traffic light ahead',
    12: 'Watch out for railroad crossing with barriers ahead',
    13: 'Watch out ahead for unknown danger',
    14: 'Warning for a road narrowing',
    15: 'Warning for a road narrowing on the left',
    16: 'Warning for a road narrowing on the right',
    17: 'Warning for side road on the right',
    18: 'Warning for an uncontrolled crossroad',
    19: 'Give way to all drivers',
    20: 'Road narrowing, give way to oncoming drivers',
    21: 'Stop and give way to all drivers',
    22: 'Entry prohibited (road with one-way traffic)',
    23: 'Cyclists prohibited',
    24: 'Vehicles heavier than indicated prohibited',
    25: 'Trucks prohibited',
    26: 'Vehicles wider than indicated prohibited',
    27: 'Vehicles higher than indicated prohibited',
    28: 'Entry prohibited',
    29: 'Turning left prohibited',
    30: 'Turning right prohibited',
    31: 'Overtaking prohibited',
    32: 'Driving faster than indicated prohibited (speed limit)',
    33: 'Mandatory shared path for pedestrians and cyclists',
    34: 'Driving straight ahead mandatory',
    35: 'Mandatory left',
    36: 'Driving straight ahead or turning right mandatory',
    37: 'Mandatory direction of the roundabout',
    38: 'Mandatory path for cyclists',
    39: 'Mandatory divided path for pedestrians and cyclists',
    40: 'Parking prohibited',
    41: 'Parking and stopping prohibited',
    42: '',
    43: '',
    44: 'Road narrowing, oncoming drivers have to give way',
    45: 'Parking is allowed',
    46: 'parking for handicapped',
    47: 'Parking for motor cars',
    48: 'Parking for goods vehicles',
    49: 'Parking for buses',
    50: 'Parking only allowed on the sidewalk',
    51: 'Begin of a residential area',
    52: 'End of the residential area',
    53: 'Road with one-way traffic',
    54: 'Dead end street',
    55: '',
    56: 'Crossing for pedestrians',
    57: 'Crossing for cyclists',
    58: 'Parking exit',
    59: 'Information Sign : Speed bump',
    60: 'End of the priority road',
    61: 'Begin of a priority road'
}


@app.route("/upload", methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            # print('NO FILE FOUND!')
            return jsonify(prepare_response(None, error_text='Image Upload Error!'))
        file = request.files['file']
        if file.filename == '':
            # print('NO FILE SELECTED')
            return jsonify(prepare_response(None, error_text='File Invalid'))
        if file:
            predict_image = skimage.io.imread(file)
            print('Raw File Shape: ', predict_image.shape)
            if 'png' in file.filename.lower():
                if len(predict_image.shape) <= 2:
                    return jsonify(prepare_response(None, error_text='Image should be either a colored JPG or PNG'))
                predict_image = skimage.color.rgba2rgb(predict_image)
            print('Input File Shape: ', predict_image.shape)
            predict_image128x128 = skimage.transform.resize(predict_image, (128, 128))
            predict_image128x128 = np.array(predict_image128x128)
            print(predict_image128x128.shape)
            predict_image128x128 = np.expand_dims(predict_image128x128, axis=0)
            print(predict_image128x128.shape)
            classes = model.predict(predict_image128x128)
            # print(classes)
            # filename = secure_filename(file.filename)
            # final_path = os.path.join(app.config['UPLOAD_FOLDER'])
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return jsonify(prepare_response(classes))
    else:
        return 'NOT POST'


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
