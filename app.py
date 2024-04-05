from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import base64
from flask import request, jsonify
import os
from image_padding import ImagePadding
from datetime import datetime

app = Flask(__name__)
image_folder_path = os.path.join("Annotations", "images")
label_folder_path = os.path.join("Annotations", "labels")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save_image', methods=['POST'])
def save_image():
    data = request.json
    image_data = data['image_data']
    image_name = data['image_name']
    output_path = os.path.join(image_folder_path, f'{image_name}.jpg')
    
    image_data = image_data.split(",")[1]  # Remove the prefix of Base64 encoding if present
    image_bytes = base64.b64decode(image_data)
    
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    image.save(output_path)
    return jsonify({'message': 'Image Saved Successfully.'})

@app.route('/save_annotations', methods=['POST'])
def save_annotations():
    
    data = request.json
    image_name = data["image_name"]
    format_type = data["format_type"]
    yolo_labels = data["yolo_labels"]
    if format_type == 'yolo':
        output_path = os.path.join(label_folder_path, f'{image_name}.txt')
        with open(output_path, 'w') as file:
            for label in yolo_labels:
                id, x, y, w, h = label.values()
                file.write(f'{id} {x} {y} {w} {h} ' + '\n')
       
    elif format_type == 'pascal':
        pass
    elif format_type == 'coco':
        pass
    else:
        return jsonify({'message': 'Error'})
    return jsonify({'message': 'Success'})

def coco_format(yolo_labels, img_size, image_name):
    pass

def pascalvoc_format(yolo_labels, img_size, image_name):
    pass
    
if __name__ == '__main__':
    if not os.path.exists("./Annotations"): 
        os.mkdir("./Annotations")
    if not os.path.exists(image_folder_path):
        os.mkdir(image_folder_path)
    if not os.path.exists(label_folder_path):
        os.mkdir(label_folder_path)
        
    app.run(host='0.0.0.0', port=8000, debug=True)