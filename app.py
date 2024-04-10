from flask import Flask, render_template, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect
from flask_cors import CORS
from PIL import Image as pilImage
import io
import base64
import os
import json
import yaml
import csv
from pascal_voc_writer import Writer
import shutil
from image_padding import ImagePadding
from datetime import datetime


app = Flask(__name__)
# app.config["SQLALCHEMY_DATABASE_URI"] = 'mysql+pymysql://admin01:symboldetection@db/symboldetection'
app.config["SQLALCHEMY_DATABASE_URI"] = 'mysql+pymysql://admin01:symboldetection@localhost:3306/symboldetection'
db = SQLAlchemy(app)

class1 = ["田地", "草地", "荒地", "墓地", "樹林", "竹林", "旱地", "茶畑"]
class2 = ["果園", "茶畑", "桑畑", "沼田", "水田", "乾田", "荒地", "樹林椶櫚科", "竹林", "樹林鍼葉", "樹林濶葉", "草地"]
classes = [class1, class2]

class User(db.Model):
    __tablename__ = "users"
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False) 
    role = db.Column(db.String(80), nullable=False)  

    def __init__(self, username, password, role):
        self.username = username
        self.password = password
        self.role = role
    
    def __repr__(self):
        return '<User %r>' % self.username

class ParentImage(db.Model):
    __tablename__ = "parentimage"
    
    id = db.Column(db.Integer, primary_key=True)
    imagename = db.Column(db.String(80), unique=True, nullable=False)
    width = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    split_size = db.Column(db.Integer, nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.now)
    
    def __init__(self, imagename, width, height, split_size):
        self.imagename = imagename
        self.width = width
        self.height = height
        self.split_size = split_size
    
    def __repr__(self):
        return f'<ParentImage {self.id}>'

class Image(db.Model):
    __tablename__ = "image"
    
    id = db.Column(db.Integer, primary_key=True)
    imagename = db.Column(db.String(80), unique=True, nullable=False)
    parentimage_id = db.Column(db.Integer, db.ForeignKey('parentimage.id'), nullable=False)   
    parentimage = db.relationship('ParentImage', backref=db.backref('images', lazy=True))
    location_h = db.Column(db.Integer, nullable=False)
    location_w = db.Column(db.Integer, nullable=False) 
    padding_xmin = db.Column(db.Integer, nullable=False) 
    padding_ymin = db.Column(db.Integer, nullable=False) 
    padding_xmax = db.Column(db.Integer, nullable=False) 
    padding_ymax = db.Column(db.Integer, nullable=False) 
    is_labeled = db.Column(db.Boolean, default=False) 
    
    def __init__(self, imagename, parentimage_id, loaction_h, location_w, padding_xmin, padding_ymin, padding_xmax, padding_ymax):
        self.imagename = imagename
        self.parentimage_id = parentimage_id
        self.location_h = loaction_h
        self.location_w = location_w
        self.padding_xmin = padding_xmin
        self.padding_ymin = padding_ymin
        self.padding_xmax = padding_xmax
        self.padding_ymax = padding_ymax
    
    def __repr__(self):
        return f'<Image {self.id}>'

class LabelHistory(db.Model):
    __tablename__ = "labelhistory"
    
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey('image.id'), nullable=False)   
    image = db.relationship('Image', backref=db.backref('label_history', lazy=True))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)   
    label_date =  db.Column(db.DateTime, default=datetime.now)
    label_file_path = db.Column(db.String(80), unique=True, nullable=False)
    
    def __init__(self, image_id, user_id, label_date, label_file_path):
        self.image_id = image_id
        self.user_id = user_id
        self.label_date = label_date
        self.label_file_path = label_file_path
    
    def __repr__(self):
        return f'<LabelHistory {self.id}>'

class Labels(db.Model):
    __tablename__ = "labels"
    
    id = db.Column(db.Integer, primary_key=True)
    label_history_id = db.Column(db.Integer, db.ForeignKey('labelhistory.id'), nullable=False)
    label_history = db.relationship('LabelHistory', backref=db.backref('labels', lazy=True))
    class_set = db.Column(db.Integer, nullable=False)
    x_center = db.Column(db.Float, nullable=False)   
    y_center = db.Column(db.Float, nullable=False)   
    width = db.Column(db.Float, nullable=False)   
    height = db.Column(db.Float, nullable=False)   
    
    def __init__(self, label_history_id, class_set, x_center, y_center, width, height):
        self.label_history_id = label_history_id
        self.class_set = class_set
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
    
    def __repr__(self):
        return f'<Labels {self.id}>'
    
def create_tables():
    with app.app_context():
        if not inspect(db.engine).has_table("users"):
            db.create_all()
            admin = User(username="admin", password="adminps", role="admin")
            user01 = User(username="user01", password="user01ps", role="user")
            # user02 = User(username="user02", password="user02ps", role="user")
            
            db.session.add(admin)
            db.session.add(user01)
            # db.session.add(user02)
            db.session.commit()
        

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save_image', methods=['POST'])
def save_image():
    data = request.json
    image_data = data['image_data']
    image_name = data["image_name"]
    format_type = data["format_type"]
    set = data["class_set"]
    folder_path = data["folder_path"]
    username = data["username"]
    print(folder_path)
    if format_type == "yolo":
        image_folder_path = os.path.join("Annotations", f"Yolo_AnnotationsSet{set}", "images")
        if not os.path.exists(os.path.join("Annotations", f"Yolo_AnnotationsSet{set}")):
            os.mkdir(os.path.join("Annotations", f"Yolo_AnnotationsSet{set}"))
        if not os.path.exists(image_folder_path):
            os.mkdir(image_folder_path)
    elif format_type == "pascal":
        image_folder_path = os.path.join("Annotations", f"PASCAL_AnnotationsSet{set}")
        if not os.path.exists(image_folder_path):
            os.mkdir(image_folder_path)
    elif format_type == 'coco':
        image_folder_path = os.path.join("Annotations", f"COCO_AnnotationsSet{set}")
        if not os.path.exists(image_folder_path):
            os.mkdir(image_folder_path)
    elif format_type == "tensorflow":
        image_name = image_name.replace(',', "_")
        image_folder_path = os.path.join("Annotations", f"Tensorflow_AnnotationsSet{set}")
        if not os.path.exists(image_folder_path):
            os.mkdir(image_folder_path)
    elif format_type == "obb":
        image_folder_path = os.path.join("Annotations", f"OrientedObject_AnnotationsSet{set}", "images")
        if not os.path.exists(os.path.join("Annotations", f"OrientedObject_AnnotationsSet{set}")):
            os.mkdir(os.path.join("Annotations", f"OrientedObject_AnnotationsSet{set}"))
        if not os.path.exists(image_folder_path):
            os.mkdir(image_folder_path)
    
    output_path = os.path.join(image_folder_path, f'{image_name}.jpg')
    
    image_data = image_data.split(",")[1]  # Remove the prefix of Base64 encoding if present
    image_bytes = base64.b64decode(image_data)
    
    image = pilImage.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    image.save(output_path)
    
    with open(os.path.join("Annotations", 'log.txt'), "w") as f:
        f.write(f'User{username} labeled image {image_name} at {datetime.now()}')
    return jsonify({'message': 'Image Saved Successfully.'})

@app.route('/save_annotations', methods=['POST'])
def save_annotations():
    data = request.json
    image_name = data["image_name"]
    format_type = data["format_type"]
    yolo_labels = data["yolo_labels"]
    image_size = data["img_size"]
    set = data["class_set"]
    if format_type == 'yolo':
        label_folder_path = os.path.join("Annotations", f"Yolo_AnnotationsSet{set}", "labels")
        if not os.path.exists(label_folder_path):
            os.mkdir(label_folder_path)
        output_path = os.path.join(label_folder_path, f'{image_name}.txt')
        with open(output_path, 'w') as file:
            for label in yolo_labels:
                id, x, y, w, h, _ = label.values()
                file.write(f'{id} {x} {y} {w} {h} ' + '\n')
       
    elif format_type == 'pascal':
        voc_folder_path = os.path.join("Annotations", f"PASCAL_AnnotationsSet{set}")
        """ 
        # create pascal voc writer (image_path, width, height)
        writer = Writer('path/to/img.jpg', 800, 598)

        # add objects (class, xmin, ymin, xmax, ymax)
        writer.addObject('truck', 1, 719, 630, 468)
        writer.addObject('person', 40, 90, 100, 150)

        # write to file
        writer.save('path/to/img.xml')
        """
        
        writer = Writer(os.path.join(voc_folder_path, f'{image_name}.jpg'), image_size, image_size)
        for label in yolo_labels:
            id, x, y, w, h, _ = label.values()
    
            x *= image_size
            y *= image_size
            w *= image_size
            h *= image_size
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            writer.addObject(classes[set][id], x1, y1, x2, y2)
        writer.save(os.path.join(voc_folder_path, f'{image_name}.xml'))
        
    elif format_type == 'coco':
        coco_folder_path = os.path.join("Annotations", f"COCO_AnnotationsSet{set}")
        
        if os.path.exists(os.path.join(coco_folder_path, "annotations.json")):
            with open(os.path.join(coco_folder_path, "annotations.json"), "r") as f:
                old_annotations = json.load(f)
        else:
            with open(os.path.join("annotations_template", "annotations.json"), "r") as f:
                old_annotations = json.load(f)
        
        if os.path.exists(os.path.join(coco_folder_path, "coco_info.yaml")):
            with open(os.path.join(coco_folder_path, "coco_info.yaml"), "r") as f:
                coco_info = yaml.safe_load(f)
                MAX_IMAGE_ID = coco_info["MAX_IMAGE_ID"]
                MAX_ANNO_ID = coco_info["MAX_ANNO_ID"]
        else:
            with open(os.path.join("annotations_template", "coco_info.yaml"), "r") as f:
                coco_info = yaml.safe_load(f)
                MAX_IMAGE_ID = coco_info["MAX_IMAGE_ID"]
                MAX_ANNO_ID = coco_info["MAX_ANNO_ID"]
        MAX_IMAGE_ID += 1
        new_image_info = {
            "id": MAX_IMAGE_ID,
            "file_name": f'{image_name}.jpg',
            "width": image_size,
            "height": image_size
        }
        old_annotations["images"].append(new_image_info)
        
        new_anno_infos = []
        for label in yolo_labels:
            MAX_ANNO_ID += 1
            new_anno_info = yolo2coco(label, MAX_IMAGE_ID, MAX_ANNO_ID, image_size)
            new_anno_infos.append(new_anno_info)
        old_annotations["annotations"].extend(new_anno_infos)
        
        with open(os.path.join(coco_folder_path, "annotations.json"), "w") as f:
            json.dump(old_annotations, f)
        
        coco_info["MAX_IMAGE_ID"] = MAX_IMAGE_ID
        coco_info["MAX_ANNO_ID"] = MAX_ANNO_ID
        with open(os.path.join(coco_folder_path, "coco_info.yaml"), "w") as f:
            yaml.dump(coco_info, f)

    elif (format_type == 'tensorflow'):
        image_name = image_name.replace(',', "_")
        annotations_path = os.path.join("Annotations", f"Tensorflow_AnnotationsSet{set}")
        if not os.path.exists(os.path.join(annotations_path, "annotations.csv")):
            shutil.copy("./annotations_template/annotations.csv", annotations_path)
        new_annos = []
        for label in yolo_labels:
            new_anno = yolo2tensorflow(label, set, f'{image_name}.jpg', 480)
            new_annos.append(new_anno)
        with open(os.path.join(annotations_path, "annotations.csv"), "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(new_annos)
    elif (format_type == 'obb'):
        label_folder_path = os.path.join("Annotations", f"OrientedObject_AnnotationsSet{set}", "labels")
        if not os.path.exists(label_folder_path):
            os.mkdir(label_folder_path)
        output_path = os.path.join(label_folder_path, f'{image_name}.txt')
        with open(output_path, 'w') as file:
            for label in yolo_labels:
                id, x, y, w, h = label.values()
                file.write(f'{id} {x} {y} {w} {h} ' + '\n')
    return jsonify({'message': 'Success'})

def yolo2coco(yololabels, image_id, anno_id, img_size):
    id, x, y, w, h, _ = yololabels.values()
    
    x *= img_size
    y *= img_size
    w *= img_size
    h *= img_size
    x1 = x - w / 2
    y1 = y - h / 2
    new_annotation = {
        "id": anno_id,
        "image_id": image_id,
        "category_id": id,
        "bbox": [x1, y1, w, h],
        "area":w*h,
        "segmentation":[],
        "iscrowd":0
    }
    return new_annotation

def yolo2tensorflow(yololabels, class_set, image_name, img_size):
    id, x, y, w, h, _ = yololabels.values()
    
    x *= img_size
    y *= img_size
    w *= img_size
    h *= img_size
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    new_annotation = [image_name, img_size, img_size, classes[class_set][id], x1, y1, x2, y2]
    return new_annotation

@app.route('/validate_password', methods=['POST'])
def validate_password():
    data = request.json
    username_input = data["username"]
    password_input = data["password"]
    
    user = User.query.filter_by(username=username_input).first()
    if user and user.password == password_input:
        return jsonify({'success': True}), 200
    else:
        return jsonify({'success': False, 'message': 'Incorrect username or password'}), 401
    
@app.route('/add_parentimg', methods=['POST'])
def add_parentimg():
    data = request.json
    img_name = data["image_name"]
    width = data["width"]
    height = data["height"]
    split_size = data["split_size"]
    print(img_name, width, height, split_size)
    existing_parentimg = ParentImage.query.filter_by(
        imagename=img_name,
        width=width,
        height=height,
        split_size=split_size
    ).first()
    
    if existing_parentimg:
        return jsonify({'exists': True, 'parent_image_id': existing_parentimg.id}), 200
    else:
        parentimg = ParentImage(img_name, width, height, split_size)
        db.session.add(parentimg)
        db.session.commit()
        return jsonify({'exists': False, 'parent_image_id': parentimg.id}), 200

@app.route('/add_image', methods=['POST'])
def add_image():
    data = request.json
    images_info = data["images_info"]
    parent_image_id = data["parent_image_id"]


    return jsonify({'success': True}), 200
    
if __name__ == '__main__':
    create_tables()
    if not os.path.exists("./Annotations"): 
        os.mkdir("./Annotations")
        
    app.run(host='0.0.0.0', port=8000, debug=True)