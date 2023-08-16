from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from flask import request
import os
from predict import predict_model
# Set your Cloudinary credentials
# ==============================
from dotenv import load_dotenv

load_dotenv()

# Import the Cloudinary libraries
# ==============================
import cloudinary
import cloudinary.uploader
import cloudinary.api

# Import to format the JSON responses
# ==============================
import json

# Set configuration parameter: return "https" URLs by setting secure=True
# ==============================
config = cloudinary.config(secure=True)

# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = "static"

# Set the path to the local YOLOv5 model
path_to_model = 'weights/best.pt'


@app.route('/api/predict', methods=['POST'])
def predict_yolov5():
    image = request.files['file']
    if image:
        # Lưu file
        path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        if not os.path.exists(os.path.dirname(path_to_save)):
            os.makedirs(os.path.dirname(path_to_save))
        print("Save = ", path_to_save)
        image.save(path_to_save)
        url, data, filename = predict_model(weights=path_to_model,
                                            source=path_to_save)  # http://server.com/static/path_to_save
        var = filename.split('.')[0]
        cloudinary.uploader.upload(url, public_id=var, unique_filename=False, overwrite=True)
        # Build the URL for the image and save it in the variable 'srcURL'
        srcURL = cloudinary.CloudinaryImage(filename).build_url()
        data = {
            "image": srcURL,
            "data": data,
        }
        return jsonify(data)

    return 'Upload file to detect'
# Start Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6868')