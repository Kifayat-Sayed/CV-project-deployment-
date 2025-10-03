import os
import uuid
import numpy as np
import cv2
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
import pydicom

# -----------------------
# Flask App Initialization
# -----------------------
app = Flask(__name__)

# File Upload Configuirations
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

# -----------------------
# Loading Models
# -----------------------
def load_classification_model():
    """
    to load our pneumonia classification model.
      - If our entire model is saved as  (model.save), thn use load_model()
    """
    try:
        base_model = VGG19(include_top=False, input_shape=(128, 128, 3))
        x = base_model.output
        flat = Flatten()(x)
        fc1 = Dense(4608, activation='relu')(flat)
        drop_out = Dropout(0.2)(fc1)
        fc2 = Dense(1152, activation='relu')(drop_out)
        output = Dense(3, activation='softmax')(fc2)

        model = Model(base_model.inputs, output)
        model.load_weights('vgg_unfrozen.h5')  # TODO: update with your final model file
        print(" Classification model loaded succesdssfully")
        return model
    except Exception as e:
        print(f" Error loading classification model: {e}")
        return None

def load_detection_model():
    """
    Load trained detection model (future step).
    For now, if no detection model exists, return None.
    """
    try:
        detection_model = load_model('pneumonia_detection_model.h5')
        print(" Detection model loaded successfully")
        return detection_model
    except:
        print(" No detection model found. Using fallback contour method.")
        return None

classification_model = load_classification_model()
detection_model = load_detection_model()

# -----------------------
# Helper Functions
# -----------------------
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_className(classNo):
    """Mapping of prediction index → human-readable class"""
    class_names = {
        0: "Normal",
        1: "Pneumonia",
        2: "Not Normal - No Lung Opacity"
    }
    return class_names.get(classNo, "Unknown")

def preprocess_image(img_path):
    """Read image (JPG/PNG/DCM), resize, normalize"""
    try:
        ext = os.path.splitext(img_path)[1].lower()

        if ext == ".dcm":
            dcm = pydicom.dcmread(img_path)
            image = dcm.pixel_array
            # Convert grayscale → RGB (since VGG expects 3 channels)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (128, 128))
        image = image / 255.0
        return np.expand_dims(image, axis=0)

    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_classification(img_path):
    """Predict pneumonia class and confidence"""
    try:
        input_img = preprocess_image(img_path)
        if input_img is None:
            return None, None

        predictions = classification_model.predict(input_img, verbose=0)
        result_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions) * 100)

        return result_class, confidence
    except Exception as e:
        print(f"Error during classification: {e}")
        return None, None

def detect_pneumonia_region(img_path, predicted_class):
    """
    Detect pneumonia region:
      - If ddetection model exists → use it.
      - Else fallback then to contour-based method.
    """
    try:
        if predicted_class != 1:  # Only detect if pneumonia
            return None

        image = cv2.imread(img_path)
        if image is None:
            return None

        # TODO: Replace this with RCNN inference once implemented
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        if w * h < 1000:  # ignore small regions
            return None

        output_image = image.copy()
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(output_image, "Pneumonia Region", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        name, ext = os.path.splitext(img_path)
        output_path = f"{name}_annotated{ext}"
        cv2.imwrite(output_path, output_image)

        return output_path
    except Exception as e:
        print(f"Error during detection: {e}")
        return None

# -----------------------
# Routes
# -----------------------
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No fil;e uploaded", prediction=None)

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error="No file selected", prediction=None)

    if not allowed_file(file.filename):
        return render_template('index.html',
                               error="Invalid file type. Please upload PNG, JPG, JPEG, or DCM",
                               prediction=None)

    try:
        # Unique filename to avoid overwrite
        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Classification
        result_class, confidence = predict_classification(file_path)
        if result_class is None:
            return render_template('index.html', error="Error processing image", prediction=None)

        class_name = get_className(result_class)

        # Detection (only if pneumonia)
        annotated_path = None
        if result_class == 1:
            annotated_path = detect_pneumonia_region(file_path, result_class)

        original_image = url_for('static', filename=f'uploads/{filename}')
        annotated_image = None
        if annotated_path:
            annotated_filename = os.path.basename(annotated_path)
            annotated_image = url_for('static', filename=f'uploads/{annotated_filename}')

        return render_template('index.html',
                               prediction=class_name,
                               confidence=round(confidence, 2),
                               original_image=original_image,
                               annotated_image=annotated_image,
                               error=None)

    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html',
                               error=f"An error occurred: {str(e)}",
                               prediction=None)

@app.route('/about')
def about():
    return render_template('about.html')

# -----------------------
# Main
# -----------------------
if __name__ == '__main__':
    print("Server starting at http://127.0.0.1:5000/")
    app.run(debug=True, host='0.0.0.0', port=5000)
