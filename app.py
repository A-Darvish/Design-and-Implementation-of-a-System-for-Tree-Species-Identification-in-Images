from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import model # Replace with your model import

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Prediction
        prediction = model.predict(file_path)  # Replace with your model's prediction method
        return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(port=3000, debug=True, use_reloader=True, reloader_type='stat')
