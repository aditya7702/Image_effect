from flask import Flask, render_template, request, send_file, jsonify
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Effect Functions
def sketch_effect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inv_gray = 255 - gray
    # Reduce the blur kernel size for a darker effect
    blurred = cv2.GaussianBlur(inv_gray, (15, 15), 0)  # Smaller blur kernel for darker lines
    sketch = cv2.divide(gray, 255 - blurred, scale=200.0)  # Adjust scale for more contrast
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def pencil_effect(image):
    # Create a pencil sketch with a lighter shade factor for a softer effect
    gray_sketch, color_sketch = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.1)  # Higher shade_factor for lighter effect
    
    # Blend with the original to retain some facial details, adjust weights as needed
    lighter_pencil_effect = cv2.addWeighted(color_sketch, 0.7, image, 0.3, 0)

    return lighter_pencil_effect




def pastel_effect(image):
    # Step 1: Apply bilateral filtering to smooth colors while keeping edges sharp
    smoothed = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 2: Use edge-preserving filter to further enhance the softness
    soft = cv2.edgePreservingFilter(smoothed, flags=1, sigma_s=60, sigma_r=0.4)

    # Step 3: Blend the original with the softened version for a pastel look
    pastel = cv2.addWeighted(image, 0.5, soft, 0.5, 0)

    return pastel


def crayon_effect(image):
    crayon = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.3)
    return crayon

def paper_effect(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a slight Gaussian blur for a smoother paper look
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Use adaptive thresholding with a higher `C` value to make it lighter
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, blockSize=11, C=15)

    # Blend with a higher weight for the original grayscale image to make it lighter
    paper_effect = cv2.addWeighted(gray, 0.6, thresh, 0.4, 0)

    # Convert back to BGR for consistent output
    return cv2.cvtColor(paper_effect, cv2.COLOR_GRAY2BGR)


def paris_effect(image):
    return cv2.applyColorMap(image, cv2.COLORMAP_OCEAN)

def santorini_effect(image):
    return cv2.applyColorMap(image, cv2.COLORMAP_SPRING)

def venice_effect(image):
    return cv2.applyColorMap(image, cv2.COLORMAP_SUMMER)

def paint_effect(image):
    paint = cv2.edgePreservingFilter(image, flags=1, sigma_s=60, sigma_r=0.4)
    return paint

# Add more effects here as needed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    effect = request.form.get('effect')
    
    if file and effect:
        # Convert file to OpenCV format
        img = Image.open(file.stream).convert('RGB')
        img_np = np.array(img)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Apply selected effect
        if effect == "Sketch":
            processed_img = sketch_effect(img_np)
        elif effect == "Pencil":
            processed_img = pencil_effect(img_np)
        elif effect == "Pastel":
            processed_img = pastel_effect(img_np)
        elif effect == "Crayon":
            processed_img = crayon_effect(img_np)
        elif effect == "Paper":
            processed_img = paper_effect(img_np)
        elif effect == "Paris":
            processed_img = paris_effect(img_np)
        elif effect == "Santorini":
            processed_img = santorini_effect(img_np)
        elif effect == "Venice":
            processed_img = venice_effect(img_np)
        elif effect == "Paint":
            processed_img = paint_effect(img_np)
        else:
            processed_img = img_np  # Default to original if no effect chosen
        
        # Convert processed image back to send as a file
        _, buffer = cv2.imencode('.jpg', processed_img)
        file_bytes = io.BytesIO(buffer)
        return send_file(file_bytes, mimetype='image/jpeg')
    
    return 'Error: No file or effect selected.'

if __name__ == '__main__':
    app.run(debug=True)
