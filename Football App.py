import streamlit as st
import cv2
import numpy as np

# Load the image
def load_image(image_path):
    image = cv2.imread(image_path)
    return image

# Preprocess the image
def preprocess_image(image):
    # Resize the image
    resized_image = cv2.resize(image, (800, 600))
    
    # Apply Gaussian blur to remove noise
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
    
    # Apply thresholding to segment out the field from the background
    _, thresh_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return thresh_image

# Detect edges using Canny edge detector
def detect_edges(image):
    edges = cv2.Canny(image, 50, 150)
    return edges

# Detect lines using Hough transform
def detect_lines(edges):
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    return lines

# Calculate field dimensions and orientation
def calculate_field_dimensions(lines):
    # Assuming the field is a rectangle, calculate the length, width, and area
    length = 0
    width = 0
    area = 0
    
    # Iterate through the detected lines and calculate the field dimensions
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        
        # Calculate the length and width of the field
        length = max(length, abs(x2 - x1))
        width = max(width, abs(y2 - y1))
        
    area = length * width
    
    return length, width, area

# Streamlit app
st.title("Football Field Detection and Measurement")

# Load the image
image_path = st.file_uploader("Upload an image of a football field", type=["jpg", "png"])
if image_path:
    image = load_image(image_path)
    st.image(image, caption="Original Image")
    
    # Preprocess the image
    thresh_image = preprocess_image(image)
    st.image(thresh_image, caption="Preprocessed Image")
    
    # Detect edges
    edges = detect_edges(thresh_image)
    st.image(edges, caption="Edges")
    
    # Detect lines
    lines = detect_lines(edges)
    st.image(lines, caption="Lines")
    
    # Calculate field dimensions and orientation
    length, width, area = calculate_field_dimensions(lines)
    st.write(f"Length: {length}, Width: {width}, Area: {area}")