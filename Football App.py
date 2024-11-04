import streamlit as st
import cv2
import numpy as np

# Load the image
def load_image(image_path):
    image = cv2.imdecode(np.frombuffer(image_path.read(), np.uint8), cv2.IMREAD_COLOR)
    return image

# Preprocess the image
def preprocess_image(image):
    # Resize the image
    resized_image = cv2.resize(image, (800, 600))
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to remove noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply thresholding to segment out the object from the background
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

# Draw lines on a blank image
def draw_lines(lines, image_shape):
    blank_image = np.zeros(image_shape, dtype=np.uint8)
    if lines is not None:
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
            cv2.line(blank_image, (x1, y1), (x2, y2), 255, 2)
    return blank_image

# Calculate object dimensions and orientation
def calculate_object_dimensions(image):
    # Find contours in the image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize variables to store the object dimensions
    area = 0
    perimeter = 0
    shape = ""
    
    # Iterate through the contours
    for contour in contours:
        # Calculate the area of the contour
        contour_area = cv2.contourArea(contour)
        
        # If the contour area is larger than the current area, update the object dimensions
        if contour_area > area:
            area = contour_area
            
            # Calculate the perimeter of the contour
            perimeter = cv2.arcLength(contour, True)
            
            # Calculate the shape of the contour
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                shape = "Rectangle"
            elif len(approx) == 5:
                shape = "Pentagon"
            elif len(approx) > 5:
                shape = "Circle"
    
    # Calculate the dimensions of the shape
    if shape == "Rectangle":
        length = 0
        width = 0
        x, y, w, h = cv2.boundingRect(contour)
        length = w
        width = h
        area = length * width
        return shape, length, width, area
    elif shape == "Circle":
        radius = 0
        (x, y), radius = cv2.minEnclosingCircle(contour)
        area = np.pi * radius ** 2
        return shape, radius, area
    else:
        return shape, area

# Streamlit app
st.title("Area Detection & Measurement")

# Load the image
image_path = st.file_uploader("Upload an image", type=["jpg", "png"])
if image_path:
    image = load_image(image_path)
    st.image(image, caption="Original Image", channels="BGR")
    
    # Preprocess the image
    thresh_image = preprocess_image(image)
    st.image(thresh_image, caption="Preprocessed Image")
    
    # Detect edges
    edges = detect_edges(thresh_image)
    st.image(edges, caption="Edges")
    
    # Detect lines
    lines = detect_lines(edges)
    lines_image = draw_lines(lines, thresh_image.shape)
    st.image(lines_image, caption="Lines")
    
    # Calculate object dimensions and orientation
    shape, *dimensions = calculate_object_dimensions(thresh_image)
    if shape == "Rectangle":
        length, width, area = dimensions
        st.write(f"Shape: {shape}, Length: {length}, Width: {width}, Area: {area}")
    elif shape == "Circle":
        radius, area = dimensions
        st.write(f"Shape: {shape}, Radius: {radius}, Area: {area}")
    else:
        st.write(f"Shape: {shape}, Area: {dimensions[0]}")