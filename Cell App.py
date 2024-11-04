import streamlit as st
import cv2
import numpy as np

# Load the dataset (microscopic images of cells)
def load_image(image_file):
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    return img

# Preprocess the image
def preprocess_image(img):
    # Resize the image
    img = cv2.resize(img, (512, 512))
    
    # Normalize the image
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply Gaussian blur to remove noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    return img

# Apply thresholding to segment out cells from the background
def apply_thresholding(img):
    # Use Otsu's thresholding method
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

# Detect and count cells using connected component labeling
def detect_cells(thresh):
    # Use connected component labeling
    _, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, None, None, None, 8, cv2.CV_32S)
    
    # Get the number of cells (connected components)
    num_cells = len(np.unique(labels)) - 1
    
    return num_cells, labels, stats

# Measure cell dimensions using morphological operations
def measure_cells(labels, stats):
    # Create a list to store cell dimensions
    cell_dimensions = []
    
    # Iterate through each connected component (cell)
    for i in range(1, len(np.unique(labels))):
        # Get the bounding box coordinates
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        # Calculate the cell area
        area = w * h
        
        # Calculate the cell diameter (approximate)
        diameter = np.sqrt(4 * area / np.pi)
        
        # Append the cell dimensions to the list
        cell_dimensions.append((area, diameter))
    
    return cell_dimensions

# Main function
def main():
    st.title("Cell Counter")
    st.subheader("Upload your microscopic image of cells to count and measure them")

    image_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

    if image_file is not None:
        # Load the image
        img = load_image(image_file)
        
        # Preprocess the image
        img = preprocess_image(img)
        
        # Apply thresholding
        thresh = apply_thresholding(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        
        # Detect and count cells
        num_cells, labels, stats = detect_cells(thresh)
        
        # Measure cell dimensions
        cell_dimensions = measure_cells(labels, stats)
        
        # Display the results
        st.write(f"Number of cells: {num_cells}")
        st.write("Cell dimensions:")
        for i, (area, diameter) in enumerate(cell_dimensions):
            st.write(f"Cell {i+1}: Area = {area:.2f}, Diameter = {diameter:.2f}")
        
        # Display the original image and the thresholded image
        st.image(img, caption="Original Image")
        st.image(thresh, caption="Thresholded Image")

if __name__ == "__main__":
    main()