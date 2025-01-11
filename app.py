import streamlit as st
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import os
from PIL import Image

st.set_page_config(page_title="Drone Detection System", layout="wide")

def detect_box(results):
    boxes = results[0].boxes
    bboxes = boxes.xyxy
    scores = boxes.conf
    classes = boxes.cls

    rois = []
    for index in range(len(boxes)):
        xmin = int(bboxes[index][0])
        ymin = int(bboxes[index][1])
        xmax = int(bboxes[index][2])
        ymax = int(bboxes[index][3])
        score = int(scores[index] * 100)
        class_id = int(classes[index])
        rois.append([xmin, ymin, xmax, ymax, class_id, score])
    return rois

def process_image(image, model_path):
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    model = YOLO(model_path)
    results = model(opencv_image, conf=.5, iou=.50)
    frame = results[0].orig_img

    rois = detect_box(results)
    
    class_labels = {0: "drone", 1: "bird"}
    
    for roi in rois:
        class_id = roi[4]
        if class_id in class_labels:
            bbox_message = class_labels[class_id]
            text_color, bbox_color, centroid_color = (255, 255, 255), (0, 255, 0), (255, 0, 255)
            
            centroid = (int((roi[0] + roi[2]) / 2), int((roi[1] + roi[3]) / 2))
            cv2.circle(frame, centroid, 5, centroid_color, -1)
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), bbox_color, thickness=3)
            
            (text_width, text_height), baseline = cv2.getTextSize(
                f"{bbox_message}-{roi[5]}", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            text_x, text_y = roi[0], roi[1] - text_height
            cv2.rectangle(
                frame,
                (text_x, text_y),
                (text_x + text_width, text_y + text_height + baseline),
                bbox_color,
                thickness=cv2.FILLED,
            )
            cv2.putText(
                frame,
                f"{bbox_message}-{roi[5]}",
                (text_x, text_y + text_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                text_color,
                2,
                lineType=cv2.LINE_AA,
            )
    
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def main():
    st.title("🎯 Military Drone Detection System")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model selection with simplified names
    model_paths = {
        "Model 1": r"C:\Users\asus\Desktop\drone-detection\model\best.pt"
    }
    
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        list(model_paths.keys())
    )
    
    # Image selection with simplified names
    image_paths = {
        "Image 1": r"C:\Users\asus\Desktop\drone-detection\synthetic_data\image_9.jpg",
        "Image 2": r"C:\Users\asus\Desktop\drone-detection\synthetic_data\image_15.jpg",
        "Image 3": r"C:\Users\asus\Desktop\drone-detection\synthetic_data\synthetic_49.jpg"
    }
    
    selected_image_name = st.sidebar.selectbox(
        "Select Image",
        list(image_paths.keys())
    )
    
    # Run button
    run_detection = st.sidebar.button("Run Detection")
    
    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
    
    # Always show input image
    input_image = Image.open(image_paths[selected_image_name])
    col1.header("Input Image")
    col1.image(input_image, use_container_width=True)
    
    # Only show detection when run button is pressed
    if run_detection:
        with st.spinner('Processing...'):
            processed_image = process_image(input_image, model_paths[selected_model_name])
            col2.header("Detected Objects")
            col2.image(processed_image, use_container_width=True)
    
    # Project Information Section
    st.markdown("---")
    st.header("📋 Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("System Capabilities")
        st.markdown("""
        - 🎯 **Real-time Drone Detection**: Advanced detection system capable of identifying drones in various conditions
        - 🦅 **Drone vs Bird Classification**: AI-powered discrimination between drones and birds
        - 🌙 **Night Vision Compatibility**: Designed to work with night vision cameras for 24/7 surveillance
        - 📊 **Confidence Scoring**: Provides confidence levels for each detection
        """)
    
    with col2:
        st.subheader("Technical Details")
        st.markdown("""
        - 🤖 **Model**: YOLO (You Only Look Once) architecture
        - 🎲 **Training Data**: Custom synthetic dataset created for military applications
        - 🎯 **Detection Features**:
          - Green boxes: Object boundaries
          - Pink dots: Object centroids
          - Percentage: Detection confidence
        - 🔍 **Real-time Processing**: Optimized for quick detection and response
        """)

if __name__ == "__main__":
    main()