# 🎯 Military Drone Detection System

A sophisticated computer vision system designed for military applications to detect and classify drones and birds using the YOLO (You Only Look Once) architecture. This system enables real-time detection and classification of aerial objects, particularly useful for nighttime surveillance and security operations.

## 🚀 Features

- **Real-time Detection**: Fast and efficient detection of aerial objects
- **Object Classification**: Accurately distinguishes between drones and birds
- **Night Vision Support**: Compatible with night vision camera feeds
- **Confidence Scoring**: Provides confidence levels for each detection
- **Visual Indicators**:
  - Green bounding boxes for detected objects
  - Pink centroids for object tracking
  - Confidence percentage display
- **User-friendly Interface**: Streamlit-based GUI for easy operation

## 💻 Technology Stack

- **Deep Learning Framework**: PyTorch
- **Object Detection**: YOLO (YOLOv8)
- **GUI Framework**: Streamlit
- **Image Processing**: OpenCV
- **Data Manipulation**: NumPy
- **Image Handling**: Pillow (PIL)

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/military-drone-detection.git
cd military-drone-detection
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 🛠️ Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Use the sidebar to:
   - Select a pre-trained model
   - Choose an input image
   - Click "Run Detection" to process the image

## 📊 Model Training

- **Dataset**: Custom synthetic dataset created specifically for military drone detection
- **Classes**: 
  - Drones (Class 0)
  - Birds (Class 1)
- **Training Method**: Transfer learning on YOLOv8 architecture

## 🎯 Detection Output

The system provides:
- Visual bounding boxes around detected objects
- Classification labels (drone/bird)
- Confidence scores for each detection
- Centroid points for object tracking

## 📁 Project Structure

```
military-drone-detection/
├── app.py                 # Main Streamlit application
├── model/
│   └── best.pt           # Trained YOLO model
├── synthetic_data/       # Training and test images
│   ├── image_9.jpg
│   ├── image_15.jpg
│   └── synthetic_49.jpg
└── requirements.txt      # Project dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the [MIT License](LICENSE).

## 🔗 Contact

For any queries regarding the project, please open an issue in the repository.

## 🌟 Acknowledgments

- YOLOv8 for the object detection architecture
- Synthetic data generation tools and libraries
- Military personnel for domain expertise and requirements

---

