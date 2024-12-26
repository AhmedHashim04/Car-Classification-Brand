# Car Classification 

## Overview
This project is a web application that uses deep learning to classify car images. Users can upload photos of cars, and the system will identify the car model using a pre-trained model ResNet50.
## Features
- Real-time car image upload and preview
- Instant car model prediction
- User-friendly interface
- Responsive design for all devices
## Technologies Used
- **Frontend:**
  - HTML5
  - CSS3
  - JavaScript (Vanilla)
- **Backend:**
  - Python
  - Flask
- **Machine Learning:**
  - PyTorch 
  - pre-trained model ResNet50.
## Project Structure
deep-learning-project/
├── app.py 
├── static/
│ ├── js/
│ │ └── deepcars.js # JavaScript functions
│ ├── css/
│ │ └── style.css # Styling
│ └── imgs/ # Static images
├── templates/
│ └── deepCars.html # Main HTML template
## Installation
- Download it
-  Run CMD
-   Run the application
bash
python app.py
- Open your browser and navigate to `http://localhost:5000`
- 
### Dataset Paths
Please update the paths in the `app.py` file to match your local environment:
- `train_path`: Path to the training dataset.
- `valid_path`: Path to the validation dataset.
- 
## Usage
1. Open the application in your web browser
2. Click on the upload button or drag and drop a car image
3. Wait for the model to process the image
4. View the prediction results




