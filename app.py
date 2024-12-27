import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import json

# Set the device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to training and validation datasets
train_path = r".\dataset\Images\Train"
valid_path = r".\dataset\Images\Test"

# Check if the paths exist
if not os.path.exists(train_path) or not os.path.exists(valid_path):
    raise FileNotFoundError(
        f"Please update the dataset paths:\n"
        f"Train Path: {train_path}\n"
        f"Validation Path: {valid_path}"
    )
        
# Data transformations for training and validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomResizedCrop(224),  # Random crop
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet stats
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),  # Crop center of the image
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load datasets
train_dataset = ImageFolder(train_path, transform=data_transforms['train'])
valid_dataset = ImageFolder(valid_path, transform=data_transforms['valid'])

# DataLoader for efficient batch loading
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Number of classes in the dataset
num_classes = len(train_dataset.classes)

# Load ResNet50 pre-trained model
resnet = models.resnet50(pretrained=True)

# Freeze pre-trained layers
for param in resnet.parameters():
    param.requires_grad = False

# Modify the final fully connected layer for our custom dataset
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

# Move the model to the selected device (GPU or CPU)
resnet = resnet.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Loss for multi-class classification
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)  # Fine-tune the last layer

# Training variables
epochs = 50
train_losses, val_losses, train_accs, val_accs = [], [], [], []  # Metrics storage

# Training loop
for epoch in range(epochs):
    resnet.train()  # Set model to training mode
    train_loss, train_correct = 0, 0

    # Training phase
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device
        optimizer.zero_grad()  # Clear gradients from the last step
        outputs = resnet(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        train_loss += loss.item()  # Accumulate batch loss
        _, preds = torch.max(outputs, 1)  # Get predicted classes
        train_correct += torch.sum(preds == labels)  # Count correct predictions

    # Validation phase
    resnet.eval()  # Set model to evaluation mode
    val_loss, val_correct = 0, 0
    with torch.no_grad():  # No gradient computation
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()  # Accumulate validation loss
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels)

    # Calculate average losses and accuracy
    train_loss /= len(train_loader)
    val_loss /= len(valid_loader)
    train_acc = train_correct.double() / len(train_dataset)
    val_acc = val_correct.double() / len(valid_dataset)

    # Store metrics for plotting later
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc.item())
    val_accs.append(val_acc.item())

    # Print epoch summary
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# Save the trained model to a file
torch.save(resnet.state_dict(), 'model_resnet50.pth')

# Flask app initialization
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Save metrics for graph rendering
metrics = {'train_loss': train_losses, 'val_loss': val_losses, 'train_acc': train_accs, 'val_acc': val_accs}

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    # Render the main HTML page
    return render_template('deepCars.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform prediction on the uploaded image
        prediction_result = predict_image(file_path)
        
        return jsonify({
            'prediction': prediction_result  # Return the predicted class name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def predict_image(image_path):
    try:
        # Load and preprocess the image
        img = Image.open(image_path)
        img_tensor = data_transforms['valid'](img).unsqueeze(0).to(device)

        # Perform inference
        resnet.eval()
        with torch.no_grad():
            outputs = resnet(img_tensor)
            _, predicted_class = torch.max(outputs, 1)

        # Map prediction to class name
        predicted_label = train_dataset.classes[predicted_class.item()]
        return predicted_label
    except Exception as e:
        print(f"Error predicting image: {e}")
        return 'Error'

@app.route('/metrics', methods=['GET'])
def get_metrics():
    # Return metrics as JSON
    return jsonify(metrics)

# Start the Flask application
if __name__ == '__main__':
    print("Starting Flask application...")
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure upload folder exists
    app.run(debug=False, host='0.0.0.0', port=5000)
