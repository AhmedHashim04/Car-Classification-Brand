// Select the icon and members list elements
const icon = document.querySelector('.icon');
const membersList = document.querySelector('.members');

// Add smooth transitions on load
window.addEventListener('load', () => {
    icon.classList.add('init-transitions');
});

// Toggle active classes for icon and members list
icon.addEventListener('click', () => {
    icon.classList.toggle('active');
    membersList.classList.toggle('active');
});

// Debugging to check the icon element
console.log('Icon element:', icon);

// Functionality for a confirmation message
function confirmAction(message) {
    if (confirm(message)) {
        console.log('Action confirmed');
        // Add additional actions for confirmation here
    } else {
        console.log('Action canceled');
        // Add additional actions for cancellation here
    }
}

// Example usage of the confirmAction function
document.addEventListener('DOMContentLoaded', () => {
    const someActionElement = document.querySelector('#someAction'); // Replace with the actual element ID
    if (someActionElement) {
        someActionElement.addEventListener('click', () => {
            confirmAction('Are you sure you want to perform this action?');
        });
    }
});

// Placeholder code for handling dynamic features (if needed)
// Example to show/hide elements based on user interactions
function toggleVisibility(element) {
    if (element.style.display === 'none' || !element.style.display) {
        element.style.display = 'block';
    } else {
        element.style.display = 'none';
    }
}

// Debugging statements for testing and ensuring functionality
console.log('JavaScript code initialized successfully.');

// Function to handle file preview
function handleFileSelect(fileInput, previewId) {
    const preview = document.getElementById(previewId);

    fileInput.addEventListener('change', function (e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();

            reader.onload = function (e) {
                preview.innerHTML = `<img src="${e.target.result}" alt="Car preview">`;
            }

            reader.readAsDataURL(file);

            // Check if all images are uploaded
            checkAllUploads();
        }
    });
}

// Function to check if all images are uploaded
function checkAllUploads() {
    const file1 = document.getElementById('imageuploader1').files[0];
    const file2 = document.getElementById('imageuploader2').files[0];
    const file3 = document.getElementById('imageuploader3').files[0];

    if (file1 && file2 && file3) {
        // Automatically submit the form when all images are uploaded
        document.getElementById('uploadForm').submit();
    }
}

// Initialize file handlers for all three upload buttons
document.addEventListener('DOMContentLoaded', function () {
    handleFileSelect(document.getElementById('imageuploader1'), 'preview1');
    handleFileSelect(document.getElementById('imageuploader2'), 'preview2');
    handleFileSelect(document.getElementById('imageuploader3'), 'preview3');
});

document.getElementById('uploadForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const formData = new FormData(this);

    try {
        // Show loading state
        document.getElementById('car-details').textContent = 'Processing...';

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        // Update the prediction results
        if (data.prediction) {
            document.getElementById('car-name').textContent = data.prediction;
            document.getElementById('car-details').textContent = data.prediction;  // Just show the car type
        }
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('car-details').textContent = 'Error occurred during prediction.';
    }
});

// Preview image immediately after selection
document.getElementById('imageuploader').addEventListener('change', function (e) {
    const carInfoImg = document.querySelector('.car-info img');
    if (this.files && this.files[0]) {
        carInfoImg.src = URL.createObjectURL(this.files[0]);
    }
});
