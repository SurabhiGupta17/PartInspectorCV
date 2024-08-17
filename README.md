# PartInspectorCV
PartInspectorCV is a Python-based image classification tool designed to differentiate between good and defective parts using a combination of OpenCV for image processing and a Random Forest classifier for machine learning. The tool allows users to collect data from images, augment the data, extract features, train a model, and classify new parts in real-time.

Features
ROI Selection: Users can select Regions of Interest (ROIs) in the images where the relevant parts are located.
Feature Extraction: Utilizes various image processing techniques such as Local Binary Patterns (LBP), Histogram of Oriented Gradients (HOG), edge detection, and texture analysis to extract features from the selected ROIs.
Data Augmentation: Automatically augments images to increase the dataset diversity, improving the classifier's performance.
Random Forest Classifier: Trains a Random Forest classifier with hyperparameter tuning using RandomizedSearchCV.
Real-time Classification: Classifies new parts in real-time using the trained model, with visual feedback on the classification result and confidence level.

# Installation
1. Clone the repository
2. Install the required Python packages
3. Ensure your webcam is properly connected.

# Usage
1. Run the script
2. Collect data
3. Follow the prompts to collect images of both good and defective parts. The tool will ask you to capture images and select ROIs for feature extraction.
4. The collected data will be automatically augmented to increase sample diversity.
5. Train the classifier
6. Classify new parts
7. After training, you can classify new parts by following the on-screen instructions. The tool will display the classification result and confidence level.
8. Exit the program
9. The tool will prompt you if you want to classify another part or exit the program.

