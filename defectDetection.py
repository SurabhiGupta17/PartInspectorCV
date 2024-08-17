import cv2
import numpy as np
import pickle
from skimage.feature import local_binary_pattern, hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from scipy.stats import randint

def select_rois(image, num_rois=6):
    rois = []
    image_with_rois = image.copy()
    
    for i in range(num_rois):
        roi = cv2.selectROI("Select ROI", image_with_rois, fromCenter=False, showCrosshair=True)
        if roi[2] and roi[3]:
            rois.append(roi)
            cv2.rectangle(image_with_rois, (int(roi[0]), int(roi[1])),
                           (int(roi[0] + roi[2]), int(roi[1] + roi[3])),
                           (0, 255, 0), 2)
    
    cv2.destroyWindow("Select ROI")
    return rois, image_with_rois

def extract_features(image, rois):
    features = []
    for roi in rois:
        x, y, w, h = map(int, roi)
        roi_img = image[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_roi = clahe.apply(gray_roi)
        
        # LBP features
        lbp = local_binary_pattern(clahe_roi, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=59, range=(0, 59), density=True)
        
        # HOG features
        hog_features = hog(clahe_roi, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        
        # Edge features
        edges = cv2.Canny(clahe_roi, 100, 200)
        edge_hist, _ = np.histogram(edges, bins=8, range=(0, 256), density=True)
        
        # Add texture features
        texture_features = cv2.Laplacian(clahe_roi, cv2.CV_64F).var()
        
        # Combine features
        roi_features = np.concatenate([lbp_hist, hog_features, edge_hist, [texture_features]])
        features.append(roi_features)
    
    return np.concatenate(features)

def augment_data(image, rois, num_augmented_samples=10):
    augmented_images = [image]
    
    for _ in range(num_augmented_samples - 1):
        # Brightness adjustment
        beta = np.random.randint(-30, 30)
        bright_dark = cv2.convertScaleAbs(image, beta=beta)
        augmented_images.append(bright_dark)
        
        # Rotation
        angle = np.random.uniform(-10, 10)
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        augmented_images.append(rotated)
    
    return augmented_images

def capture_image():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        cv2.imshow('Live Webcam Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cap.release()
            cv2.destroyAllWindows()
            return frame
    
    cap.release()
    cv2.destroyAllWindows()
    return None

def collect_data(label, num_samples=10, rois=None):
    data = []
    
    print(f"Capturing image for {label} part...")
    image = capture_image()
    if image is None:
        return None, None
    
    cv2.imshow(f'{label.capitalize()} Part', image)
    cv2.waitKey(0)
    
    if rois is None:
        rois, image_with_rois = select_rois(image)
    else:
        image_with_rois = image.copy()
        for roi in rois:
            x, y, w, h = map(int, roi)
            cv2.rectangle(image_with_rois, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    augmented_images = augment_data(image, rois, num_samples)
    for aug_image in augmented_images:
        features = extract_features(aug_image, rois)
        data.append(features)
    
    cv2.imshow(f'{label.capitalize()} Part with ROIs', image_with_rois)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return np.array(data), rois

# Collect data
print("Collecting data for good parts:")
good_data, rois = collect_data("good")
print("Collecting data for bad parts:")
bad_data, _ = collect_data("bad", rois=rois)  # Use the same ROIs for bad parts

# Prepare data for training
X = np.vstack((good_data, bad_data))
y = np.hstack((np.ones(len(good_data)), np.zeros(len(bad_data))))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 11),
    'class_weight': ['balanced', None]
}

# Create a base model
rf = RandomForestClassifier(random_state=42)

# Instantiate RandomizedSearchCV object
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=20, cv=5, random_state=42, n_jobs=-1)

# Perform random search
random_search.fit(X_train_scaled, y_train)

# Get the best model
best_rf = random_search.best_estimator_

# Evaluate the model
y_pred = best_rf.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Save the model and scaler
with open('part_classifier_model.pkl', 'wb') as f:
    pickle.dump((best_rf, scaler, rois), f)

print("Model trained and saved.")

# Classification phase
while True:
    input("Press Enter to classify a new part...")
    
    print("Capturing image for classification...")
    new_image = capture_image()
    if new_image is None:
        continue
    
    cv2.imshow('New Part', new_image)
    cv2.waitKey(0)
    
    # Load the model
    with open('part_classifier_model.pkl', 'rb') as f:
        rf_model, scaler, rois = pickle.load(f)
    
    # Extract features from the new image
    new_features = extract_features(new_image, rois)
    new_features_scaled = scaler.transform([new_features])
    
    # Predict
    prediction = rf_model.predict(new_features_scaled)[0]
    probability = rf_model.predict_proba(new_features_scaled)[0]
    
    result = "Good" if prediction == 1 else "Bad"
    confidence = probability[1] if prediction == 1 else probability[0]
    
    # Display result
    result_image = new_image.copy()
    for roi in rois:
        x, y, w, h = map(int, roi)
        color = (0, 255, 0) if result == "Good" else (0, 0, 255)
        cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
    
    cv2.putText(result_image, f"Classification: {result}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(result_image, f"Confidence: {confidence:.2f}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Classification Result', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    choice = input("Do you want to classify another part? (y/n): ")
    if choice.lower() != 'y':
        break

print("Program ended.")
