"""
train_model.py
Train a classifier to distinguish between handwritten and printed text
"""

import cv2
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import glob

# ============================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================

def extract_features(image_path):
    """
    Extract features from an image for classification
    Returns feature vector
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / edges.size
    
    # 2. Pixel intensity variance (handwriting varies more)
    intensity_variance = np.var(gray)
    
    # 3. Stroke width (using morphological operations)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        avg_contour_area = np.mean([cv2.contourArea(c) for c in contours])
        avg_perimeter = np.mean([cv2.arcLength(c, True) for c in contours])
        stroke_width = avg_contour_area / (avg_perimeter + 1)
    else:
        stroke_width = 0
    
    # 4. Horizontal projection variance (text line alignment)
    horizontal_projection = np.sum(thresh, axis=1)
    h_proj_variance = np.var(horizontal_projection)
    
    # 5. Vertical projection variance
    vertical_projection = np.sum(thresh, axis=0)
    v_proj_variance = np.var(vertical_projection)
    
    # 6. Number of connected components
    num_labels, _ = cv2.connectedComponents(thresh)
    component_count = num_labels - 1  # subtract background
    
    # 7. Aspect ratio of bounding boxes
    heights = []
    widths = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        heights.append(h)
        widths.append(w)
    
    avg_height = np.mean(heights) if heights else 0
    avg_width = np.mean(widths) if widths else 0
    aspect_ratio_var = np.var([h/w if w > 0 else 0 for h, w in zip(heights, widths)]) if heights else 0
    
    # Combine all features
    features = [
        edge_density,
        intensity_variance / 255.0,  # Normalize
        stroke_width / 100.0,  # Normalize
        h_proj_variance / 1000.0,
        v_proj_variance / 1000.0,
        component_count / 100.0,
        avg_height / 100.0,
        avg_width / 100.0,
        aspect_ratio_var
    ]
    
    return features

def load_dataset(handwritten_dir, printed_dir):
    """
    Load images from folders and extract features
    """
    features = []
    labels = []
    
    # Load handwritten images (label = 1)
    print("Loading handwritten images...")
    for img_path in glob.glob(os.path.join(handwritten_dir, "*.*")):
        feat = extract_features(img_path)
        if feat:
            features.append(feat)
            labels.append(1)  # 1 = Handwritten
            print(f"  ✓ {os.path.basename(img_path)}")
    
    # Load printed images (label = 0)
    print("Loading printed images...")
    for img_path in glob.glob(os.path.join(printed_dir, "*.*")):
        feat = extract_features(img_path)
        if feat:
            features.append(feat)
            labels.append(0)  # 0 = Printed
            print(f"  ✓ {os.path.basename(img_path)}")
    
    return np.array(features), np.array(labels)

# ============================================
# CREATE SYNTHETIC DATASET (For demo purposes)
# ============================================

def create_synthetic_dataset():
    """
    Create synthetic feature data for demonstration
    In production, use real images
    """
    np.random.seed(42)
    
    # Printed text features (more uniform)
    printed_features = []
    for _ in range(100):
        printed_features.append([
            np.random.uniform(0.3, 0.5),  # edge_density
            np.random.uniform(0.2, 0.4),  # intensity_variance
            np.random.uniform(0.05, 0.15), # stroke_width
            np.random.uniform(0.1, 0.3),  # h_proj_variance
            np.random.uniform(0.1, 0.3),  # v_proj_variance
            np.random.uniform(0.2, 0.5),  # component_count
            np.random.uniform(0.3, 0.6),  # avg_height
            np.random.uniform(0.4, 0.7),  # avg_width
            np.random.uniform(0.05, 0.2)  # aspect_ratio_var
        ])
    
    # Handwritten features (more variation)
    handwritten_features = []
    for _ in range(100):
        handwritten_features.append([
            np.random.uniform(0.4, 0.7),  # edge_density
            np.random.uniform(0.5, 0.8),  # intensity_variance
            np.random.uniform(0.15, 0.35), # stroke_width
            np.random.uniform(0.4, 0.7),  # h_proj_variance
            np.random.uniform(0.4, 0.7),  # v_proj_variance
            np.random.uniform(0.6, 1.0),  # component_count
            np.random.uniform(0.5, 0.9),  # avg_height
            np.random.uniform(0.3, 0.8),  # avg_width
            np.random.uniform(0.2, 0.5)   # aspect_ratio_var
        ])
    
    X = np.array(printed_features + handwritten_features)
    y = np.array([0]*100 + [1]*100)  # 0=Printed, 1=Handwritten
    
    return X, y

# ============================================
# TRAIN AND SAVE MODEL
# ============================================

def train_and_save_model():
    """
    Main training function
    """
    print("="*50)
    print("INK DETECTION MODEL TRAINING")
    print("="*50)
    
    # Try to load real dataset, fallback to synthetic
    try:
        # Change these paths to your actual image folders
        X, y = load_dataset("sample_images/handwritten", "sample_images/printed")
        print(f"\n✅ Loaded {len(X)} real images")
    except:
        print("\n⚠️ Real dataset not found. Using synthetic data for demonstration.")
        print("📝 To use real images, create folders:")
        print("   - sample_images/handwritten/")
        print("   - sample_images/printed/")
        print("   and add your images there.\n")
        X, y = create_synthetic_dataset()
        print(f"✅ Generated {len(X)} synthetic samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Random Forest Classifier
    print("\n🔄 Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Printed', 'Handwritten']))
    
    # Feature importance
    feature_names = [
        'Edge Density', 'Intensity Variance', 'Stroke Width',
        'Horizontal Projection Var', 'Vertical Projection Var',
        'Component Count', 'Avg Height', 'Avg Width', 'Aspect Ratio Var'
    ]
    
    print(f"\n📈 Feature Importance:")
    for name, importance in sorted(zip(feature_names, model.feature_importances_), 
                                   key=lambda x: x[1], reverse=True):
        print(f"   {name}: {importance*100:.1f}%")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/ink_classifier.pkl")
    print(f"\n💾 Model saved to: models/ink_classifier.pkl")
    
    return model

if __name__ == "__main__":
    train_and_save_model()
    print("\n✨ Training complete! Run 'streamlit run app.py' to start the web app.")