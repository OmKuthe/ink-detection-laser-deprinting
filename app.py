"""
app.py
Main Streamlit application for Ink Detection + Classification + Laser Deprinting Simulation
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Ink Detection & Laser Deprinting System",
    page_icon="🖋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI - FIXED VISIBILITY
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        color: white !important;
        margin: 0;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9) !important;
        margin-top: 10px;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #2d2d44 0%, #1a1a2e 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        text-align: center;
        color: white;
    }
    
    /* Text colors for better visibility */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #e0e0e0 !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-163ttbj, .stSidebar {
        background-color: #0f0f1a !important;
    }
    
    .stSidebar .stMarkdown, .stSidebar p, .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: #e0e0e0 !important;
    }
    
    /* Success/Warning/Info boxes */
    .stAlert {
        background-color: rgba(30,30,50,0.9) !important;
        border-radius: 10px !important;
    }
    
    .stAlert p {
        color: white !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    
    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white !important;
    }
    
    /* Metrics */
    .stMetric label, .stMetric .stMetricLabel {
        color: #b0b0b0 !important;
    }
    
    .stMetric .stMetricValue {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: bold !important;
    }
    
    /* Slider */
    .stSlider label {
        color: #e0e0e0 !important;
    }
    
    /* Radio buttons */
    .stRadio label {
        color: #e0e0e0 !important;
    }
    
    /* Checkbox */
    .stCheckbox label {
        color: #e0e0e0 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #2d2d44 !important;
        color: #e0e0e0 !important;
        border-radius: 10px;
    }
    
    .streamlit-expanderContent {
        background-color: #1a1a2e !important;
        color: #e0e0e0 !important;
    }
    
    /* Info box */
    .stInfo {
        background-color: rgba(23, 162, 184, 0.2) !important;
        border-left: 4px solid #17a2b8 !important;
    }
    
    .stInfo p {
        color: #e0e0e0 !important;
    }
    
    /* Success box */
    .stSuccess {
        background-color: rgba(40, 167, 69, 0.2) !important;
        border-left: 4px solid #28a745 !important;
    }
    
    .stSuccess p {
        color: #e0e0e0 !important;
    }
    
    /* Warning box */
    .stWarning {
        background-color: rgba(255, 193, 7, 0.2) !important;
        border-left: 4px solid #ffc107 !important;
    }
    
    .stWarning p {
        color: #e0e0e0 !important;
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: #2d2d44 !important;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stFileUploader label {
        color: #e0e0e0 !important;
    }
    
    /* Dataframe/Table */
    .stTable {
        background-color: #2d2d44 !important;
        color: #e0e0e0 !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: #0f0f1a !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #e0e0e0 !important;
    }
    
    /* Column text */
    .column p, .stColumn p {
        color: #e0e0e0 !important;
    }
    
    /* Caption text */
    .stCaption, .stCaption p {
        color: #b0b0b0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HELPER FUNCTIONS
# ============================================

def load_model():
    """Load the trained classifier model"""
    model_path = "models/ink_classifier.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def detect_ink(image):
    """
    Detect ink regions in the image
    Returns: binary mask, ink percentage, processed image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding for better results
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Calculate ink percentage
    ink_pixels = np.sum(binary == 255)
    total_pixels = binary.size
    ink_percentage = (ink_pixels / total_pixels) * 100
    
    return binary, ink_percentage

def extract_features_for_classification(image, binary_mask):
    """
    Extract features for handwritten vs printed classification
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / edges.size
    
    # Intensity variance
    intensity_variance = np.var(gray) / 255.0
    
    # Stroke width estimation
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours and len(contours) > 0:
        areas = [cv2.contourArea(c) for c in contours]
        perimeters = [cv2.arcLength(c, True) for c in contours]
        avg_area = np.mean(areas)
        avg_perimeter = np.mean(perimeters)
        stroke_width = avg_area / (avg_perimeter + 1) / 100.0
    else:
        stroke_width = 0
    
    # Horizontal and vertical projection variance
    h_proj = np.sum(binary_mask, axis=1)
    v_proj = np.sum(binary_mask, axis=0)
    h_var = np.var(h_proj) / 1000.0
    v_var = np.var(v_proj) / 1000.0
    
    # Number of connected components
    num_labels, _ = cv2.connectedComponents(binary_mask)
    component_count = (num_labels - 1) / 100.0
    
    # Aspect ratio variations
    if contours:
        heights = [cv2.boundingRect(c)[3] for c in contours]
        widths = [cv2.boundingRect(c)[2] for c in contours]
        avg_height = np.mean(heights) / 100.0 if heights else 0
        avg_width = np.mean(widths) / 100.0 if widths else 0
        aspect_ratios = [h/w if w > 0 else 0 for h, w in zip(heights, widths)]
        ar_var = np.var(aspect_ratios) if aspect_ratios else 0
    else:
        avg_height = avg_width = ar_var = 0
    
    features = [[
        edge_density, intensity_variance, stroke_width,
        h_var, v_var, component_count, avg_height, avg_width, ar_var
    ]]
    
    return features

def classify_text_type(model, features):
    """Classify as printed (0) or handwritten (1)"""
    if model is None:
        return None, None
    
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    text_type = "✍️ Handwritten" if prediction == 1 else "🖨️ Printed"
    confidence = max(probabilities) * 100
    
    return text_type, confidence

def simulate_laser_deprinting(image, binary_mask, intensity):
    """
    Simulate laser deprinting by progressively removing ink
    """
    # Create a copy
    result = image.copy()
    
    # Calculate removal factor based on intensity (0-100)
    removal_factor = intensity / 100.0
    
    # Apply laser effect to ink regions
    ink_regions = binary_mask > 0
    
    # For each channel, reduce ink intensity
    for c in range(3):
        channel = result[:, :, c]
        # Reduce ink intensity (simulate laser removal)
        channel[ink_regions] = channel[ink_regions] * (1 - removal_factor * 0.8)
        # Add slight "burn" effect at high intensities
        if intensity > 70:
            burn_intensity = (intensity - 70) / 30
            channel[ink_regions] = np.clip(
                channel[ink_regions] + np.random.normal(0, 5, channel[ink_regions].shape),
                0, 255
            )
        result[:, :, c] = np.clip(channel, 0, 255)
    
    return result.astype(np.uint8)

def create_heatmap_overlay(image, binary_mask, intensity):
    """
    Create a heatmap showing laser application areas
    """
    # Create heatmap
    heatmap = np.zeros_like(image)
    
    # Apply heat effect based on laser intensity
    ink_regions = binary_mask > 0
    heat_intensity = intensity / 100.0
    
    # Red channel for heat
    heatmap[ink_regions, 2] = (255 * heat_intensity).astype(np.uint8)
    # Yellow overlay
    heatmap[ink_regions, 1] = (150 * heat_intensity).astype(np.uint8)
    
    # Blend with original
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    
    return overlay

def generate_report(ink_percentage, text_type, confidence, laser_intensity):
    """
    Generate analysis report
    """
    report = f"""
    # 📊 Analysis Report
    
    **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    ## Ink Analysis
    - Ink Coverage: **{ink_percentage:.2f}%**
    - Status: **{"✅ Ink Detected" if ink_percentage > 2 else "❌ No Significant Ink"}**
    
    ## Text Classification
    - Type: **{text_type}**
    - Confidence: **{confidence:.1f}%**
    
    ## Laser Simulation
    - Laser Intensity: **{laser_intensity}%**
    - Effect: **{"Aggressive Removal" if laser_intensity > 70 else "Moderate Removal" if laser_intensity > 30 else "Mild Removal"}**
    """
    return report

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🖋️ Ink Detection & Laser Deprinting System</h1>
        <p>Advanced Document Analysis | Handwritten vs Printed Classification | Laser Removal Simulation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎮 Controls")
        
        # Input method
        input_method = st.radio(
            "Select Input Method",
            ["📁 Upload Image", "📸 Capture from Camera"]
        )
        
        st.markdown("---")
        st.markdown("## ⚙️ Laser Settings")
        
        # Laser intensity slider
        laser_intensity = st.slider(
            "Laser Power (%)",
            min_value=0,
            max_value=100,
            value=50,
            help="Higher intensity removes more ink"
        )
        
        st.markdown("---")
        st.markdown("## 📈 Visualization")
        
        show_heatmap = st.checkbox("Show Laser Heatmap", value=False)
        show_comparison = st.checkbox("Show Before/After Comparison", value=True)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        **System Features:**
        - Ink detection using adaptive thresholding
        - ML-based text classification  
        - Laser deprinting simulation
        - Real-time analysis
        """)
    
    # Main content area - two columns
    col1, col2 = st.columns([1, 1])
    
    image = None
    
    # Handle image input
    if input_method == "📁 Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Upload a clear image of handwritten or printed text"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = np.array(image)
            # Convert RGBA to RGB if needed
            if len(image.shape) == 3 and image.shape[-1] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    else:  # Camera capture
        camera_image = st.camera_input("Take a picture")
        if camera_image is not None:
            image = Image.open(camera_image)
            image = np.array(image)
    
    if image is not None:
        # Convert to BGR for OpenCV
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[-1] == 3:  # RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Process image
        with st.spinner("🔍 Analyzing image..."):
            binary_mask, ink_percentage = detect_ink(image)
            
            # Load model and classify
            model = load_model()
            if model is not None and ink_percentage > 2:
                features = extract_features_for_classification(image, binary_mask)
                text_type, confidence = classify_text_type(model, features)
            else:
                text_type = "❓ Insufficient Ink"
                confidence = 0
            
            # Simulate laser deprinting
            laser_result = simulate_laser_deprinting(image, binary_mask, laser_intensity)
            
            if show_heatmap:
                heatmap_view = create_heatmap_overlay(image, binary_mask, laser_intensity)
        
        # Display in columns
        with col1:
            st.markdown("### 📷 Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # Metrics
            st.markdown("### 📊 Metrics")
            col1a, col1b, col1c = st.columns(3)
            with col1a:
                st.metric("Ink Coverage", f"{ink_percentage:.1f}%")
            with col1b:
                st.metric("Laser Intensity", f"{laser_intensity}%")
            with col1c:
                st.metric("Ink Status", "Present ✅" if ink_percentage > 2 else "None ❌")
        
        with col2:
            if show_comparison:
                st.markdown("### 🔄 Before vs After Laser")
                compare_col1, compare_col2 = st.columns(2)
                with compare_col1:
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Before", use_column_width=True)
                with compare_col2:
                    st.image(cv2.cvtColor(laser_result, cv2.COLOR_BGR2RGB), caption="After Laser", use_column_width=True)
            else:
                if show_heatmap:
                    st.markdown("### 🔥 Laser Heatmap")
                    st.image(cv2.cvtColor(heatmap_view, cv2.COLOR_BGR2RGB), use_column_width=True)
                else:
                    st.markdown("### 🎯 Detected Ink Regions")
                    st.image(binary_mask, use_column_width=True)
                
                st.markdown("### 💥 Laser Simulation Result")
                st.image(cv2.cvtColor(laser_result, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        # Classification result (full width)
        st.markdown("---")
        st.markdown("### 🧠 Classification Result")
        
        if text_type == "✍️ Handwritten":
            st.success(f"### {text_type} Detected!")
            st.info(f"**Confidence:** {confidence:.1f}% | **Features:** Irregular stroke width, varying intensity, non-uniform alignment")
        elif text_type == "🖨️ Printed":
            st.info(f"### {text_type} Detected!")
            st.success(f"**Confidence:** {confidence:.1f}% | **Features:** Uniform stroke width, consistent intensity, aligned text")
        else:
            st.warning(f"### {text_type}")
            st.info("Please upload an image with clearer text content for classification")
        
        # Laser simulation analysis
        st.markdown("---")
        st.markdown("### 🔬 Laser Deprinting Analysis")
        
        lazer_col1, lazer_col2, lazer_col3 = st.columns(3)
        
        with lazer_col1:
            removal_efficiency = min(95, laser_intensity * 0.85)
            st.metric("Estimated Removal", f"{removal_efficiency:.1f}%")
        
        with lazer_col2:
            paper_safety = max(85, 100 - laser_intensity * 0.3)
            st.metric("Paper Safety", f"{paper_safety:.1f}%", 
                     delta="Good" if paper_safety > 80 else "Moderate")
        
        with lazer_col3:
            energy_used = laser_intensity * 0.5
            st.metric("Laser Energy", f"{energy_used:.1f} J/cm²")
        
        # Technical explanation
        with st.expander("🔬 Technical Details & Explanation"):
            st.markdown("""
            ### How It Works:
            
            #### 1. Ink Detection
            - **Adaptive Thresholding**: Automatically adjusts to varying lighting conditions
            - **Morphological Operations**: Removes noise and connects broken text regions
            - **Pixel Analysis**: Counts ink pixels to calculate coverage percentage
            
            #### 2. Classification (Handwritten vs Printed)
            - **Edge Density**: Printed text has more consistent edges
            - **Stroke Width Variation**: Handwriting shows more variation
            - **Projection Analysis**: Printed text creates more uniform horizontal/vertical projections
            - **Connected Components**: Handwriting typically has more separate components
            
            #### 3. Laser Deprinting Simulation
            - **Progressive Removal**: Higher intensity removes more ink
            - **Burn Effect Simulation**: At high intensities (>70%), adds realistic "laser burn" artifacts
            - **Paper Preservation**: Ink removal without damaging paper background
            
            ### Real-World Application:
            This system simulates a laser-based deprinting process used in:
            - Document recycling
            - Secure document destruction
            - Paper reuse technology
            - Archival restoration
            
            ### Laser Parameters (Theoretical):
            - **Wavelength**: 532nm (green laser) - optimal for ink absorption
            - **Pulse Duration**: Nanosecond pulses for precise removal
            - **Energy Density**: 0.5-2.0 J/cm² depending on ink type
            """)
        
        # Download report
        report = generate_report(ink_percentage, text_type, confidence, laser_intensity)
        st.download_button(
            label="📥 Download Analysis Report",
            data=report,
            file_name=f"ink_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
    
    else:
        # No image loaded - show instructions
        st.info("👈 Please upload an image or use the camera to begin analysis")
        
        # Show example expectations
        st.markdown("""
        ### 📋 What This System Does:
        
        | Feature | Description |
        |---------|-------------|
        | 🔍 **Ink Detection** | Identifies and quantifies ink presence in the image |
        | 🧠 **Text Classification** | Distinguishes between handwritten and printed text using ML |
        | 💥 **Laser Simulation** | Simulates laser-based ink removal with adjustable intensity |
        | 📊 **Analysis Report** | Generates comprehensive report of findings |
        
        ### 🎯 Try With:
        - ✍️ Handwritten notes
        - 🖨️ Printed documents
        - 📄 Mixed content
        - ✨ Clean paper (negative test)
        
        ### 💡 Pro Tip:
        For best results, use well-lit, high-contrast images with clear text.
        """)

if __name__ == "__main__":
    main()