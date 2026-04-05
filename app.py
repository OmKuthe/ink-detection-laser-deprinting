"""
app.py - CORRECTED VERSION WITH PROPER METRICS
Ink Detection + Classification + Laser Path Planning + Actual Ink Removal
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Ink Detection & Laser Deprinting System",
    page_icon="🖋️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #e0e0e0 !important;
    }
    .success-message {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# INK DETECTION FUNCTIONS
# ============================================

def detect_ink_regions(image, is_original=True):
    """
    Detect ink regions in the image
    For original: dark pixels = ink
    For processed: we need to detect remaining dark ink
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if is_original:
        # For original image: dark pixels are ink
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # For processed image: detect dark pixels (remaining ink)
        # Use a threshold that ignores white (laser-removed) areas
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Clean up
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Calculate ink percentage (black pixels)
    ink_pixels = np.sum(binary == 255)
    total_pixels = binary.size
    ink_percentage = (ink_pixels / total_pixels) * 100
    
    return binary, ink_percentage

def extract_features(image, binary_mask):
    """Extract features for classification"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / edges.size if edges.size > 0 else 0
    
    intensity_variance = np.var(gray) / (255 * 255)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours and len(contours) > 5:
        widths = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > 5:
                widths.append(w / h if h > 0 else 0)
        stroke_width_var = np.var(widths) if widths else 0
    else:
        stroke_width_var = 0
    
    h_projection = np.sum(binary_mask, axis=1)
    h_var = np.var(h_projection) / 1000 if len(h_projection) > 0 else 0
    
    v_projection = np.sum(binary_mask, axis=0)
    v_var = np.var(v_projection) / 1000 if len(v_projection) > 0 else 0
    
    num_labels, _ = cv2.connectedComponents(binary_mask)
    component_count = (num_labels - 1) / 100
    
    if contours:
        heights = [cv2.boundingRect(c)[3] for c in contours if cv2.boundingRect(c)[3] > 5]
        widths = [cv2.boundingRect(c)[2] for c in contours if cv2.boundingRect(c)[2] > 5]
        avg_height = np.mean(heights) / 100 if heights else 0
        avg_width = np.mean(widths) / 100 if widths else 0
        aspect_ratios = [w/h if h > 0 else 0 for w, h in zip(widths, heights)]
        ar_var = np.var(aspect_ratios) if aspect_ratios else 0
    else:
        avg_height = avg_width = ar_var = 0
    
    lbp_variance = np.var(cv2.Laplacian(gray, cv2.CV_64F)) / 1000
    
    features = [[
        edge_density, intensity_variance, stroke_width_var,
        h_var, v_var, component_count, avg_height, avg_width, ar_var, lbp_variance
    ]]
    
    return features, contours

def classify_text(features):
    """Classify handwritten vs printed"""
    edge_density = features[0][0]
    intensity_var = features[0][1]
    stroke_var = features[0][2]
    h_var = features[0][3]
    ar_var = features[0][8]
    
    handwritten_score = 0
    if stroke_var > 0.12:
        handwritten_score += 35
    if h_var > 0.4:
        handwritten_score += 25
    if ar_var > 0.07:
        handwritten_score += 20
    if intensity_var > 0.1:
        handwritten_score += 20
    
    printed_score = 0
    if stroke_var < 0.08:
        printed_score += 35
    if h_var < 0.25:
        printed_score += 25
    if ar_var < 0.04:
        printed_score += 20
    if edge_density > 0.3:
        printed_score += 20
    
    confidence = max(handwritten_score, printed_score)
    
    if handwritten_score > printed_score and confidence > 35:
        return "✍️ Handwritten", confidence
    elif printed_score > handwritten_score and confidence > 35:
        return "🖨️ Printed", confidence
    else:
        return "🤔 Uncertain", confidence

# ============================================
# LASER PATH PLANNING & REMOVAL
# ============================================

def plan_laser_path(binary_mask, strategy="raster"):
    """Plan laser scanning path over ink regions"""
    height, width = binary_mask.shape
    laser_points = []
    
    if strategy == "raster":
        for y in range(0, height, 2):
            if y % 4 < 2:
                for x in range(0, width, 2):
                    if binary_mask[y, x] == 255:
                        laser_points.append((x, y))
            else:
                for x in range(width - 1, -1, -2):
                    if binary_mask[y, x] == 255:
                        laser_points.append((x, y))
    
    elif strategy == "contour":
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            for point in contour:
                laser_points.append((int(point[0][0]), int(point[0][1])))
    
    elif strategy == "spiral":
        center_x, center_y = width // 2, height // 2
        for radius in range(0, max(width, height), 5):
            for angle in np.arange(0, 2 * np.pi, 0.2):
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                if 0 <= x < width and 0 <= y < height:
                    if binary_mask[y, x] == 255:
                        laser_points.append((x, y))
    
    return list(set(laser_points))  # Remove duplicates

def apply_laser_removal(image, binary_mask, laser_points, intensity):
    """
    Remove ink by turning dark pixels to match paper background
    """
    result = image.copy()
    
    # Estimate paper background color (mode of light pixels)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    light_pixels = gray[gray > 150]
    if len(light_pixels) > 0:
        paper_color = int(np.mean(light_pixels))
    else:
        paper_color = 255
    
    removal_count = 0
    points_to_process = int(len(laser_points) * (intensity / 100))
    
    # Process laser points
    for i in range(min(points_to_process, len(laser_points))):
        x, y = laser_points[i]
        if 0 <= x < result.shape[1] and 0 <= y < result.shape[0]:
            # Check if this is actually an ink pixel
            if binary_mask[y, x] == 255:
                # Turn to paper color
                result[y, x] = [paper_color, paper_color, paper_color]
                removal_count += 1
                
                # Spread effect for high intensity
                if intensity > 70:
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < result.shape[1] and 0 <= ny < result.shape[0]:
                                if binary_mask[ny, nx] == 255:
                                    result[ny, nx] = [paper_color, paper_color, paper_color]
                                    removal_count += 1
    
    return result, removal_count, points_to_process

def draw_laser_path_on_image(image, laser_points, processed_count=0):
    """Draw laser path visualization"""
    result = image.copy()
    
    for i, (x, y) in enumerate(laser_points):
        if i < processed_count:
            cv2.circle(result, (x, y), 2, (0, 255, 0), -1)  # Processed - green
        else:
            cv2.circle(result, (x, y), 1, (0, 0, 255), -1)  # Pending - red
    
    if len(laser_points) > 1:
        points_array = np.array(laser_points, dtype=np.int32)
        cv2.polylines(result, [points_array], False, (0, 255, 255), 1)
    
    return result

def calculate_ink_removal_metrics(original_ink_pixels, final_ink_pixels, original_percentage, final_percentage):
    """Calculate accurate removal metrics"""
    
    # Absolute removal
    pixels_removed = max(0, original_ink_pixels - final_ink_pixels)
    
    # Percentage removal (based on original ink)
    if original_ink_pixels > 0:
        removal_percentage = (pixels_removed / original_ink_pixels) * 100
    else:
        removal_percentage = 0
    
    # Change in percentage points
    percentage_points_reduced = max(0, original_percentage - final_percentage)
    
    return {
        'pixels_removed': pixels_removed,
        'removal_percentage': removal_percentage,
        'percentage_points_reduced': percentage_points_reduced,
        'remaining_ink_percentage': final_percentage
    }

# ============================================
# ANIMATION FUNCTION
# ============================================

def animate_laser_removal(image, laser_points, binary_mask, intensity, progress_bar, status_text):
    """Animate laser removal process"""
    result = image.copy()
    
    # Estimate paper color
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    light_pixels = gray[gray > 150]
    paper_color = int(np.mean(light_pixels)) if len(light_pixels) > 0 else 255
    
    total_points = len(laser_points)
    points_to_process = int(total_points * (intensity / 100))
    
    for i in range(0, points_to_process, max(1, points_to_process // 50)):
        batch_end = min(i + (total_points // 50), points_to_process)
        
        for j in range(i, batch_end):
            if j < len(laser_points):
                x, y = laser_points[j]
                if 0 <= x < result.shape[1] and 0 <= y < result.shape[0]:
                    if binary_mask[y, x] == 255:
                        result[y, x] = [paper_color, paper_color, paper_color]
                        
                        if intensity > 70:
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    nx, ny = x + dx, y + dy
                                    if 0 <= nx < result.shape[1] and 0 <= ny < result.shape[0]:
                                        if binary_mask[ny, nx] == 255:
                                            result[ny, nx] = [paper_color, paper_color, paper_color]
        
        progress = (batch_end / total_points) * 100
        progress_bar.progress(min(100, progress / 100))
        status_text.text(f"🔥 Laser processing... {progress:.1f}% complete")
        
        yield result.copy()
        time.sleep(0.03)
    
    status_text.text("✅ Laser deprinting complete!")
    yield result

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🖋️ Laser Deprinting System</h1>
        <p>Ink Detection | Classification | Laser Path Planning | Actual Ink Removal</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎮 Controls")
        input_method = st.radio("Input Method", ["📁 Upload Image", "📸 Capture from Camera"])
        
        st.markdown("---")
        st.markdown("## ⚙️ Laser Settings")
        laser_intensity = st.slider("Laser Power (%)", 0, 100, 80)
        laser_strategy = st.selectbox("Scanning Pattern", ["raster", "contour", "spiral"])
        
        st.markdown("---")
        st.markdown("## 🎬 Display")
        animate_mode = st.checkbox("Animate Laser Process", value=True)
        show_path = st.checkbox("Show Laser Path", value=True)
    
    # Image input
    image = None
    
    if input_method == "📁 Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            image = np.array(image)
            if len(image.shape) == 3 and image.shape[-1] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        camera_image = st.camera_input("Take a picture")
        if camera_image:
            image = Image.open(camera_image)
            image = np.array(image)
    
    if image is not None:
        # Convert to BGR
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # STEP 1: Detect Ink on ORIGINAL image
        with st.spinner("🔍 Detecting ink regions..."):
            original_binary, original_ink_percentage = detect_ink_regions(image, is_original=True)
            features, contours = extract_features(image, original_binary)
            text_type, confidence = classify_text(features)
        
        # Display original analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📷 Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            st.markdown("### 📊 Original Analysis")
            st.metric("Initial Ink Coverage", f"{original_ink_percentage:.2f}%")
            st.metric("Text Type", text_type)
            st.metric("Confidence", f"{confidence:.0f}%")
        
        with col2:
            st.markdown("### 🎯 Detected Ink Regions")
            st.image(original_binary, use_column_width=True)
            ink_pixels = np.sum(original_binary == 255)
            st.metric("Ink Pixels Detected", f"{ink_pixels:,}")
            st.metric("Contours Found", len(contours))
        
        # STEP 2: Plan Laser Path
        st.markdown("---")
        st.markdown("## 🔴 Laser Path Planning")
        
        laser_points = plan_laser_path(original_binary, laser_strategy)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Laser Points Planned", f"{len(laser_points):,}")
        with col2:
            coverage = (len(laser_points) / max(1, ink_pixels)) * 100
            st.metric("Path Coverage", f"{coverage:.1f}%")
        with col3:
            est_time = len(laser_points) / 500
            st.metric("Est. Time", f"{est_time:.1f}s")
        
        if show_path:
            path_image = draw_laser_path_on_image(image, laser_points, 0)
            st.image(cv2.cvtColor(path_image, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.caption("🔴 Red: Target | 🟡 Line: Laser Path")
        
        # STEP 3: Execute Laser Removal
        st.markdown("---")
        st.markdown("## 💥 Execute Laser Deprinting")
        
        if st.button("🚀 START LASER REMOVAL", use_container_width=True):
            if animate_mode:
                # Animated removal
                progress_bar = st.progress(0)
                status_text = st.empty()
                animation_placeholder = st.empty()
                
                for frame in animate_laser_removal(image, laser_points, original_binary, laser_intensity, progress_bar, status_text):
                    animation_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
                
                final_result, removed_count, processed = apply_laser_removal(image, original_binary, laser_points, laser_intensity)
            else:
                # Instant removal
                with st.spinner("🔥 Laser processing..."):
                    final_result, removed_count, processed = apply_laser_removal(image, original_binary, laser_points, laser_intensity)
                st.image(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # STEP 4: Calculate FINAL ink metrics on processed image
            final_binary, final_ink_percentage = detect_ink_regions(final_result, is_original=False)
            final_ink_pixels = np.sum(final_binary == 255)
            original_ink_pixels = np.sum(original_binary == 255)
            
            # Calculate accurate metrics
            metrics = calculate_ink_removal_metrics(
                original_ink_pixels, 
                final_ink_pixels,
                original_ink_percentage,
                final_ink_percentage
            )
            
            # Display success and metrics
            st.markdown(f"""
            <div class="success-message">
                ✅ Laser Deprinting Complete!<br>
                Laser Power: {laser_intensity}% | Strategy: {laser_strategy}
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics dashboard
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pixels Removed", f"{metrics['pixels_removed']:,}")
            with col2:
                st.metric("Removal Rate", f"{metrics['removal_percentage']:.1f}%")
            with col3:
                st.metric("Ink Before", f"{original_ink_percentage:.2f}%")
            with col4:
                st.metric("Ink After", f"{final_ink_percentage:.2f}%")
            
            # Progress visualization
            st.markdown("### 📈 Removal Progress")
            progress_bar2 = st.progress(metrics['removal_percentage'] / 100)
            st.caption(f"🎯 {metrics['removal_percentage']:.1f}% of ink removed")
            
            # Before/After comparison
            st.markdown("### 📸 Before vs After")
            compare_col1, compare_col2 = st.columns(2)
            with compare_col1:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="BEFORE", use_column_width=True)
            with compare_col2:
                st.image(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB), caption="AFTER LASER", use_column_width=True)
            
            # Download button
            result_rgb = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)
            import io
            buf = io.BytesIO()
            result_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="📥 Download Laser Processed Image",
                data=byte_im,
                file_name=f"laser_deprinted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )
        
        # Technical explanation
        with st.expander("🔬 How Laser Removal Works"):
            st.markdown("""
            ### The Deprinting Process:
            
            1. **Ink Detection**: Identifies all dark ink pixels
            2. **Path Planning**: Calculates optimal laser scanning pattern
            3. **Laser Application**: Virtual laser turns ink pixels to paper color
            4. **Verification**: Measures remaining ink after processing
            
            ### Why Metrics Are Now Accurate:
            - Original ink: Detects dark pixels (ink)
            - After laser: Detects remaining dark pixels
            - Removal = Original - Remaining
            
            ### Real Laser Parameters:
            | Parameter | Value |
            |-----------|-------|
            | Wavelength | 532nm (green) |
            | Pulse Duration | Nanoseconds |
            | Spot Size | 20-50μm |
            | Fluence | 0.5-2.5 J/cm² |
            """)
    
    else:
        st.info("👈 Upload an image or use camera to begin")
        
        st.markdown("""
        ### 🎯 Laser Deprinting System
        
        **Complete workflow:**
        1. **Ink Detection** - Find all ink pixels
        2. **Classification** - Handwritten vs Printed
        3. **Path Planning** - Optimize laser scanning
        4. **Laser Removal** - Remove ink (turn to paper color)
        5. **Verification** - Calculate removal metrics
        
        **Try with:**
        - Handwritten notes
        - Printed documents
        - Text on white paper
        """)

if __name__ == "__main__":
    import numpy as np
    main()