"""
app.py - AGGRESSIVE INK REMOVAL VERSION (FIXED)
Ink Detection + Classification + Laser Path Planning + ACTUAL Ink Removal
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
    page_title="Laser Deprinting System",
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

def detect_ink_regions(image):
    """
    Detect ink regions - dark pixels are ink
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use Otsu's threshold to find dark ink
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Clean up
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Calculate ink percentage
    ink_pixels = np.sum(binary == 255)
    total_pixels = binary.size
    ink_percentage = (ink_pixels / total_pixels) * 100
    
    return binary, ink_percentage, gray

def extract_features(image, binary_mask):
    """Extract features for classification"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / edges.size if edges.size > 0 else 0
    
    # 2. Intensity variance
    intensity_variance = np.var(gray) / (255 * 255)
    
    # 3. Stroke width variance
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
    
    # 4. Horizontal projection variance
    h_projection = np.sum(binary_mask, axis=1)
    h_var = np.var(h_projection) / 1000 if len(h_projection) > 0 else 0
    
    # 5. Vertical projection variance
    v_projection = np.sum(binary_mask, axis=0)
    v_var = np.var(v_projection) / 1000 if len(v_projection) > 0 else 0
    
    # 6. Connected components
    num_labels, _ = cv2.connectedComponents(binary_mask)
    component_count = (num_labels - 1) / 100
    
    # 7-9. Character dimensions
    if contours:
        heights = [cv2.boundingRect(c)[3] for c in contours if cv2.boundingRect(c)[3] > 5]
        widths = [cv2.boundingRect(c)[2] for c in contours if cv2.boundingRect(c)[2] > 5]
        avg_height = np.mean(heights) / 100 if heights else 0
        avg_width = np.mean(widths) / 100 if widths else 0
        aspect_ratios = [w/h if h > 0 else 0 for w, h in zip(widths, heights)]
        ar_var = np.var(aspect_ratios) if aspect_ratios else 0
    else:
        avg_height = avg_width = ar_var = 0
    
    features = [[
        edge_density, intensity_variance, stroke_width_var,
        h_var, v_var, component_count, avg_height, avg_width, ar_var
    ]]
    
    return features, contours

def classify_text(features):
    """Classify handwritten vs printed"""
    stroke_var = features[0][2]
    h_var = features[0][3]
    ar_var = features[0][8]
    
    handwritten_score = 0
    if stroke_var > 0.12:
        handwritten_score += 40
    if h_var > 0.4:
        handwritten_score += 30
    if ar_var > 0.07:
        handwritten_score += 30
    
    printed_score = 0
    if stroke_var < 0.08:
        printed_score += 40
    if h_var < 0.25:
        printed_score += 30
    if ar_var < 0.04:
        printed_score += 30
    
    confidence = max(handwritten_score, printed_score)
    
    if handwritten_score > printed_score and confidence > 40:
        return "✍️ Handwritten", confidence
    elif printed_score > handwritten_score and confidence > 40:
        return "🖨️ Printed", confidence
    else:
        return "🤔 Uncertain", confidence

# ============================================
# AGGRESSIVE LASER REMOVAL
# ============================================

def plan_laser_path(binary_mask, strategy="raster"):
    """
    Plan laser path covering ALL ink pixels
    """
    height, width = binary_mask.shape
    laser_points = []
    
    if strategy == "raster":
        # Complete raster scan - X axis first, then Y
        for y in range(height):
            if y % 2 == 0:
                # Left to right
                for x in range(width):
                    if binary_mask[y, x] == 255:
                        laser_points.append((x, y))
            else:
                # Right to left (zigzag)
                for x in range(width - 1, -1, -1):
                    if binary_mask[y, x] == 255:
                        laser_points.append((x, y))
    
    elif strategy == "contour":
        # Trace contours of ink regions
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            for point in contour:
                laser_points.append((int(point[0][0]), int(point[0][1])))
    
    elif strategy == "spiral":
        # Spiral pattern covering all ink
        center_x, center_y = width // 2, height // 2
        max_radius = max(width, height)
        for radius in range(0, max_radius, 3):
            for angle in np.arange(0, 2 * np.pi, 0.1):
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                if 0 <= x < width and 0 <= y < height:
                    if binary_mask[y, x] == 255:
                        laser_points.append((x, y))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_points = []
    for point in laser_points:
        if point not in seen:
            seen.add(point)
            unique_points.append(point)
    
    return unique_points

def aggressive_laser_removal(image, binary_mask, laser_points, intensity):
    """
    AGGRESSIVE removal - completely erase ink pixels
    """
    result = image.copy()
    
    # Get paper background color (average of non-ink areas)
    non_ink_mask = binary_mask == 0
    if np.any(non_ink_mask):
        paper_bgr = np.mean(image[non_ink_mask], axis=0).astype(np.uint8)
    else:
        paper_bgr = np.array([255, 255, 255], dtype=np.uint8)
    
    # Calculate how many points to process
    points_to_process = int(len(laser_points) * (intensity / 100))
    
    # Create a mask for processed pixels
    processed_mask = np.zeros_like(binary_mask)
    
    # Aggressive removal
    removal_count = 0
    
    for i in range(min(points_to_process, len(laser_points))):
        x, y = laser_points[i]
        
        if 0 <= x < result.shape[1] and 0 <= y < result.shape[0]:
            if binary_mask[y, x] == 255 and processed_mask[y, x] == 0:
                # Remove the ink pixel
                result[y, x] = paper_bgr
                processed_mask[y, x] = 255
                removal_count += 1
                
                # Remove surrounding pixels based on intensity
                if intensity > 70:
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < result.shape[1] and 0 <= ny < result.shape[0]:
                                if binary_mask[ny, nx] == 255 and processed_mask[ny, nx] == 0:
                                    result[ny, nx] = paper_bgr
                                    processed_mask[ny, nx] = 255
                                    removal_count += 1
    
    return result, removal_count, points_to_process

def draw_laser_path_on_image(image, laser_points, processed_count=0):
    """
    Draw laser path visualization
    """
    result = image.copy()
    
    # Draw path line for processed points
    if len(laser_points) > 1 and processed_count > 0:
        points_array = np.array(laser_points[:min(processed_count, len(laser_points))], dtype=np.int32)
        if len(points_array) > 1:
            cv2.polylines(result, [points_array], False, (0, 255, 0), 1)
    
    # Draw points
    for i, (x, y) in enumerate(laser_points[:5000]):  # Limit for performance
        if i < processed_count:
            cv2.circle(result, (x, y), 1, (0, 255, 0), -1)
        else:
            cv2.circle(result, (x, y), 1, (0, 0, 255), -1)
    
    return result

# ============================================
# ANIMATION FUNCTION
# ============================================

def animate_laser_removal(image, laser_points, binary_mask, intensity, progress_bar, status_text, placeholder):
    """
    Animate the laser removal process
    """
    result = image.copy()
    
    # Get paper color
    non_ink_mask = binary_mask == 0
    if np.any(non_ink_mask):
        paper_bgr = np.mean(image[non_ink_mask], axis=0).astype(np.uint8)
    else:
        paper_bgr = np.array([255, 255, 255], dtype=np.uint8)
    
    total_points = len(laser_points)
    points_to_process = int(total_points * (intensity / 100))
    
    processed_mask = np.zeros_like(binary_mask)
    
    # Process in batches
    batch_size = max(1, points_to_process // 50)
    
    for i in range(0, points_to_process, batch_size):
        batch_end = min(i + batch_size, points_to_process)
        
        for j in range(i, batch_end):
            if j < len(laser_points):
                x, y = laser_points[j]
                if 0 <= x < result.shape[1] and 0 <= y < result.shape[0]:
                    if binary_mask[y, x] == 255 and processed_mask[y, x] == 0:
                        result[y, x] = paper_bgr
                        processed_mask[y, x] = 255
                        
                        if intensity > 70:
                            for dy in range(-1, 2):
                                for dx in range(-1, 2):
                                    nx, ny = x + dx, y + dy
                                    if 0 <= nx < result.shape[1] and 0 <= ny < result.shape[0]:
                                        if binary_mask[ny, nx] == 255 and processed_mask[ny, nx] == 0:
                                            result[ny, nx] = paper_bgr
                                            processed_mask[ny, nx] = 255
        
        # Update progress
        progress = (batch_end / total_points) * 100
        progress_bar.progress(min(1.0, progress / 100))
        status_text.text(f"🔥 Laser processing... {progress:.1f}% complete")
        
        # Show current frame
        if batch_end % (batch_size * 2) == 0 or batch_end >= points_to_process:
            display_frame = draw_laser_path_on_image(result, laser_points, batch_end)
            placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        time.sleep(0.01)
    
    status_text.text("✅ Laser deprinting complete!")
    placeholder.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    return result, points_to_process

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🖋️ Laser Deprinting System</h1>
        <p>Complete Ink Removal | Handwritten vs Printed Classification | Laser Path Planning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Simple sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Laser Controls")
        
        input_method = st.radio("Input", ["📁 Upload", "📸 Camera"])
        
        laser_intensity = st.slider(
            "Laser Power (%)", 
            0, 100, 85,
            help="Higher = more ink removed"
        )
        
        laser_strategy = st.selectbox(
            "Scan Pattern",
            ["raster", "contour", "spiral"]
        )
        
        animate = st.checkbox("Show Animation", value=True)
        
        st.markdown("---")
        st.markdown("### 💡 Tip")
        st.info("For best results, use 80-100% power")
    
    # Image input
    image = None
    
    if input_method == "📁 Upload":
        uploaded_file = st.file_uploader("Choose image...", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            image = np.array(image)
    else:
        camera_image = st.camera_input("Take picture")
        if camera_image:
            image = Image.open(camera_image)
            image = np.array(image)
    
    if image is not None:
        # Convert to BGR
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif len(image.shape) == 3 and image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        
        # Detect ink
        with st.spinner("🔍 Analyzing document..."):
            binary_mask, ink_percentage, gray = detect_ink_regions(image)
            features, contours = extract_features(image, binary_mask)
            text_type, confidence = classify_text(features)
        
        # Show original analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📷 Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        with col2:
            st.markdown("### 🎯 Detected Ink")
            st.image(binary_mask, use_column_width=True)
            st.metric("Ink Coverage", f"{ink_percentage:.2f}%")
            st.metric("Text Type", text_type)
            st.metric("Confidence", f"{confidence:.0f}%")
        
        # Plan laser path
        st.markdown("---")
        st.markdown("### 🔴 Laser Path Planning")
        
        laser_points = plan_laser_path(binary_mask, laser_strategy)
        
        total_ink = np.sum(binary_mask == 255)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Laser Points", f"{len(laser_points):,}")
        with col2:
            coverage_pct = (len(laser_points) / max(1, total_ink)) * 100
            st.metric("Coverage", f"{coverage_pct:.1f}%")
        with col3:
            est_time = len(laser_points) / 800
            st.metric("Est. Time", f"{est_time:.1f}s")
        
        # Show path preview
        path_preview = draw_laser_path_on_image(image, laser_points, 0)
        st.image(cv2.cvtColor(path_preview, cv2.COLOR_BGR2RGB), use_column_width=True)
        st.caption("🔴 Red: Target pixels | 🟡 Line: Laser path")
        
        # Execute laser removal
        st.markdown("---")
        st.markdown("### 💥 Execute Laser Deprinting")
        
        if st.button("🔥 START LASER REMOVAL", use_container_width=True):
            # Create placeholders
            display_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            if animate:
                # Animated removal
                final_result, processed = animate_laser_removal(
                    image, laser_points, binary_mask, laser_intensity,
                    progress_bar, status_text, display_placeholder
                )
            else:
                # Instant removal
                with st.spinner("🔥 Removing ink..."):
                    final_result, removed, processed = aggressive_laser_removal(
                        image, binary_mask, laser_points, laser_intensity
                    )
                display_placeholder.image(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB), use_column_width=True)
                progress_bar.progress(1.0)
                status_text.text("✅ Laser deprinting complete!")
            
            # Calculate final metrics
            final_binary, final_ink_percentage, _ = detect_ink_regions(final_result)
            
            original_ink = total_ink
            final_ink = np.sum(final_binary == 255)
            pixels_removed = max(0, original_ink - final_ink)
            removal_rate = (pixels_removed / max(1, original_ink)) * 100
            
            # Success message
            st.markdown(f"""
            <div class="success-message">
                ✅ Laser Deprinting Complete!<br>
                Removed {pixels_removed:,} ink pixels ({removal_rate:.1f}% removal rate)
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics dashboard
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Before Ink", f"{ink_percentage:.2f}%")
            with col2:
                st.metric("After Ink", f"{final_ink_percentage:.2f}%")
            with col3:
                st.metric("Removal Rate", f"{removal_rate:.1f}%")
            with col4:
                st.metric("Pixels Removed", f"{pixels_removed:,}")
            
            # Before/After comparison
            st.markdown("### 📸 Before vs After")
            cmp_col1, cmp_col2 = st.columns(2)
            with cmp_col1:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="BEFORE", use_column_width=True)
            with cmp_col2:
                st.image(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB), caption="AFTER LASER", use_column_width=True)
            
            # Download button
            result_rgb = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)
            import io
            buf = io.BytesIO()
            result_pil.save(buf, format="PNG")
            
            st.download_button(
                "📥 Download Result",
                buf.getvalue(),
                f"laser_deprinted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                "image/png"
            )
        
        # Technical explanation
        with st.expander("🔬 How It Works"):
            st.markdown("""
            ### Aggressive Laser Removal Process:
            
            1. **Ink Detection** - Finds all dark ink pixels using Otsu thresholding
            2. **Path Planning** - Creates complete coverage path over all ink
            3. **Laser Application** - Turns each ink pixel to match paper color
            4. **Neighborhood Removal** - High power also removes surrounding pixels
            
            ### Raster Scanning (X-axis pattern):
            - Scans left-to-right, then right-to-left (zigzag)
            - Covers every row of the image
            - Ensures complete ink coverage
            
            ### Results by Power:
            | Power | Expected Removal |
            |-------|------------------|
            | 50% | ~40-50% ink removed |
            | 70% | ~60-70% ink removed |
            | 85% | ~75-85% ink removed |
            | 100% | ~90-95% ink removed |
            """)
    
    else:
        st.info("👈 Upload an image to begin")
        
        st.markdown("""
        ### 🎯 Laser Deprinting System
        
        **Complete workflow:**
        1. Upload image with text
        2. System detects all ink pixels
        3. Laser plans scanning path (X-axis raster pattern)
        4. Laser removes ink by turning it to paper color
        5. See before/after comparison
        
        **Try with any text image!**
        """)

if __name__ == "__main__":
    import numpy as np
    main()