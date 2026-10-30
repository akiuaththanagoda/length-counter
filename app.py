import cv2
import numpy as np
import streamlit as st
import tempfile
import os

st.set_page_config(page_title="Relaxed Fabric Length QC System", layout="wide")

st.title("🧵 Ocean Lanka - Advanced Computer Vision Relaxed Fabric Length Measurement")
st.markdown("මෙමඟින් ටෙන්ෂන් එක නිසා එන්කෝඩර් එකෙන් වැඩිපුර පෙන්නන දිග මඟහරවා, රෙද්දේ සැබෑ (Relaxed) දිග කැමරාව මඟින් නිවැරදිව ගණනය කරයි.")

# Sidebar Settings
st.sidebar.header("⚙️ Calibration & Settings")
calibration_factor = st.sidebar.number_input(
    "Pixels per Centimeter (px/cm)", 
    min_value=1.0, 
    max_value=100.0, 
    value=10.0, 
    step=0.1,
    help="කැමරා රාමුව තුළ 1cm කට අදාළ පික්සල් ගණන මෙහි ඇතුළත් කරන්න."
)

color_threshold = st.sidebar.slider(
    "Fabric Color Sensitivity (Threshold)", 
    min_value=10, 
    max_value=200, 
    value=50
)

app_mode = st.sidebar.selectbox("Choose Mode", ["Upload Video", "Live Camera Feed"])

def process_frame(frame, px_per_cm):
    """
    Frame එකක් පරික්ෂා කර රෙද්දේ කොන්ටූර් (Contour) සොයා පික්සල් මඟින් දිග මැන සෙන්ටිමීටර වලට හරවයි.
    """
    height, width, _ = frame.shape
    
    # Preprocessing: Convert to Grayscale & Blur to remove noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Thresholding / Edge Detection to isolate fabric in the relax zone
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours of the fabric
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_length_cm = 0.0
    output_frame = frame.copy()
    
    if contours:
        # Find the largest contour representing the relaxed fabric segment
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) > 1000: # Filter small noise
            # Get rotated bounding box to handle angled fabric flow
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            
            # Draw contour and bounding box
            cv2.drawContours(output_frame, [box], 0, (0, 255, 0), 2)
            
            # Calculate length (taking the maximum dimension of the bounding box)
            width_px = rect[1][0]
            height_px = rect[1][1]
            fabric_length_px = max(width_px, height_px)
            
            # Convert pixels to centimeters / meters
            max_length_cm = fabric_length_px / px_per_cm
            
            # Display measurement text on frame
            center_x, center_y = int(rect[0][0]), int(rect[0][1])
            cv2.putText(output_frame, f"Length: {max_length_cm:.2f} cm", 
                        (center_x - 50, center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return output_frame, max_length_cm

# --- MODE 1: UPLOAD VIDEO ---
if app_mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload Fabric QC Video (.mp4, .avi)", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        cap = cv2.VideoCapture(video_path)
        st.video(uploaded_file)
        
        if st.button("▶️ Process Video for Relaxed Length"):
            stt = st.empty()
            stt.text("ප්‍රොසෙස් වෙමින් පවතී...")
            
            frame_placeholder = st.empty()
            metric_placeholder = st.empty()
            
            total_lengths = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, length_cm = process_frame(frame, calibration_factor)
                if length_cm > 0:
                    total_lengths.append(length_cm)
                
                # Show processed frame in Streamlit
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)
            
            cap.release()
            stt.text("විමර්ෂණය සාර්ථකයි!")
            
            if total_lengths:
                avg_length = np.mean(total_lengths)
                metric_placeholder.success(f"✅ සකස් කළ සැබෑ රෙදි දිග (Average Relaxed Length): **{avg_length:.2f} cm** ({avg_length/100:.2f} මීටර්)")

# --- MODE 2: LIVE CAMERA FEED ---
elif app_mode == "Live Camera Feed":
    st.warning("ලයිව් කැමරාව ක්‍රියාත්මක කිරීමට ප්‍රාදේශීය පරිගණකයේ OpenCV සහ වෙබ් කැමරාව කන්ෆිගර් කර තිබිය යුතුය.")
    run_cam = st.checkbox("Turn on Live Camera")
    
    camera_placeholder = st.empty()
    length_display = st.empty()
    
    if run_cam:
        cap = cv2.VideoCapture(0) # 0 for default webcam or IP camera stream URL
        while run_cam:
            ret, frame = cap.read()
            if not ret:
                st.error("කැමරාව වෙත සම්බන්ධ වීම අසාර්ථකයි.")
                break
            
            processed_frame, length_cm = process_frame(frame, calibration_factor)
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            camera_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)
            length_display.metric(label="Measured Relaxed Length", value=f"{length_cm:.2f} cm")
            
        cap.release()
