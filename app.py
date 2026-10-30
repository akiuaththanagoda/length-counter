import cv2
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(page_title="Ocean Lanka - Fabric QC Length Counter", layout="wide")

st.title("🧵 Ocean Lanka - Continuous Fabric Length Counter (Multi-Camera QC)")
st.markdown("ටෙන්ෂන් වලින් තොරව සැබෑ දිග (Continuous Length) මැනීම සඳහා බිල්ට් කැමරාව හෝ වයර් සහිත USB කැමරාව භාවිත කරන්න.")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Camera & Calibration Setup")

# Camera Source Selection
camera_source = st.sidebar.radio(
    "Select Camera Source",
    ["Built-in Device Camera", "Wired USB Camera"]
)

if camera_source == "Built-in Device Camera":
    camera_index = 0
    st.sidebar.info("ලැප්ටොප්/පරිගණකයේ බිල්ට්-ඉන් කැමරාව තෝරා ඇත (Index: 0).")
else:
    camera_index = st.sidebar.number_input(
        "Wired USB Camera Index", 
        min_value=0, 
        max_value=10, 
        value=1, 
        step=1,
        help="බාහිර USB කැමරාව සඳහා අංකය (උදා: 1 හෝ 2) දෙන්න."
    )

px_per_cm = st.sidebar.number_input(
    "Calibration Factor (Pixels per cm)",
    min_value=0.1,
    max_value=500.0,
    value=25.0,
    step=0.5,
    help="1cm සඳහා කැමරාවට පෙනෙන Pixel ගණන."
)

direction_invert = st.sidebar.checkbox("Invert Motion Direction (+/-)", value=False)
motion_threshold = st.sidebar.slider("Noise Filter Threshold", min_value=0.05, max_value=2.0, value=0.3, step=0.05)

# --- SESSION STATE MANAGEMENT ---
if "total_distance_cm" not in st.session_state:
    st.session_state.total_distance_cm = 0.0
if "is_measuring" not in st.session_state:
    st.session_state.is_measuring = False

# --- CONTROL BUTTONS ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("▶️ Start Live Inspection", use_container_width=True):
        st.session_state.is_measuring = True

with col2:
    if st.button("⏸️ Pause Inspection", use_container_width=True):
        st.session_state.is_measuring = False

with col3:
    if st.button("🏁 End Roll & Save Final Length", use_container_width=True):
        final_len = st.session_state.total_distance_cm
        st.success(f"🎉 Roll Completed Successfully! Final Fabric Length: **{final_len/100:.2f} meters** ({final_len:.2f} cm)")
        st.session_state.total_distance_cm = 0.0
        st.session_state.is_measuring = False

with col4:
    if st.button("🔄 Reset to 0", use_container_width=True):
        st.session_state.total_distance_cm = 0.0

st.divider()

# --- LIVE METRICS DASHBOARD ---
metric_col1, metric_col2, metric_col3 = st.columns(3)
m_display = metric_col1.empty()
cm_display = metric_col2.empty()
status_display = metric_col3.empty()

video_box = st.empty()

# Set initial values
m_display.metric("Total Length (Meters)", f"{st.session_state.total_distance_cm / 100:.3f} m")
cm_display.metric("Total Length (Centimeters)", f"{st.session_state.total_distance_cm:.2f} cm")
status_display.info("Status: Ready (Click Start Live Inspection)")

# --- MAIN INSPECTION LOOP ---
if st.session_state.is_measuring:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
    
    if not cap.isOpened():
        st.error(f"❌ කැමරාව (Index: {camera_index}) වෙත සම්බන්ධ වීම අසාර්ථකයි! කරුණාකර කැමරා සම්බන්ධතාවය හෝ Index අංකය පරීක්ෂා කරන්න.")
        st.session_state.is_measuring = False
    else:
        # Set resolution for stable processing
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        ret, prev_frame = cap.read()
        if ret:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            status_display.success("Status: 🟢 Live Measuring Active...")
            
            while st.session_state.is_measuring:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ කැමරා දර්ශන ලබා ගැනීම ඇණහිට ඇත (Camera feed interrupted).")
                    break
                
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Optical Flow Calculation (Dense Farneback)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                
                # Track vertical flow movement (Y-axis displacement)
                flow_y = flow[..., 1]
                avg_dy = np.mean(flow_y)
                
                if direction_invert:
                    avg_dy = -avg_dy
                
                # Accumulate distance if movement exceeds noise threshold
                if abs(avg_dy) > motion_threshold:
                    delta_cm = avg_dy / px_per_cm
                    st.session_state.total_distance_cm += delta_cm
                
                # Prevent negative values below zero
                if st.session_state.total_distance_cm < 0:
                    st.session_state.total_distance_cm = 0.0
                
                # --- RENDER OVERLAYS ON VIDEO ---
                output_image = frame.copy()
                h, w, _ = output_image.shape
                
                # HUD Background Box
                cv2.rectangle(output_image, (15, 15), (460, 100), (20, 20, 20), -1)
                
                current_meters = st.session_state.total_distance_cm / 100.0
                cv2.putText(output_image, f"LIVE LEN: {current_meters:.3f} m", (25, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Motion State Detection Text
                if avg_dy > motion_threshold:
                    motion_text, motion_color = "MOVING FORWARD (+)", (0, 255, 255)
                elif avg_dy < -motion_threshold:
                    motion_text, motion_color = "ROLLING BACK (-)", (0, 0, 255)
                else:
                    motion_text, motion_color = "STATIONARY (0)", (200, 200, 200)
                
                cv2.putText(output_image, f"STATUS: {motion_text}", (25, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, motion_color, 2)
                
                # Live UI Refresh in Streamlit
                m_display.metric("Total Length (Meters)", f"{current_meters:.3f} m")
                cm_display.metric("Total Length (Centimeters)", f"{st.session_state.total_distance_cm:.2f} cm")
                
                # Convert BGR to RGB for Streamlit display
                frame_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                video_box.image(frame_rgb, channels="RGB", use_column_width=True)
                
                prev_gray = curr_gray
                
            cap.release()
