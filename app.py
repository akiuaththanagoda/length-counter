import cv2
import numpy as np
import streamlit as st
import time

st.set_page_config(page_title="Ocean Lanka - Optical Flow Fabric Length Counter", layout="wide")

st.title("🧵 Ocean Lanka - QC Live Continuous Fabric Length Counter (Optical Flow)")
st.caption("USB Camera Inspection System for Tension-Free Real-time Fabric Measurement")

st.sidebar.header("⚙️ System Calibration & Setup")
camera_index = st.sidebar.number_input("USB Camera Index (e.g., 0, 1, 2)", min_value=0, max_value=5, value=0, step=1)

px_per_cm = st.sidebar.number_input(
    "Calibration Factor (Pixels / cm)",
    min_value=0.1,
    max_value=500.0,
    value=25.0,
    step=0.5,
    help="1 cm සඳහා කැමරාවට පෙනෙන Pixel ගණන ඇතුළත් කරන්න."
)

direction_invert = st.sidebar.checkbox("Invert Motion Direction (+/-)", value=False)
motion_threshold = st.sidebar.slider("Noise Reduction Threshold", min_value=0.1, max_value=5.0, value=0.5, step=0.1)

if "total_distance_cm" not in st.session_state:
    st.session_state.total_distance_cm = 0.0
if "is_measuring" not in st.session_state:
    st.session_state.is_measuring = False

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("▶️ Start Inspection", use_container_width=True):
        st.session_state.is_measuring = True

with col2:
    if st.button("⏸️ Pause Inspection", use_container_width=True):
        st.session_state.is_measuring = False

with col3:
    if st.button("🏁 End Current Roll & Save", use_container_width=True):
        final_len = st.session_state.total_distance_cm
        st.success(f"🎉 Roll Inspection Finished! Final Length: **{final_len/100:.2f} m** ({final_len:.2f} cm)")
        st.session_state.total_distance_cm = 0.0
        st.session_state.is_measuring = False

with col4:
    if st.button("🔄 Reset Counter to 0", use_container_width=True):
        st.session_state.total_distance_cm = 0.0

st.divider()

kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
metric_m = kpi_col1.empty()
metric_cm = kpi_col2.empty()
status_indicator = kpi_col3.empty()

video_placeholder = st.empty()

metric_m.metric(label="Total Measured Distance (Meters)", value=f"{st.session_state.total_distance_cm / 100:.3f} m")
metric_cm.metric(label="Total Distance (Centimeters)", value=f"{st.session_state.total_distance_cm:.2f} cm")
status_indicator.info("Status: Ready to Inspect")

if st.session_state.is_measuring:
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        st.error(f"❌ Camera at index {camera_index} could not be accessed. Please check USB connection.")
        st.session_state.is_measuring = False
    else:
        ret, prev_frame = cap.read()
        if ret:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            status_indicator.success("Status: 🟢 Live Measuring...")
            
            while st.session_state.is_measuring:
                ret, current_frame = cap.read()
                if not ret:
                    st.warning("⚠️ Frame drop detected or camera disconnected.")
                    break
                
                current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, 
                    current_gray, 
                    None, 
                    pyr_scale=0.5, 
                    levels=3, 
                    winsize=15, 
                    iterations=3, 
                    poly_n=5, 
                    poly_sigma=1.2, 
                    flags=0
                )
                
                flow_y = flow[..., 1] 
                avg_dy_px = np.mean(flow_y)
                
                if direction_invert:
                    avg_dy_px = -avg_dy_px
                
                if abs(avg_dy_px) > motion_threshold:
                    delta_cm = avg_dy_px / px_per_cm
                    st.session_state.total_distance_cm += delta_cm
                
                if st.session_state.total_distance_cm < 0:
                    st.session_state.total_distance_cm = 0.0

                output_frame = current_frame.copy()
                h, w, _ = output_frame.shape
                
                cv2.rectangle(output_frame, (10, 10), (450, 90), (0, 0, 0), -1)
                
                current_m = st.session_state.total_distance_cm / 100.0
                cv2.putText(output_frame, f"LIVE LENGTH: {current_m:.3f} m", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                motion_dir_text = "FORWARD (+)" if avg_dy_px > motion_threshold else ("BACKWARD (-)" if avg_dy_px < -motion_threshold else "STOPPED (0)")
                color = (0, 255, 255) if "FORWARD" in motion_dir_text else ((0, 0, 255) if "BACKWARD" in motion_dir_text else (255, 255, 255))
                
                cv2.putText(output_frame, f"DIR: {motion_dir_text}", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                if abs(avg_dy_px) > motion_threshold:
                    arrow_start = (w - 50, h // 2)
                    arrow_end = (w - 50, (h // 2) + int(np.sign(avg_dy_px) * 50))
                    cv2.arrowedLine(output_frame, arrow_start, arrow_end, color, 3)

                metric_m.metric(label="Total Measured Distance (Meters)", value=f"{current_m:.3f} m")
                metric_cm.metric(label="Total Distance (Centimeters)", value=f"{st.session_state.total_distance_cm:.2f} cm")
                
                frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                prev_gray = current_gray
                
            cap.release()
