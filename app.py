import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Page Configuration
st.set_page_config(page_title="Ocean Lanka - Phone Camera Fabric QC", layout="wide")

st.title("🧵 Ocean Lanka - Phone / Browser Camera Fabric Length Counter")
st.markdown("ඔබගේ **ස්මාර්ට්ෆෝන් කැමරාව** (သို့ත් බ්‍රව්සර් කැමරාව) භාවිතයෙන් ක්ලවුඩ් සර්වර් එක හරහා සජීවීව රෙදි දිග මැනීමේ පද්ධතිය.")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Calibration & Settings")

px_per_cm = st.sidebar.number_input(
    "Calibration Factor (Pixels per cm)",
    min_value=0.1,
    max_value=500.0,
    value=25.0,
    step=0.5,
    help="1cm සඳහා කැමරාවට පෙනෙන Pixel ගණන."
)

motion_threshold = st.sidebar.slider("Noise Filter Threshold", min_value=0.05, max_value=2.0, value=0.3, step=0.05)
direction_invert = st.sidebar.checkbox("Invert Motion Direction (+/-)", value=False)

# --- WEBRTC VIDEO PROCESSOR CLASS ---
class FabricProcessor:
    def __init__(self):
        self.prev_gray = None
        self.total_distance_cm = 0.0
        self.current_meters = 0.0
        self.status_text = "READY"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        curr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is not None:
            # Dense Optical Flow Calculation (Farneback)
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            flow_y = flow[..., 1]
            avg_dy = np.mean(flow_y)
            
            if direction_invert:
                avg_dy = -avg_dy
            
            # Accumulate distance if movement exceeds noise threshold
            if abs(avg_dy) > motion_threshold:
                delta_cm = avg_dy / px_per_cm
                self.total_distance_cm += delta_cm
            
            # Prevent negative values below zero
            if self.total_distance_cm < 0:
                self.total_distance_cm = 0.0
            
            self.current_meters = self.total_distance_cm / 100.0
            
            # Motion State Text
            if avg_dy > motion_threshold:
                self.status_text = "MOVING FORWARD (+)"
            elif avg_dy < -motion_threshold:
                self.status_text = "ROLLING BACK (-)"
            else:
                self.status_text = "STATIONARY (0)"
        
        self.prev_gray = curr_gray
        
        # --- RENDER HUD OVERLAY ON VIDEO ---
        cv2.rectangle(img, (15, 15), (460, 100), (20, 20, 20), -1)
        cv2.putText(img, f"LIVE LEN: {self.current_meters:.3f} m", (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"STATUS: {self.status_text}", (25, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# STUN server configuration for WebRTC streaming
rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# --- LAYOUT & CONTROLS ---
col_video, col_control = st.columns([2, 1])

with col_video:
    st.markdown("### 📱 Live Camera Feed")
    # WebRTC Streamer allows using phone's rear camera ("environment")
    ctx = webrtc_streamer(
        key="fabric-qc-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=FabricProcessor,
        media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
        async_processing=True,
    )

with col_control:
    st.markdown("### 📊 Roll Control Panel")
    
    if st.button("🔄 Reset Distance to 0", use_container_width=True):
        if ctx.video_processor:
            ctx.video_processor.total_distance_cm = 0.0
            st.success("Counter reset successfully!")
            
    if st.button("🏁 End Roll & Save Length", use_container_width=True):
        if ctx.video_processor:
            final_len_m = ctx.video_processor.current_meters
            final_len_cm = ctx.video_processor.total_distance_cm
            st.success(f"🎉 Roll Saved! Final Length: **{final_len_m:.3f} m** ({final_len_cm:.2f} cm)")
            ctx.video_processor.total_distance_cm = 0.0

st.info("💡 **පාවිච්චි කරන ආකාරය:** Streamlit Cloud ලින්ක් එක ඔබගේ ජංගම දුරකථනයේ (Phone Browser) විවෘත කර, කැමරාවට අවසර දී (Allow), පසුපස කැමරාව (Rear Camera) මඟින් රෙදි පටිය නිරීක්ෂණය කරන්න.")
