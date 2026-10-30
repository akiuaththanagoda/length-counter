import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Page Configuration
st.set_page_config(page_title="Ocean Lanka - Live Fabric QC", layout="wide")

st.title("🧵 Ocean Lanka - Mobile Live Fabric Length Counter")
st.markdown("📱 දුරකථනයෙන් කැමරාව ක්‍රියාත්මක කර රෙද්ද ගෙන යන විට, තිරය මතම සජීවීව දිග වෙනස් වීම (+ සහ -) බලාගත හැක.")

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
        self.motion_color = (255, 255, 255)

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
            
            # Motion State & Colors
            if avg_dy > motion_threshold:
                self.status_text = "FORWARD (+)"
                self.motion_color = (0, 255, 0) # Green for forward
            elif avg_dy < -motion_threshold:
                self.status_text = "ROLLBACK (-)"
                self.motion_color = (0, 0, 255) # Red for backward/rollback
            else:
                self.status_text = "STATIONARY"
                self.motion_color = (200, 200, 200)
        
        self.prev_gray = curr_gray
        
        # --- CLEAR & LARGE HUD OVERLAY ON MOBILE SCREEN ---
        h, w, _ = img.shape
        
        # Draw background box for text at the top
        cv2.rectangle(img, (10, 10), (min(w - 10, 500), 130), (0, 0, 0), -1)
        
        # Display Live Meters (Large font for easy viewing on phone)
        cv2.putText(img, f"LEN: {self.current_meters:.3f} m", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        
        # Display Centimeters
        cv2.putText(img, f"({self.total_distance_cm:.1f} cm)", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display Direction Status (+ / -)
        cv2.putText(img, f"DIR: {self.status_text}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.motion_color, 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# --- UI LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📱 Live Phone Camera Stream")
    ctx = webrtc_streamer(
        key="fabric-qc-mobile",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=FabricProcessor,
        media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("📊 Roll Controls")
    
    # Placeholder to show final saved length
    final_display = st.empty()
    
    if st.button("🔄 Reset Length to 0", use_container_width=True):
        if ctx.video_processor:
            ctx.video_processor.total_distance_cm = 0.0
            st.success("Counter reset to 0!")
            
    if st.button("🏁 End Roll & Get Final Length", use_container_width=True):
        if ctx.video_processor:
            final_m = ctx.video_processor.current_meters
            final_cm = ctx.video_processor.total_distance_cm
            final_display.success(f"🎉 **Final Roll Length:**\n\n🔹 **{final_m:.3f} meters**\n🔹 ({final_cm:.2f} cm)")

st.info("💡 **භාවිතා කරන ආකාරය:** දුරකථන බ්‍රව්සරයෙන් ඇප් එක විවෘත කර **START** දෙන්න. රෙද්ද ඉදිරියට යන විට තිරය උඩින් දිග වැඩි වන ආකාරයත්, ආපස්සට යන විට අඩුවන ආකාරයත් සජීවීව බලාගත හැක.")
