import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Page Configuration with Industrial Dark Theme Feel
st.set_page_config(
    page_title="Ocean Lanka - Virtual Encoder QC System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Industrial LED/Digital Display Look
st.markdown("""
    <style>
    .metric-card {
        background-color: #0e1117;
        border: 2px solid #1f2937;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .led-meter {
        font-size: 45px;
        font-weight: bold;
        color: #00ffcc;
        font-family: 'Courier New', Courier, monospace;
    }
    .led-status {
        font-size: 24px;
        font-weight: bold;
        font-family: 'Courier New', Courier, monospace;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧵 Ocean Lanka - Virtual Optical Encoder Fabric QC")
st.markdown("⚡ **Virtual Encoder Mode:** කැමරාව සජීවී සංවේදකයක් (Sensor) ලෙස ක්‍රියා කරමින් රෙද්දේ චලනයන් (+ / -) තත්පරයෙන් තත්පරයට ගණනය කරයි.")

# --- SIDEBAR: ENCODER CONFIGURATION ---
st.sidebar.header("⚙️ Encoder Calibration")

px_per_cm = st.sidebar.number_input(
    "Encoder Scale (Pixels per cm)",
    min_value=0.1,
    max_value=500.0,
    value=25.0,
    step=0.5,
    help="කැමරා දර්ශනය තුළ 1cm සඳහා ඇති Pixel ගණන."
)

motion_threshold = st.sidebar.slider("Noise Filter (Deadzone)", min_value=0.01, max_value=2.0, value=0.15, step=0.05)
direction_invert = st.sidebar.checkbox("Invert Encoder Direction (+/-)", value=False)

# --- VIRTUAL ENCODER PROCESSOR CLASS ---
class VirtualEncoderProcessor:
    def __init__(self):
        self.prev_gray = None
        self.total_distance_cm = 0.0
        self.current_meters = 0.0
        self.motion_state = "IDLE (0)"
        self.speed_indicator = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        curr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is not None:
            # High-precision Optical Flow for Virtual Encoder pulses
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            flow_y = flow[..., 1]
            avg_dy = np.mean(flow_y)
            
            if direction_invert:
                avg_dy = -avg_dy
            
            # Filter minor vibrations/noise
            if abs(avg_dy) > motion_threshold:
                delta_cm = avg_dy / px_per_cm
                self.total_distance_cm += delta_cm
                self.speed_indicator = abs(avg_dy)
            else:
                self.speed_indicator = 0.0
            
            # Encoder lower bound limit protection
            if self.total_distance_cm < 0:
                self.total_distance_cm = 0.0
            
            self.current_meters = self.total_distance_cm / 100.0
            
            # Determine Encoder Motion State
            if avg_dy > motion_threshold:
                self.motion_state = "FORWARD (+)"
            elif avg_dy < -motion_threshold:
                self.motion_state = "ROLLBACK (-)"
            else:
                self.motion_state = "STATIONARY"
        
        self.prev_gray = curr_gray
        
        # --- RENDER VIRTUAL ENCODER HUD ON CAMERA FEED ---
        h, w, _ = img.shape
        
        # Top-left Industrial HUD Panel
        cv2.rectangle(img, (10, 10), (420, 120), (10, 10, 10), -1)
        cv2.rectangle(img, (10, 10), (420, 120), (0, 255, 204), 2)
        
        cv2.putText(img, f"LEN: {self.current_meters:.3f} m", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 204), 2)
        cv2.putText(img, f"({self.total_distance_cm:.1f} cm)", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        state_color = (0, 255, 0) if "FORWARD" in self.motion_state else ((0, 0, 255) if "ROLLBACK" in self.motion_state else (200, 200, 200))
        cv2.putText(img, f"ENCODER: {self.motion_state}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# --- LAYOUT: LIVE CAMERA & VIRTUAL ENCODER DISPLAY ---
col_cam, col_panel = st.columns([2, 1])

with col_cam:
    st.subheader("🔴 Live Camera Sensor Feed")
    ctx = webrtc_streamer(
        key="virtual-encoder-qc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=VirtualEncoderProcessor,
        media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
        async_processing=True,
    )

with col_panel:
    st.subheader("📊 Virtual Encoder Readout")
    
    # Industrial display containers
    display_m = st.empty()
    display_cm = st.empty()
    display_status = st.empty()
    
    st.divider()
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("🔄 Reset Encoder (0)", use_container_width=True):
            if ctx.video_processor:
                ctx.video_processor.total_distance_cm = 0.0
                st.success("Reset OK!")
    with col_b2:
        if st.button("🏁 End Roll", use_container_width=True):
            if ctx.video_processor:
                final_m = ctx.video_processor.current_meters
                final_cm = ctx.video_processor.total_distance_cm
                st.success(f"Final: {final_m:.3f} m ({final_cm:.1f} cm)")
                ctx.video_processor.total_distance_cm = 0.0

# Real-time synchronization loop for encoder UI readout
if ctx.state.playing and ctx.video_processor:
    m_val = ctx.video_processor.current_meters
    cm_val = ctx.video_processor.total_distance_cm
    status_val = ctx.video_processor.motion_state
    
    display_m.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 14px; color: #9ca3af;">TOTAL LENGTH (METERS)</div>
            <div class="led-meter">{m_val:.3f} m</div>
        </div>
    """, unsafe_allow_html=True)
    
    display_cm.markdown(f"""
        <div class="metric-card" style="margin-top: 10px;">
            <div style="font-size: 14px; color: #9ca3af;">TOTAL LENGTH (CM)</div>
            <div style="font-size: 28px; color: #60a5fa; font-family: monospace;">{cm_val:.2f} cm</div>
        </div>
    """, unsafe_allow_html=True)
    
    display_status.info(f"⚡ Encoder State: **{status_val}**")
else:
    display_m.markdown("""
        <div class="metric-card">
            <div style="font-size: 14px; color: #9ca3af;">TOTAL LENGTH (METERS)</div>
            <div class="led-meter">0.000 m</div>
        </div>
    """, unsafe_allow_html=True)
    display_cm.markdown("""
        <div class="metric-card" style="margin-top: 10px;">
            <div style="font-size: 14px; color: #9ca3af;">TOTAL LENGTH (CM)</div>
            <div style="font-size: 28px; color: #60a5fa; font-family: monospace;">0.00 cm</div>
        </div>
    """, unsafe_allow_html=True)
    display_status.warning("⚠️ Encoder Offline. Click 'START' on camera feed.")
