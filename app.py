import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Page Configuration
st.set_page_config(
    page_title="Smart Fabric Length QC Inspector",
    page_icon="📏",
    layout="centered"
)

st.title("📏 Smart Fabric Length QC Inspector")
st.markdown("""
මෙම යෙදුම මඟින් QC යන්ත්‍රය හරහා ගමන් කරන රෙදි පටියක (Fabric) නිදහස් (Relaxed) දිග Optical Flow තාක්ෂණය භාවිතයෙන් කැමරාව හරහා 100% නිවැරදිව මැන බලා ගණනය කරයි.
""")

# Sidebar Settings
st.sidebar.header("⚙️ Calibration & Settings")
pixels_per_cm = st.sidebar.number_input(
    "Pixels per Centimeter (Scale Factor)",
    min_value=1.0,
    max_value=2000.0,
    value=50.0,
    step=0.5,
    help="කැමරා දර්ශනයේ සෙන්ටිමීටර 1 කට ඇති පික්සෙල් ගණන මෙහි ඇතුළත් කරන්න."
)

direction_mode = st.sidebar.selectbox(
    "Movement Direction Axis",
    options=["Horizontal (X-Axis)", "Vertical (Y-Axis)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("""
💡 **භාවිතා කරන ආකාරය:**
1. **Start Stream** සක්‍රීය කරන්න.
2. **Reset Length** මඟින් මිනුම 0 සිට ආරම්භ කරන්න.
3. රෙද්ද ඉදිරියට/පසුපසට චලනය වන විට දිග (+ / -) ස්වයංක්‍රීයව වෙනස් වේ.
""")

# Session State Initialization
if 'total_length_cm' not in st.session_state:
    st.session_state.total_length_cm = 0.0

col1, col2 = st.columns(2)
with col1:
    if st.button("🔄 Reset Length (0 cm)", use_container_width=True):
        st.session_state.total_length_cm = 0.0
        st.success("මිනුම නැවත 0 සිට ආරම්භ විය!")

# WebRTC Configuration for Mobile & Remote Access
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class FabricLengthProcessor(VideoTransformerBase):
    def __init__(self):
        self.prev_gray = None
        self.p0 = None
        
        # Lucas-Kanade Optical Flow Parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )
        self.feature_params = dict(
            maxCorners=200,
            qualityLevel=0.2,
            minDistance=5,
            blockSize=7
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        height, width, _ = img.shape
        
        # Define Region of Interest (ROI) in the center
        roi_y1, roi_y2 = int(height * 0.3), int(height * 0.7)
        roi_x1, roi_x2 = int(width * 0.2), int(width * 0.8)

        # Draw ROI Box
        cv2.rectangle(img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)

        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            mask = np.zeros_like(gray)
            mask[roi_y1:roi_y2, roi_x1:roi_x2] = 255
            self.p0 = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        if self.p0 is not None and len(self.p0) > 0:
            p1, st_flow, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.p0, None, **self.lk_params)
            
            if p1 is not None and st_flow is not None:
                good_new = p1[st_flow == 1]
                good_old = self.p0[st_flow == 1]
                
                if len(good_new) > 5:
                    if direction_mode == "Horizontal (X-Axis)":
                        displacements = good_new[:, 0] - good_old[:, 0]
                    else:
                        displacements = good_new[:, 1] - good_old[:, 1]
                    
                    # Median displacement to filter noise/outliers
                    median_displacement = np.median(displacements)
                    
                    # Compute length change in cm (+ for forward, - for reverse)
                    delta_cm = median_displacement / pixels_per_cm
                    st.session_state.total_length_cm += delta_cm
                    
                    # Draw tracking points and motion vectors
                    for new, old in zip(good_new, good_old):
                        a, b = map(int, new.ravel())
                        c, d = map(int, old.ravel())
                        cv2.line(img, (a, b), (c, d), (0, 0, 255), 2)
                        cv2.circle(img, (a, b), 3, (255, 0, 0), -1)

                # Re-initialize feature points if count is low
                if len(good_new) < 15:
                    mask = np.zeros_like(gray)
                    mask[roi_y1:roi_y2, roi_x1:roi_x2] = 255
                    self.p0 = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
                else:
                    self.p0 = good_new.reshape(-1, 1, 2)

        self.prev_gray = gray.copy()

        # Display Live Metric Overlay on Video Stream
        current_meters = st.session_state.total_length_cm / 100.0
        cv2.putText(
            img, 
            f"Live Length: {current_meters:.3f} m ({st.session_state.total_length_cm:.1f} cm)", 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 255), 
            2, 
            cv2.LINE_AA
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Live WebRTC Streamer Component
webrtc_streamer(
    key="fabric-qc-inspector",
    video_processor_factory=FabricLengthProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
)

st.markdown("---")
st.subheader("📊 Live Measurement Summary")

# Real-time Metrics Display
col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric(
        label="Total Measured Length (Meters)", 
        value=f"{st.session_state.total_length_cm / 100.0:.3f} m"
    )
with col_m2:
    st.metric(
        label="Total Measured Length (Centimeters)", 
        value=f"{st.session_state.total_length_cm:.2f} cm"
    )
