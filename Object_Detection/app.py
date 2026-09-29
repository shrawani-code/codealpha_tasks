import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os


# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="AI Object Detection & Tracking",
    page_icon="🎯",
    layout="wide"
)


# ==============================
# CUSTOM CSS
# ==============================

st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
}

.info-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ==============================
# TITLE
# ==============================

st.markdown(
    '<div class="main-title">🎯 AI Object Detection & Tracking</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Detect objects in images and track them in videos using YOLO</div>',
    unsafe_allow_html=True
)

st.write("")


# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


model = load_model()


# ==============================
# SIDEBAR
# ==============================

st.sidebar.title("⚙️ Settings")

mode = st.sidebar.radio(
    "Choose a mode:",
    ["🖼️ Image Detection", "🎥 Video Tracking"]
)


# ==============================
# IMAGE DETECTION
# ==============================

if mode == "🖼️ Image Detection":

    st.header("🖼️ Object Detection")

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image)

        st.subheader("Original Image")
        st.image(image, use_container_width=True)

        if st.button("🔍 Detect Objects"):

            with st.spinner("Detecting objects..."):

                results = model(image)

                result_image = results[0].plot()

            st.success("Object detection completed!")

            st.subheader("🎯 Detection Result")

            st.image(
                result_image,
                channels="BGR",
                use_container_width=True
            )


# ==============================
# VIDEO TRACKING
# ==============================

else:

    st.header("🎥 Object Tracking")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button("🚀 Start Tracking"):

            with st.spinner("Tracking objects..."):

                # Save uploaded video
                input_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )

                input_file.write(uploaded_video.read())
                input_file.close()

                # Open input video
                cap = cv2.VideoCapture(input_file.name)

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # Permanent output file
                output_path = "tracked_output.mp4"

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                out = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    fps,
                    (width, height)
                )

                while True:

                    success, frame = cap.read()

                    if not success:
                        break

                    results = model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml"
                    )

                    annotated_frame = results[0].plot()

                    out.write(annotated_frame)

                cap.release()
                out.release()

                os.remove(input_file.name)

            st.success("✅ Object tracking completed!")

            st.subheader("🎯 Tracking Result")

            # Read processed video
            with open(output_path, "rb") as video_file:
                video_bytes = video_file.read()

            # Display processed video
            st.video(video_bytes)

            # Download option
            st.download_button(
                label="⬇️ Download Tracked Video",
                data=video_bytes,
                file_name="tracked_output.mp4",
                mime="video/mp4"
            )