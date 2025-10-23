import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import time

# Page configuration
st.set_page_config(
    page_title="Détecteur de Tumeur",
    page_icon="",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #1f77b4;
        font-size: 3rem !important;
        font-weight: bold;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("Détecteur de Tumeur")
st.markdown('<p class="subtitle">Upload an image and detect Tumeur objects</p>', unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.divider()

# Main layout - Two columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.header("📤 Upload Image")
    
    uploaded_image = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a football image for detection"
    )
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Original Image", use_container_width=True)
        
        st.divider()
        
        # Detection button
        if st.button("🚀 Run Detection", type="primary", use_container_width=True):
            with st.spinner("🔍 Detecting objects..."):
                start_time = time.time()
                
                # Run detection
                results = model(image, conf=0.25, verbose=False)
                
                inference_time = time.time() - start_time
                
                # Store results in session state
                st.session_state['results'] = results
                st.session_state['inference_time'] = inference_time
                st.session_state['image_processed'] = True

with col2:
    st.header("📊 Detection Results")
    
    if 'image_processed' in st.session_state and st.session_state['image_processed']:
        results = st.session_state['results']
        
        # Plot results
        res_plotted = results[0].plot()
        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        st.image(res_plotted_rgb, caption="Detected Objects", use_container_width=True)
        
        st.divider()
        
        # Statistics
        st.subheader("📈 Statistics")
        
        num_detections = len(results[0].boxes)
        inference_time = st.session_state['inference_time']
        
        # Metrics in columns
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric(
                label="Objects Detected",
                value=num_detections,
                delta=None
            )
        
        with metric_col2:
            st.metric(
                label="Inference Time",
                value=f"{inference_time:.3f}s"
            )
        
        with metric_col3:
            st.metric(
                label="FPS",
                value=f"{1/inference_time:.1f}"
            )
        
        # Detailed detections table
        if num_detections > 0:
            st.divider()
            st.subheader("🔍 Detected Objects Details")
            
            detection_data = []
            for i, box in enumerate(results[0].boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = results[0].names[class_id]
                bbox = box.xyxy[0].cpu().numpy()
                
                detection_data.append({
                    "#": i + 1,
                    "Object": class_name,
                    "Confidence": f"{confidence:.2%}",
                    "X1": int(bbox[0]),
                    "Y1": int(bbox[1]),
                    "X2": int(bbox[2]),
                    "Y2": int(bbox[3])
                })
            
            st.dataframe(detection_data, use_container_width=True, hide_index=True)
            
            # Download button
            st.divider()
            import io
            result_img = Image.fromarray(res_plotted_rgb)
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="⬇️ Download Result Image",
                data=byte_im,
                file_name="football_detection_result.png",
                mime="image/png",
                use_container_width=True
            )
    else:
        st.info("👈 Upload an image and click 'Run Detection' to see results here")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Détecteur de Tumeur</p>
    </div>
""", unsafe_allow_html=True)