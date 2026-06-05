"""
frontend.py  —  Seat Belt Detection System
==========================================
Streamlit web app with THREE input modes:
  1. Upload Image     (existing feature — unchanged)
  2. Live Camera      (NEW feature added)
  3. Multiple Images  (existing feature — unchanged)

HOW CAMERA WORKS (simple explanation):
  Streamlit has a built-in function called st.camera_input()
  It opens your webcam directly inside the browser.
  When you click the capture button, it returns the photo as bytes.
  We convert those bytes into a NumPy array (same format as uploaded images).
  Then we pass it through the exact same prediction pipeline.
  Result is shown exactly like an uploaded image.

RUN:
  streamlit run frontend.py

NO NEW DEPENDENCIES NEEDED:
  st.camera_input() is built into Streamlit >= 1.10.0
  If you get an error, run:  pip install --upgrade streamlit
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import io          # converts camera bytes to image
import os          # file path checks
import time        # inference timing
import tempfile    # temporary video files
import pandas as pd

import cv2                          # image drawing and processing
import numpy as np                  # array operations
import streamlit as st              # web UI
import matplotlib.pyplot as plt     # confidence chart
from PIL import Image               # image format conversion
from tensorflow import keras        # load trained model


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Seat Belt Detector",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS  —  clean dark theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Result cards */
.card-safe {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1px solid #16a34a;
    border-left: 5px solid #22c55e;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    margin: 10px 0;
}
.card-unsafe {
    background: linear-gradient(135deg, #1c0505, #450a0a);
    border: 1px solid #dc2626;
    border-left: 5px solid #ef4444;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    margin: 10px 0;
}
.card-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: 0.03em;
}
.card-safe   .card-title { color: #4ade80; }
.card-unsafe .card-title { color: #f87171; }
.card-conf {
    font-size: 0.9rem;
    color: #94a3b8;
    margin-top: 6px;
}

/* Camera section box */
.camera-box {
    background: #0d1526;
    border: 2px dashed #1e3a5f;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 10px;
}

/* Metric boxes */
[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 12px 16px;
}

/* Divider */
hr { border-color: #1e293b !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH     = 'seatbelt_model.h5'
SAFE_THRESHOLD = 70   # model must be 70%+ confident to say SAFE


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADER
#  @st.cache_resource → loads model only ONCE per session (not every rerun)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the trained seatbelt model from disk. Cached after first load."""
    if not os.path.exists(MODEL_PATH):
        return None
    return keras.models.load_model(MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
#  CORE PREDICTION FUNCTION
#  This is the SINGLE function used by ALL three input modes
#  (upload, camera, multi-image) — same logic, no duplication
# ─────────────────────────────────────────────────────────────────────────────

def predict(model, img_rgb):
    """
    Takes any RGB NumPy image array and returns prediction.

    Steps:
      1. Read model's expected input size automatically
      2. Resize image to that size
      3. Normalise pixel values from 0-255 to 0.0-1.0
      4. Reshape to [1, height, width, 3] (batch of 1 image)
      5. Run model.predict()
      6. Apply safety threshold

    Parameters:
      model   : loaded Keras model
      img_rgb : RGB NumPy array of any size

    Returns:
      label      : 0 = seatbelt, 1 = no seatbelt
      confidence : float, confidence % of the chosen label
      belt_conf  : float, confidence % specifically for seatbelt class
      ms         : inference time in milliseconds
    """
    t0 = time.perf_counter()

    # Step 1: Auto-detect what size model expects
    # model.input_shape returns e.g. (None, 160, 160, 3)
    input_shape = model.input_shape
    h, w        = input_shape[1], input_shape[2]

    # Step 2: Resize image to model's expected size
    img = cv2.resize(img_rgb, (w, h))

    # Step 3: Normalise pixels from [0,255] → [0.0, 1.0]
    img = img / 255.0

    # Step 4: Reshape to [1, h, w, 3]  (model expects batch dimension)
    img = np.reshape(img, [1, h, w, 3])

    # Step 5: Run prediction
    pred       = model.predict(img, verbose=0)
    label      = int(np.argmax(pred))         # 0 or 1
    confidence = float(pred[0][label] * 100)  # confidence of chosen class
    belt_conf  = float(pred[0][0] * 100)      # confidence for class 0 (belt)

    # Step 6: Safety threshold — only call SAFE if confident enough
    if belt_conf < SAFE_THRESHOLD:
        label = 1   # treat as NO SEATBELT

    ms = (time.perf_counter() - t0) * 1000
    return label, confidence, belt_conf, ms


# ─────────────────────────────────────────────────────────────────────────────
#  ANNOTATE IMAGE
#  Draws coloured banner on top of image showing result
# ─────────────────────────────────────────────────────────────────────────────

def annotate_image(img_rgb, label, confidence):
    """
    Draws a green or red banner at the top of the image.
    Automatically scales text size based on image width.
    Returns annotated RGB image.
    """
    img_bgr   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w      = img_bgr.shape[:2]
    annotated = img_bgr.copy()

    # Choose colour and text based on prediction
    color = (0, 180, 0) if label == 0 else (0, 0, 220)
    text  = f"SEATBELT ON  {confidence:.0f}%" if label == 0 \
            else f"NO SEATBELT  {confidence:.0f}%"

    # Scale font size based on image width (so it fits any image)
    font_scale = max(0.5, min(w / 500, 1.6))
    thickness  = max(1, int(font_scale * 2))

    # Measure how tall the text is
    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness
    )

    # Draw filled banner rectangle
    banner_h = th + 24
    cv2.rectangle(annotated, (0, 0), (w, banner_h), color, -1)

    # Draw white text on banner
    cv2.putText(annotated, text, (12, banner_h - 10),
                cv2.FONT_HERSHEY_DUPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIDENCE CHART
#  Horizontal bar chart showing belt vs no-belt confidence
# ─────────────────────────────────────────────────────────────────────────────

def confidence_chart(belt_conf):
    """Creates a matplotlib confidence bar chart."""
    no_belt_conf = 100 - belt_conf

    fig, ax = plt.subplots(figsize=(5, 2))
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')

    bars = ax.barh(
        ['No Seatbelt', 'Seatbelt'],
        [no_belt_conf, belt_conf],
        color=['#ef4444', '#22c55e'],
        height=0.5
    )

    # Add percentage labels on bars
    for bar, val in zip(bars, [no_belt_conf, belt_conf]):
        ax.text(min(val + 1, 96), bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', color='white',
                fontsize=11, fontweight='bold')

    # Orange threshold line
    ax.axvline(SAFE_THRESHOLD, color='orange', linestyle='--',
               linewidth=1.5, label=f'Safe threshold ({SAFE_THRESHOLD}%)')

    ax.set_xlim(0, 100)
    ax.set_xlabel('Confidence %', color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#94a3b8')
    ax.spines[:].set_visible(False)
    ax.legend(fontsize=8, facecolor='#1e293b', labelcolor='white')
    plt.tight_layout(pad=0.5)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  SHOW RESULT  —  reusable function used by ALL input modes
#  Shows: result card + side-by-side images + metrics + chart + download button
# ─────────────────────────────────────────────────────────────────────────────

def show_result(model, img_rgb, source_label="Input"):
    """
    Complete result display pipeline.
    Called the same way whether image came from upload or camera.

    Parameters:
      model       : loaded Keras model
      img_rgb     : RGB NumPy array
      source_label: string shown as image caption ("Uploaded" or "Camera")
    """

    # ── Run prediction ────────────────────────────────────────────────────────
    with st.spinner("Analysing image..."):
        label, confidence, belt_conf, ms = predict(model, img_rgb)

    # ── Result card ───────────────────────────────────────────────────────────
    if label == 0:
        st.markdown(f"""
        <div class="card-safe">
            <p class="card-title">✅ SEATBELT DETECTED</p>
            <p class="card-conf">
                Confidence: {confidence:.1f}% &nbsp;|&nbsp; Inference: {ms:.0f} ms
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="card-unsafe">
            <p class="card-title">❌ NO SEATBELT DETECTED</p>
            <p class="card-conf">
                Confidence: {confidence:.1f}% &nbsp;|&nbsp; Inference: {ms:.0f} ms
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Side-by-side: original + annotated ───────────────────────────────────
    annotated  = annotate_image(img_rgb, label, confidence)
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown(f"**{source_label}**")
        st.image(img_rgb, use_container_width=True)

    with col2:
        st.markdown("**Annotated**")
        st.image(annotated, use_container_width=True)

        # Download annotated image button
        buf = io.BytesIO()
        Image.fromarray(annotated).save(buf, format="PNG")
        st.download_button(
            "⬇️ Download annotated image",
            buf.getvalue(),
            file_name="seatbelt_result.png",
            mime="image/png",
            use_container_width=True
        )

    # ── Metrics row ───────────────────────────────────────────────────────────
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Result",     "✅ SAFE"   if label == 0 else "❌ UNSAFE")
    m2.metric("Confidence", f"{confidence:.1f}%")
    m3.metric("Belt Conf",  f"{belt_conf:.1f}%")
    m4.metric("Inference",  f"{ms:.0f} ms")

    # ── Confidence chart ──────────────────────────────────────────────────────
    st.markdown("")
    st.markdown("**Confidence Breakdown**")
    fig = confidence_chart(belt_conf)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    # ── Save to session history ───────────────────────────────────────────────
    st.session_state.history.append({
        "source":     source_label,
        "result":     "Seatbelt" if label == 0 else "No Seatbelt",
        "label":      label,
        "confidence": confidence,
        "belt_conf":  belt_conf,
        "ms":         ms,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE  —  stores prediction history across reruns
# ─────────────────────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    # Model info
    st.markdown("### 📦 Model")
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        st.success(f"✅ Model loaded")
        st.caption(f"File: `{MODEL_PATH}`")
        st.caption(f"Size: `{size_mb:.1f} MB`")
    else:
        st.error("❌ seatbelt_model.h5 not found")
        st.caption("Run `python seatbelt_cnn.py` to train first.")

    st.divider()
    st.markdown("🟢 **Green** = Belt ON (Safe)")
    st.markdown("🔴 **Red** = Belt OFF (Unsafe)")
    st.divider()

    # Clear history button
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.markdown(
        "<p style='color:#475569;font-size:0.75rem;text-align:center;"
        "margin-top:2rem'>Seat Belt Detection System<br>"
        "CNN · TensorFlow · Streamlit</p>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("🚗 Seat Belt Detection System")
st.markdown(
    "<p style='color:#64748b'>Detect whether vehicle occupants are wearing "
    "seat belts — upload an image, use your camera, or test multiple images.</p>",
    unsafe_allow_html=True
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL CHECK  —  stop page if model not found
# ─────────────────────────────────────────────────────────────────────────────

model = load_model()

if model is None:
    st.error("""
    ❌ **`seatbelt_model.h5` not found.**

    Train the model first by running in terminal:
    ```
    python seatbelt_cnn.py
    ```
    Then refresh this page.
    """)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
#  TABS  —  3 modes
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️  Upload Image",      # existing feature
    "📷  Live Camera",        # NEW feature
    "📦  Multiple Images",    # existing feature
    "📊  History"             # existing feature
])


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 1 — UPLOAD IMAGE  (existing feature — unchanged)
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("#### Upload an image from your computer")
    st.caption("Supports: JPG, JPEG, PNG, BMP, WebP")

    # File uploader widget
    uploaded = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="upload_single",
        label_visibility="collapsed"
    )

    if uploaded:
        # Decode uploaded file bytes → NumPy RGB array
        file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("❌ Could not read image. Please try a different file.")
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            # Run full prediction + display pipeline
            show_result(model, img_rgb, source_label="Original")


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 2 — LIVE CAMERA CAPTURE  (NEW FEATURE)
#
#  HOW IT WORKS:
#  st.camera_input() is Streamlit's built-in webcam component.
#  It renders a live video feed in the browser.
#  When user clicks "Take Photo", it captures one frame and returns
#  the image as bytes (exactly like a file upload).
#  We decode those bytes using PIL → NumPy, then run through the
#  exact same predict() and show_result() functions as uploaded images.
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("#### Capture a photo using your webcam")
    st.caption(
        "Click **Take Photo** to capture → detection runs automatically. "
        "Click **Clear photo** to retake."
    )

    # ── Instructions box ──────────────────────────────────────────────────────
    st.info(
        "📋 **How to use:**\n"
        "1. Allow camera access when your browser asks\n"
        "2. Position yourself/vehicle in frame\n"
        "3. Click **Take Photo** button below the camera\n"
        "4. Detection result appears instantly below"
    )

    # ── Camera input widget ───────────────────────────────────────────────────
    # st.camera_input() opens webcam in browser
    # Returns None if no photo taken yet, or image bytes after capture
    camera_photo = st.camera_input(
        label="Webcam",              # internal label (hidden)
        key="camera_capture",
        label_visibility="collapsed" # hide the label text
    )

    # ── Process captured photo ────────────────────────────────────────────────
    if camera_photo is not None:
        # camera_photo is a BytesIO-like object containing PNG image bytes
        # Step 1: Open with PIL (handles PNG from camera)
        try:
            pil_image = Image.open(camera_photo)

            # Step 2: Convert PIL → RGB NumPy array
            # This is identical to how uploaded images are handled
            img_rgb = np.array(pil_image.convert("RGB"))

            st.success("✅ Photo captured! Running detection...")
            st.markdown("")

            # Step 3: Run full prediction + display
            # Exactly same pipeline as uploaded images
            show_result(model, img_rgb, source_label="Camera Capture")

        except Exception as e:
            # Handle any camera/conversion errors gracefully
            st.error(f"❌ Could not process camera image: {e}")
            st.caption("Try taking the photo again or use the Upload tab instead.")

    else:
        # Show placeholder when no photo taken yet
        st.markdown("")
        st.markdown(
            "<div style='text-align:center; color:#475569; padding:20px'>"
            "📷 Camera preview will appear above.<br>"
            "Click <b>Take Photo</b> to capture and detect.</div>",
            unsafe_allow_html=True
        )

    # ── Camera not available notice ───────────────────────────────────────────
    with st.expander("📌 Camera not working? Read this"):
        st.markdown("""
        **Common reasons camera may not work:**

        | Problem | Fix |
        |---|---|
        | Browser blocked camera | Click the 🔒 icon in address bar → Allow camera |
        | No webcam on your PC | Use the **Upload Image** tab instead |
        | Camera used by another app | Close Zoom/Teams/Skype and retry |
        | HTTP connection | Camera only works on `localhost` or HTTPS |

        **Alternative:** Use the **🖼️ Upload Image** tab to upload a photo
        taken from your phone or downloaded from Google.
        """)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 3 — MULTIPLE IMAGES  (existing feature — unchanged)
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("#### Upload multiple images at once")
    st.caption("All images will be tested and a summary shown at the end.")

    uploaded_files = st.file_uploader(
        "Choose image files",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="upload_multi",
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} image(s) uploaded**")
        st.divider()

        safe_count   = 0
        unsafe_count = 0

        # Process each image one by one
        for i, uf in enumerate(uploaded_files):
            file_bytes = np.frombuffer(uf.read(), dtype=np.uint8)
            img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img_bgr is None:
                st.warning(f"⚠️ Could not read: {uf.name}")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Run prediction
            label, confidence, belt_conf, ms = predict(model, img_rgb)
            annotated = annotate_image(img_rgb, label, confidence)

            if label == 0:
                safe_count += 1
            else:
                unsafe_count += 1

            # Display image + result side by side
            col_img, col_res = st.columns([1, 1], gap="medium")
            with col_img:
                st.image(annotated, caption=uf.name, use_container_width=True)
            with col_res:
                st.markdown(f"**{uf.name}**")
                if label == 0:
                    st.success(f"✅ SEATBELT DETECTED — {confidence:.1f}%")
                else:
                    st.error(f"❌ NO SEATBELT — {confidence:.1f}%")
                st.caption(f"Inference: {ms:.0f} ms")

                # Mini confidence chart per image
                fig = confidence_chart(belt_conf)
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)

            st.divider()

            # Save to history
            st.session_state.history.append({
                "source":     uf.name,
                "result":     "Seatbelt" if label == 0 else "No Seatbelt",
                "label":      label,
                "confidence": confidence,
                "belt_conf":  belt_conf,
                "ms":         ms,
            })

        # ── Batch summary ─────────────────────────────────────────────────────
        st.markdown("### 📊 Batch Summary")
        b1, b2, b3 = st.columns(3)
        b1.metric("Total Images",  len(uploaded_files))
        b2.metric("✅ Safe",        safe_count)
        b3.metric("❌ Unsafe",      unsafe_count)

        if unsafe_count > 0:
            st.error(f"🚨 {unsafe_count} image(s) detected without seatbelt!")
        else:
            st.success("✅ All images show seatbelts detected.")


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 4 — HISTORY  (existing feature — unchanged)
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("#### All predictions this session")

    if not st.session_state.history:
        st.info("No predictions yet. Use the other tabs to test images.")
    else:
        history = st.session_state.history
        total   = len(history)
        safe    = sum(1 for h in history if h["label"] == 0)
        unsafe  = total - safe
        avg_ms  = sum(h["ms"] for h in history) / total

        # Summary metrics
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Total Tested",  total)
        h2.metric("✅ Safe",        safe)
        h3.metric("❌ Unsafe",      unsafe)
        h4.metric("Avg Inference", f"{avg_ms:.0f} ms")

        st.divider()

        # History list — newest first
        for item in reversed(history):
            icon      = "✅" if item["label"] == 0 else "❌"
            color     = "#166534" if item["label"] == 0 else "#7f1d1d"
            txt_color = "#bbf7d0" if item["label"] == 0 else "#fecaca"
            st.markdown(
                f"<div style='background:{color};border-radius:8px;"
                f"padding:10px 16px;margin-bottom:8px;display:flex;"
                f"justify-content:space-between'>"
                f"<span style='color:{txt_color}'>{icon} "
                f"<b>{item.get('source','Image')}</b></span>"
                f"<span style='color:#94a3b8;font-size:0.85rem'>"
                f"{item['result']} | {item['confidence']:.1f}% | "
                f"{item['ms']:.0f} ms</span></div>",
                unsafe_allow_html=True
            )

        # Export to CSV
        st.markdown("")
        df  = pd.DataFrame(history)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export history as CSV",
            csv,
            file_name="seatbelt_history.csv",
            mime="text/csv",
            use_container_width=True,
            key="camera_csv_export"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<p style='text-align:center;color:#475569;font-size:0.78rem'>"
    "Seat Belt Detection System &nbsp;·&nbsp; CNN + TensorFlow + Streamlit "
    "&nbsp;·&nbsp; Final Year University Project</p>",
    unsafe_allow_html=True
)
