import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("Détecteur de Tumeur")
st.write("Comparez les modèles avec et sans augmentations")

model_choice = st.selectbox("Choisir un modèle :", ["Sans Augmentation", "Avec Augmentation"])
if model_choice == "Sans Augmentation":
    model = YOLO("../Models/best.pt")
else:
    model = YOLO("../Models/best.pt")

uploaded_file = st.file_uploader("Uploader une image ou une vidéo", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    if uploaded_file.type.startswith("image"):
        img = Image.open(uploaded_file)
        st.image(img, caption="Image originale")
        results = model(img)
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Résultat YOLO", use_container_width=True)
    else:
        st.video(uploaded_file)
        results = model.predict(source=uploaded_file.name, show=False, save=True)
        st.success("✅ Détection terminée ! Résultat sauvegardé dans runs/detect/predict")

