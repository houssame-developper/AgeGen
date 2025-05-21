import streamlit as st
import pandas as pd
from PIL import Image
import torch
from joblib import load
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForImageClassification
from streamlit.components.v1 import html

# Titre de l'application
st.title("Système d'analyse d'image et estimation de l'âge et du sexe")

# Chargement des données CSV
def load_data():
    try:
        data = pd.read_csv('recom.csv', encoding='utf-8')
        return data
    except FileNotFoundError:
        st.warning("Fichier de données introuvable")
        return pd.DataFrame()

data = load_data()

# Chargement des modèles de prédiction
@st.cache_resource
def load_models():
    try:
        processor = AutoImageProcessor.from_pretrained("dima806/fairface_age_image_detection")
        age_model = AutoModelForImageClassification.from_pretrained("dima806/fairface_age_image_detection")
        gender_model = AutoModelForImageClassification.from_pretrained("dima806/fairface_gender_image_detection")
        return processor, age_model, gender_model
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles : {e}")
        return None, None, None

processor, age_model, gender_model = load_models()

# Fonction pour prédire l'âge et le sexe
def predict_image(image):
    try:
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            age_logits = age_model(**inputs).logits
            gender_logits = gender_model(**inputs).logits
            predicted_age_idx = age_logits.argmax(-1).item()
            predicted_gender_idx = gender_logits.argmax(-1).item()
            predicted_age = age_model.config.id2label[predicted_age_idx]
            predicted_gender = gender_model.config.id2label[predicted_gender_idx]
            return predicted_age, predicted_gender
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        return None, None

def cards(recommendations):
    # CSS personnalisé pour les cartes
    css = """
    <style>
    body {
     margin: 0;
    padding: 0;
    }
    [data-testid="stAppViewContainer"], section {
        background-color: #3559A0;
    }
    .flex-container {
        display: flex;
        flex-wrap: wrap;  /* changé de nowrap à wrap */
        gap: 15px;
        justify-content: flex-start;
        direction: rtl;
        padding: 20px;
        background-color: #22305C;
        overflow-x: auto;
        overflow-y: hidden;
        scrollbar-color: #fff #000;
        scrollbar-width: thin;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    iframe {
        border: 1px solid #3559A0;
        border-radius: 20px;
        background-color: #22305C;
    }
    .card {
        background: #f9f9f9;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        padding: 16px;
        width: 220px;
        text-align: right;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin-bottom: 15px;
        flex: 0 0 auto;
    }
    .card img {
        width: 100%;
        height: 160px;
        object-fit: cover;
        border-radius: 8px;
        margin-bottom: 12px;
        border: 1px solid #eee;
    }
    .card h4 {
        margin: 8px 0;
        color: #333;
        font-size: 16px;
    }
    .card p {
        margin: 4px 0;
        color: #555;
        font-size: 14px;
    }
    </style>
    """
    
    cards_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        {css}
    </head>
    <body>
        <div class="flex-container">
    """
    
    for _, row in recommendations.iterrows():
        image_url = row.get('image', '') if pd.notna(row.get('image', '')) else "https://via.placeholder.com/220x160?text=No+Image"
        cards_html += f"""
        <div class="card">
            <img src="{image_url}" alt="{row.get('name', '')}" onerror="this.src='https://via.placeholder.com/220x160?text=Image+Error'"/>
            <h4>{row.get('name', '')}</h4>
            <p>{row.get('type', '')} | {row.get('genre', '')}</p>
            <p>Âge : {row.get('age_group', '')} | Sexe : {row.get('gender', '')}</p>
        </div>
        """
    
    cards_html += """
        </div>
    </body>
    </html>
    """
    
    html(cards_html, height=320, scrolling=False)

# Fonction pour afficher les recommandations basées sur l'âge et le sexe
def show_recommendations(age, gender, data):
    if data.empty:
        st.warning("Aucune donnée de recommandation disponible")
        return
    
    try:
        recommendations = data.loc[
            (data['gender'] == gender) & 
            (data['age_group'] == age)
        ]
        
        st.subheader("Recommandations suggérées :")
        
        if not recommendations.empty:
            cards(recommendations)
        else:
            st.warning("Aucune recommandation trouvée")
    except Exception as e:
        st.error(f"Erreur lors de l'affichage des recommandations : {e}")
        st.dataframe(data.sample(5))

# Barre latérale pour le choix de la méthode d'entrée d'image
option = st.sidebar.selectbox(
    "Choisissez la méthode d'entrée d'image",
    ("Télécharger depuis le fichier", "Prendre une photo avec la caméra")
)

uploaded_image = None
captured_image = None
image_to_predict = None

if option == "Télécharger depuis le fichier":
    uploaded_file = st.file_uploader("Choisissez une image à télécharger", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        with st.spinner('Traitement de l\'image en cours...'):
            uploaded_image = Image.open(uploaded_file)
            image_to_predict = uploaded_image
            st.success("Image téléchargée avec succès !")
            st.image(uploaded_image, caption="Image téléchargée", use_container_width=True)
else:
    st.write("Appuyez sur le bouton pour activer la caméra")
    picture = st.camera_input("Prendre une photo")
    if picture:
        with st.spinner('Traitement de l\'image en cours...'):
            captured_image = Image.open(picture)
            image_to_predict = captured_image
            st.success("Photo prise avec succès !")

if st.button("Analyser l'image"):
    if image_to_predict is not None and processor is not None and age_model is not None and gender_model is not None:
        with st.spinner('Analyse de l\'image en cours...'):
            predicted_age, predicted_gender = predict_image(image_to_predict)
            if predicted_age is not None and predicted_gender is not None:
                st.subheader("Résultats de l'analyse :")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Âge estimé", predicted_age)
                with col2:
                    st.metric("Sexe estimé", predicted_gender)
                show_recommendations(predicted_age.lower().strip(), predicted_gender.lower().strip(), data)
            else:
                st.error("Échec de l'analyse de l'image")
    else:
        if image_to_predict is None:
            st.warning("Veuillez télécharger ou prendre une photo d'abord")
        else:
            st.error("Les modèles ne sont pas prêts pour l'analyse")

# Section pour ajouter un nouveau fichier CSV si nécessaire
st.sidebar.header("Gestion des données")
new_csv = st.sidebar.file_uploader("Télécharger un nouveau fichier de données (CSV)", type=['csv'])
if new_csv is not None:
    try:
        new_data = pd.read_csv(new_csv, encoding='utf-8')
        new_data.to_csv('data.csv', index=False)
        st.sidebar.success("Fichier CSV mis à jour avec succès !")
        data = load_data()
    except Exception as e:
        st.sidebar.error(f"Erreur survenue : {e}")
