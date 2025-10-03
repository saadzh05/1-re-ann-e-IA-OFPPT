import streamlit as st
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import time
import tempfile
import zipfile
import io
import base64
from pathlib import Path

# Configuration de la page Streamlit
st.set_page_config(
    page_title="🚶‍♂️ Détecteur de Piétons IA",
    page_icon="🚶‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PedestrianDetectorStreamlit:
    """Version adaptée du détecteur pour Streamlit"""
    
    def __init__(self, model_path='best_model.pth', confidence_threshold=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        """Charge le modèle avec mise en cache pour Streamlit"""
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        num_classes = 2
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        model.to(self.device)
        model.eval()
        return model
    
    def _preprocess_image(self, image_array):
        """Prétraitement adapté pour les images uploadées via Streamlit"""
        # L'image vient déjà en RGB depuis PIL/Streamlit
        height, width = image_array.shape[:2]
        max_size = 1024
        
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_array = cv2.resize(image_array, (new_width, new_height))
        
        image_tensor = torch.from_numpy(image_array / 255.0).permute(2, 0, 1).float()
        image_tensor = image_tensor.to(self.device).unsqueeze(0)
        
        return image_array, image_tensor
    
    def detect_pedestrians(self, image_array):
        """Détection adaptée pour Streamlit"""
        start_time = time.time()
        
        try:
            image_rgb, image_tensor = self._preprocess_image(image_array)
            
            with torch.no_grad():
                predictions = self.model(image_tensor)[0]
            
            boxes = predictions['boxes'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
            
            confident_detections = scores >= self.confidence_threshold
            final_boxes = boxes[confident_detections]
            final_scores = scores[confident_detections]
            final_labels = labels[confident_detections]
            
            annotated_image = self._draw_detections(
                image_rgb.copy(), final_boxes, final_scores, final_labels
            )
            
            processing_time = time.time() - start_time
            
            detection_stats = {
                'total_detections': len(final_boxes),
                'processing_time': processing_time,
                'average_confidence': np.mean(final_scores) if len(final_scores) > 0 else 0,
                'detections': [
                    {
                        'box': box.tolist(),
                        'confidence': float(score),
                        'label': 'pedestrian'
                    }
                    for box, score in zip(final_boxes, final_scores)
                ]
            }
            
            return annotated_image, detection_stats
            
        except Exception as e:
            st.error(f"Erreur pendant la détection: {e}")
            return None, {'error': str(e), 'total_detections': 0}
    
    def _draw_detections(self, image, boxes, scores, labels):
        """Dessine les détections sur l'image"""
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.astype(int)
            
            if score > 0.8:
                color = (0, 255, 0)
            elif score > 0.6:
                color = (255, 255, 0)
            else:
                color = (255, 165, 0)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            text = f'Piéton {score:.2f}'
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            cv2.rectangle(image, (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0], y1), color, -1)
            
            cv2.putText(image, text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image

@st.cache_resource
def load_detector(model_path, confidence):
    """Charge le détecteur avec mise en cache"""
    return PedestrianDetectorStreamlit(model_path, confidence)

def create_download_link(img_array, filename):
    """Crée un lien de téléchargement pour l'image"""
    img_pil = Image.fromarray(img_array)
    buf = io.BytesIO()
    img_pil.save(buf, format='PNG')
    buf.seek(0)
    
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">📥 Télécharger l\'image annotée</a>'
    return href

def main():
    # Header principal
    st.title("🚶‍♂️ Détecteur de Piétons par Intelligence Artificielle")
    st.markdown("---")
    
    # Sidebar pour les paramètres
    st.sidebar.header("⚙️ Paramètres")
    
    # Paramètres du modèle
    model_path = st.sidebar.text_input("Chemin du modèle", value="best_model.pth")
    confidence_threshold = st.sidebar.slider(
        "Seuil de confiance", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Plus le seuil est élevé, plus les détections sont sûres mais moins nombreuses"
    )
    
    # Informations système
    st.sidebar.markdown("### 💻 Informations Système")
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.sidebar.info(f"**Device:** {device}")
    
    if not os.path.exists(model_path):
        st.sidebar.warning("⚠️ Modèle personnalisé non trouvé. Utilisation du modèle pré-entraîné.")
    else:
        st.sidebar.success("✅ Modèle personnalisé chargé")
    
    # Interface principale avec onglets
    tab1, tab2, tab3, tab4 = st.tabs(["📸 Image unique", "📁 Lot d'images", "📹 Temps réel", "📊 Statistiques"])
    
    # Onglet 1: Image unique
    with tab1:
        st.header("📸 Détection sur image unique")
        
        uploaded_file = st.file_uploader(
            "Choisissez une image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Formats supportés: JPG, JPEG, PNG, BMP"
        )
        
        if uploaded_file is not None:
            # Affichage de l'image originale
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Image originale")
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                st.image(image, caption="Image uploadée", use_column_width=True)
                
                # Informations sur l'image
                st.info(f"**Dimensions:** {image.size[0]} x {image.size[1]} pixels")
            
            with col2:
                st.subheader("Résultat de la détection")
                
                if st.button("🔍 Détecter les piétons", type="primary"):
                    # Chargement du détecteur
                    detector = load_detector(model_path, confidence_threshold)
                    
                    # Barre de progression
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Initialisation...")
                    progress_bar.progress(25)
                    
                    status_text.text("Analyse de l'image...")
                    progress_bar.progress(50)
                    
                    # Détection
                    annotated_image, stats = detector.detect_pedestrians(image_array)
                    
                    progress_bar.progress(75)
                    status_text.text("Finalisation...")
                    
                    if annotated_image is not None:
                        progress_bar.progress(100)
                        status_text.text("Terminé!")
                        
                        # Affichage du résultat
                        st.image(annotated_image, caption="Détections", use_column_width=True)
                        
                        # Statistiques
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        
                        with col_stat1:
                            st.metric("🎯 Piétons détectés", stats['total_detections'])
                        
                        with col_stat2:
                            st.metric("⏱️ Temps (s)", f"{stats['processing_time']:.2f}")
                        
                        with col_stat3:
                            if stats['total_detections'] > 0:
                                st.metric("📊 Confiance moy.", f"{stats['average_confidence']:.2f}")
                            else:
                                st.metric("📊 Confiance moy.", "N/A")
                        
                        # Détails des détections
                        if stats['total_detections'] > 0:
                            st.subheader("📋 Détails des détections")
                            for i, detection in enumerate(stats['detections'], 1):
                                with st.expander(f"Piéton {i} - Confiance: {detection['confidence']:.2f}"):
                                    box = detection['box']
                                    st.write(f"**Position:** x1={box[0]:.0f}, y1={box[1]:.0f}, x2={box[2]:.0f}, y2={box[3]:.0f}")
                                    st.write(f"**Largeur:** {box[2]-box[0]:.0f} pixels")
                                    st.write(f"**Hauteur:** {box[3]-box[1]:.0f} pixels")
                        
                        # Lien de téléchargement
                        st.markdown("---")
                        download_link = create_download_link(annotated_image, "detection_result.png")
                        st.markdown(download_link, unsafe_allow_html=True)
                        
                        # Nettoyage de la barre de progression
                        progress_bar.empty()
                        status_text.empty()
    
    # Onglet 2: Lot d'images
    with tab2:
        st.header("📁 Traitement par lot")
        st.info("💡 Uploadez plusieurs images pour un traitement automatique en lot")
        
        uploaded_files = st.file_uploader(
            "Choisissez plusieurs images",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            help="Sélectionnez plusieurs fichiers pour le traitement en lot"
        )
        
        if uploaded_files:
            st.write(f"📊 {len(uploaded_files)} images sélectionnées")
            
            if st.button("🚀 Traiter toutes les images", type="primary"):
                detector = load_detector(model_path, confidence_threshold)
                
                # Conteneurs pour les résultats
                results = []
                total_detections = 0
                total_time = 0
                
                # Barre de progression globale
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Traitement de chaque image
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Traitement de {uploaded_file.name}...")
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    image = Image.open(uploaded_file)
                    image_array = np.array(image)
                    
                    annotated_image, stats = detector.detect_pedestrians(image_array)
                    
                    if annotated_image is not None:
                        results.append({
                            'name': uploaded_file.name,
                            'image': annotated_image,
                            'stats': stats
                        })
                        total_detections += stats['total_detections']
                        total_time += stats['processing_time']
                
                # Affichage des résultats
                status_text.text("Traitement terminé!")
                
                # Résumé global
                st.subheader("📊 Résumé du traitement")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📁 Images traitées", len(results))
                with col2:
                    st.metric("🎯 Total détections", total_detections)
                with col3:
                    st.metric("⏱️ Temps total", f"{total_time:.1f}s")
                with col4:
                    st.metric("📈 Moyenne/image", f"{total_detections/len(results):.1f}")
                
                # Affichage des résultats individuels
                st.subheader("🖼️ Résultats détaillés")
                
                for result in results:
                    with st.expander(f"{result['name']} - {result['stats']['total_detections']} piétons"):
                        col_img, col_stats = st.columns([2, 1])
                        
                        with col_img:
                            st.image(result['image'], caption=result['name'], use_column_width=True)
                        
                        with col_stats:
                            st.write(f"**Détections:** {result['stats']['total_detections']}")
                            st.write(f"**Temps:** {result['stats']['processing_time']:.2f}s")
                            if result['stats']['total_detections'] > 0:
                                st.write(f"**Confiance moy.:** {result['stats']['average_confidence']:.2f}")
                
                progress_bar.empty()
                status_text.empty()
    
    # Onglet 3: Temps réel
    with tab3:
        st.header("📹 Détection en temps réel")
        st.info("🚧 Fonctionnalité en développement - Utilisez la webcam pour la détection en temps réel")
        
        # Placeholder pour la webcam
        st.markdown("""
        ### 🎥 Webcam (Bientôt disponible)
        
        Cette fonctionnalité permettra de:
        - 📹 Capturer le flux vidéo de votre webcam
        - 🔄 Analyser les images en temps réel
        - 📊 Afficher les statistiques live
        - 💾 Enregistrer les détections
        
        **Note technique:** Cette fonctionnalité nécessite des composants Streamlit avancés 
        et sera implémentée dans une version future.
        """)
        
        # Simulation avec images de test
        st.subheader("🎬 Simulation avec images de test")
        
        if st.button("▶️ Lancer la simulation"):
            st.info("Simulation d'un flux vidéo avec des images de test...")
            
            # Placeholder pour la simulation
            placeholder = st.empty()
            
            for i in range(5):
                with placeholder.container():
                    st.write(f"Frame {i+1}/5")
                    st.progress((i+1)/5)
                    time.sleep(1)
            
            st.success("Simulation terminée!")
    
    # Onglet 4: Statistiques
    with tab4:
        st.header("📊 Statistiques et monitoring")
        
        # Informations sur le modèle
        st.subheader("🧠 Informations du modèle")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Modèle:** Faster R-CNN ResNet50
            **Device:** {device}
            **Seuil de confiance:** {confidence_threshold}
            **Classes:** Piéton, Arrière-plan
            """)
        
        with col2:
            st.info(f"""
            **Fichier modèle:** {model_path}
            **Existe:** {'✅ Oui' if os.path.exists(model_path) else '❌ Non'}
            **Taille max image:** 1024px
            **Format d'entrée:** RGB
            """)
        
        # Conseils d'optimisation
        st.subheader("💡 Conseils d'optimisation")
        
        st.markdown("""
        ### 🎯 Réglage du seuil de confiance
        - **0.3-0.5:** Plus de détections, plus de faux positifs
        - **0.5-0.7:** Équilibre détections/précision
        - **0.7-0.9:** Moins de détections, plus précises
        
        ### 🖼️ Optimisation des images
        - **Résolution:** 800-1200px pour un bon équilibre vitesse/qualité
        - **Format:** JPG pour des images plus légères
        - **Éclairage:** Images bien éclairées donnent de meilleurs résultats
        
        ### ⚡ Performance
        - **GPU:** Recommandé pour le traitement en lot
        - **Batch size:** Traiter par groupes de 10-20 images
        - **Mémoire:** Surveiller l'usage mémoire pour de gros lots
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🚶‍♂️ **Détecteur de Piétons IA** | "
        "Propulsé par PyTorch et Streamlit | "
        "Modèle: Faster R-CNN ResNet50"
    )

if __name__ == "__main__":
    main()
