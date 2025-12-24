import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model  
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os


# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Batik Classification Dashboard",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 24px;
        background-color: #f0f2f6;
        border-radius: 8px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# FUNGSI HELPER - LOAD MODELS
# ==========================================

@st.cache_resource
def load_cnn_model():
    """Load model CNN dan metadata"""
    try:
        # PERBAIKI PATH: Gunakan forward slash atau os.path.join
        model = load_model(os.path.join('models', 'model_batik_cnn (1).h5'))
        with open(os.path.join('models', 'cnn_batik (1).pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        # DEBUG: Check metadata keys
        st.sidebar.info(f"CNN Metadata keys: {list(metadata.keys())[:5]}...")
        
        return model, metadata
    except Exception as e:
        st.error(f"Error loading CNN model: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None


@st.cache_resource
def load_mobilenet_model():
    """Load model MobileNetV2 dan metadata"""
    try:
        # PERBAIKI PATH
        model = load_model(os.path.join('models', 'model_batik_mobilenetv2_fixed (1).h5'))
        with open(os.path.join('models', 'model_batik_mobilenetv2_fixed (1).pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        # DEBUG: Check metadata keys
        st.sidebar.info(f"MobileNet Metadata keys: {list(metadata.keys())[:5]}...")
        
        return model, metadata
    except Exception as e:
        st.error(f"Error loading MobileNetV2 model: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None


@st.cache_resource
def load_resnet_model():
    """Load model ResNet50 dan metadata dengan multiple fallback methods"""
    try:
        # PERBAIKI PATH
        model_path = os.path.join('models', 'model_batik_resnet50_final (1).h5')
        pkl_path = os.path.join('models', 'model_dashboard_data (2).pkl')
        
        # Check if files exist
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            return None, None
        
        if not os.path.exists(pkl_path):
            st.error(f"❌ Metadata file not found: {pkl_path}")
            return None, None
        
        # Load metadata first
        with open(pkl_path, 'rb') as f:
            metadata = pickle.load(f)
        
        st.sidebar.success("✅ Metadata loaded")
        
        # Method 1: Try direct load
        try:
            st.sidebar.info("📥 Method 1: Direct loading...")
            model = tf.keras.models.load_model(model_path, compile=False)
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            st.sidebar.success("✅ Model loaded directly")
            return model, metadata
            
        except Exception as e:
            st.sidebar.warning(f"⚠️ Direct loading failed: {str(e)[:100]}")
        
        # Method 2: Rebuild architecture
        try:
            st.sidebar.info("🔄 Method 2: Rebuilding architecture...")
            
            inputs = tf.keras.layers.Input(shape=(224, 224, 3), name='input_layer')
            base_model = tf.keras.applications.ResNet50(
                weights=None, 
                include_top=False, 
                input_tensor=inputs
            )
            
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
            x = tf.keras.layers.Dense(256, activation='relu', 
                                       kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                       name='dense_256')(x)
            x = tf.keras.layers.BatchNormalization(name='bn_256')(x)
            x = tf.keras.layers.Dropout(0.5, name='dropout_256')(x)
            x = tf.keras.layers.Dense(128, activation='relu',
                                       kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                       name='dense_128')(x)
            x = tf.keras.layers.BatchNormalization(name='bn_128')(x)
            x = tf.keras.layers.Dropout(0.4, name='dropout_128')(x)
            outputs = tf.keras.layers.Dense(5, activation='softmax', name='output_layer')(x)
            
            new_model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='ResNet50_Batik')
            
            st.sidebar.info("📦 Loading weights...")
            old_model = tf.keras.models.load_model(model_path, compile=False)
            new_model.set_weights(old_model.get_weights())
            del old_model
            
            new_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            st.sidebar.success("✅ Model rebuilt successfully")
            return new_model, metadata
            
        except Exception as e:
            st.error(f"❌ Rebuild method failed: {e}")
            import traceback
            with st.expander("🔍 Show error details"):
                st.code(traceback.format_exc())
            return None, None
            
    except Exception as e:
        st.error(f"❌ Failed to load ResNet50: {e}")
        return None, None


# ==========================================
# FUNGSI HELPER - PREDICTION & PLOTTING
# ==========================================

def preprocess_image(image, img_size):
    """Preprocess gambar untuk prediksi"""
    image = image.resize(img_size)
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(model, image, img_size, class_names):
    """Melakukan prediksi pada gambar"""
    processed_img = preprocess_image(image, img_size)
    predictions = model.predict(processed_img, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx] * 100)
    predicted_class = class_names[predicted_class_idx]
    all_probs = {class_names[i]: float(predictions[0][i] * 100) 
                 for i in range(len(class_names))}
    return predicted_class, confidence, all_probs


def plot_training_history(history, title="Training History"):
    """Plot training dan validation accuracy"""
    fig = go.Figure()
    epochs = list(range(1, len(history['accuracy']) + 1))
    
    fig.add_trace(go.Scatter(
        x=epochs, y=history['accuracy'],
        mode='lines+markers', name='Training Accuracy',
        line=dict(color='#2ca02c', width=3), marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs, y=history['val_accuracy'],
        mode='lines+markers', name='Validation Accuracy',
        line=dict(color='#d62728', width=3), marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=title, xaxis_title="Epoch", yaxis_title="Accuracy",
        template="plotly_white", hovermode='x unified', height=400
    )
    return fig


def plot_loss_history(history, title="Loss History"):
    """Plot training dan validation loss"""
    fig = go.Figure()
    epochs = list(range(1, len(history['loss']) + 1))
    
    fig.add_trace(go.Scatter(
        x=epochs, y=history['loss'],
        mode='lines+markers', name='Training Loss',
        line=dict(color='#1f77b4', width=3), marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs, y=history['val_loss'],
        mode='lines+markers', name='Validation Loss',
        line=dict(color='#ff7f0e', width=3), marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=title, xaxis_title="Epoch", yaxis_title="Loss",
        template="plotly_white", hovermode='x unified', height=400
    )
    return fig


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """Plot confusion matrix dengan heatmap"""
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    text = [[f"{count}<br>({pct:.1f}%)" for count, pct in zip(row_counts, row_pcts)]
            for row_counts, row_pcts in zip(cm, cm_percent)]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=class_names, y=class_names,
        text=text, texttemplate="%{text}", textfont={"size": 12},
        colorscale='Blues', showscale=True
    ))
    
    fig.update_layout(
        title=title, xaxis_title="Predicted Label", yaxis_title="True Label",
        template="plotly_white", height=500, width=600
    )
    return fig


def plot_per_class_accuracy(cm, class_names, title="Per-Class Accuracy"):
    """Plot akurasi per kelas"""
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig = go.Figure(data=[
        go.Bar(
            x=class_names, y=per_class_acc, marker_color=colors,
            text=[f"{acc:.2f}%" for acc in per_class_acc],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=title, xaxis_title="Batik Class", yaxis_title="Accuracy (%)",
        template="plotly_white", height=400, yaxis=dict(range=[0, 110])
    )
    return fig


def plot_resnet_phases(metadata):
    """Plot khusus untuk ResNet50 dengan 3 fase training"""
    history = metadata['history_combined']
    epochs = list(range(1, len(history['accuracy']) + 1))
    phase1_end = metadata['epoch_phase1_end']
    phase2_end = metadata['epoch_phase2_end']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=epochs, y=history['accuracy'],
        mode='lines', name='Training Accuracy',
        line=dict(color='#2ca02c', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs, y=history['val_accuracy'],
        mode='lines', name='Validation Accuracy',
        line=dict(color='#d62728', width=3)
    ))
    
    if phase1_end > 0:
        fig.add_vline(x=phase1_end, line_dash="dash", line_color="red", 
                      annotation_text=f"Phase 2 Start (Epoch {phase1_end})",
                      annotation_position="top")
    
    if phase2_end > phase1_end:
        fig.add_vline(x=phase2_end, line_dash="dash", line_color="green",
                      annotation_text=f"Phase 3 Start (Epoch {phase2_end})",
                      annotation_position="top")
    
    fig.update_layout(
        title="Training & Validation Accuracy (3 Phases)",
        xaxis_title="Epoch", yaxis_title="Accuracy",
        template="plotly_white", hovermode='x unified', height=500
    )
    return fig


# ==========================================
# MAIN APP
# ==========================================

def main():
    st.markdown('<p class="main-header">🎨 Batik Classification Dashboard</p>', 
                unsafe_allow_html=True)
    
    # Sidebar - Model Selection
    st.sidebar.title("⚙️ Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose Model:",
        ["CNN (Custom)", "MobileNetV2 (Transfer Learning)", "ResNet50 (Transfer Learning)"]
    )
    
    # Load model berdasarkan pilihan
    model, metadata = None, None
    
    if model_choice == "CNN (Custom)":
        model, metadata = load_cnn_model()
        model_type = "CNN"
        img_size = (128, 128)
    elif model_choice == "MobileNetV2 (Transfer Learning)":
        model, metadata = load_mobilenet_model()
        model_type = "MobileNetV2"
        img_size = (224, 224)
    else:
        model, metadata = load_resnet_model()
        model_type = "ResNet50"
        img_size = (224, 224)
    
    if model is None or metadata is None:
        st.error("⚠️ Failed to load model. Please check if model files exist.")
        return
    
    # Get class names
    class_names = metadata.get('class_names', [])
    
    st.sidebar.success(f"✅ {model_type} loaded successfully!")
    st.sidebar.info(f"📊 Image Size: {img_size[0]}x{img_size[1]}")
    st.sidebar.info(f"🎯 Classes: {len(class_names)}")
    
    # ==========================================
    # TABS
    # ==========================================
    tab1, tab2 = st.tabs(["📊 Model Evaluation", "🔍 Image Prediction"])
    
    # ==========================================
    # TAB 1: MODEL EVALUATION
    # ==========================================
    with tab1:
        st.markdown('<p class="sub-header">Model Performance Overview</p>', 
                    unsafe_allow_html=True)
        
        # Metrics row
        if model_type == "ResNet50":
            final_acc = metadata.get('final_accuracy', 0) * 100
            final_loss = metadata.get('final_loss', 0)
            best_val_acc = metadata.get('best_val_accuracy', 0) * 100
            total_epochs = metadata.get('total_epochs', 0)
        else:
            history = metadata.get('history', {})
            final_acc = history.get('val_accuracy', [0])[-1] * 100 if history.get('val_accuracy') else 0
            final_loss = history.get('val_loss', [0])[-1] if history.get('val_loss') else 0
            best_val_acc = max(history.get('val_accuracy', [0])) * 100 if history.get('val_accuracy') else 0
            total_epochs = len(history.get('accuracy', []))
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Final Accuracy", f"{final_acc:.2f}%")
        with col2:
            st.metric("📉 Final Loss", f"{final_loss:.4f}")
        with col3:
            st.metric("⭐ Best Val Accuracy", f"{best_val_acc:.2f}%")
        with col4:
            st.metric("🔄 Total Epochs", total_epochs)
        
        st.markdown("---")
        
        # Training History Plots
        st.markdown('<p class="sub-header">Training History</p>', 
                    unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if model_type == "ResNet50":
                fig_acc = plot_resnet_phases(metadata)
                st.plotly_chart(fig_acc, use_container_width=True)
            else:
                history = metadata.get('history', {})
                if history:
                    fig_acc = plot_training_history(history, f"{model_type} - Accuracy")
                    st.plotly_chart(fig_acc, use_container_width=True)
        
        with col2:
            if model_type == "ResNet50":
                history = metadata['history_combined']
            else:
                history = metadata.get('history', {})
            
            if history:
                fig_loss = plot_loss_history(history, f"{model_type} - Loss")
                st.plotly_chart(fig_loss, use_container_width=True)
        
        st.markdown("---")
        
        # Confusion Matrix dan Per-Class Accuracy
        st.markdown('<p class="sub-header">Classification Metrics</p>', 
                    unsafe_allow_html=True)
        
        # CHECK: Apakah confusion matrix ada di metadata?
        cm = None
        if 'confusion_matrix' in metadata:
            cm = np.array(metadata['confusion_matrix'])
            st.success("✅ Confusion matrix found in metadata!")
        else:
            st.warning(f"⚠️ Confusion matrix not found in metadata. Available keys: {list(metadata.keys())}")
            st.info("💡 Please re-train the model with the updated training code that saves confusion matrix.")
        
        if cm is not None and len(cm) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_cm = plot_confusion_matrix(cm, class_names, f"{model_type} - Confusion Matrix")
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col2:
                fig_per_class = plot_per_class_accuracy(cm, class_names, f"{model_type} - Per-Class Accuracy")
                st.plotly_chart(fig_per_class, use_container_width=True)
        
        # Classification Report (hanya untuk ResNet50 yang punya data lengkap)
        if model_type == "ResNet50" or 'precision' in metadata:
            st.markdown('<p class="sub-header">Detailed Classification Report</p>', 
                        unsafe_allow_html=True)
            
            precision = metadata.get('precision', [])
            recall = metadata.get('recall', [])
            f1_score = metadata.get('f1_score', [])
            support = metadata.get('support', [])
            per_class_acc = metadata.get('per_class_accuracy', [])
            
            if precision and recall and f1_score:
                df_report = pd.DataFrame({
                    'Class': class_names,
                    'Precision': [f"{p:.4f}" for p in precision],
                    'Recall': [f"{r:.4f}" for r in recall],
                    'F1-Score': [f"{f:.4f}" for f in f1_score],
                    'Support': support,
                    'Accuracy': [f"{a:.2f}%" for a in per_class_acc]
                })
                
                st.dataframe(df_report, use_container_width=True, hide_index=True)
                
                # Macro Average
                macro_avg = metadata.get('macro_avg', {})
                if macro_avg:
                    st.markdown(f"""
                    **Macro Average:**
                    - Precision: {macro_avg.get('precision', 0):.4f}
                    - Recall: {macro_avg.get('recall', 0):.4f}
                    - F1-Score: {macro_avg.get('f1_score', 0):.4f}
                    - Accuracy: {macro_avg.get('accuracy', 0):.2f}%
                    """)
    
    # ==========================================
    # TAB 2: IMAGE PREDICTION
    # ==========================================
    with tab2:
        st.markdown('<p class="sub-header">Upload Image for Prediction</p>', 
                    unsafe_allow_html=True)
        
        st.info(f"📌 Current Model: **{model_type}** | Image Size: **{img_size[0]}x{img_size[1]}**")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a batik image to classify"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 📸 Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", use_container_width=True)
                
                st.markdown(f"""
                **Image Information:**
                - Format: {image.format}
                - Size: {image.size[0]} x {image.size[1]} pixels
                - Mode: {image.mode}
                """)
            
            with col2:
                st.markdown("#### 🎯 Prediction Results")
                
                if st.button("🚀 Classify Image", type="primary", use_container_width=True):
                    with st.spinner("Processing..."):
                        try:
                            predicted_class, confidence, all_probs = predict_image(
                                model, image, img_size, class_names
                            )
                            
                            st.success(f"### ✅ Predicted Class: **{predicted_class}**")
                            st.metric("Confidence", f"{confidence:.2f}%")
                            
                            progress_value = min(max(float(confidence) / 100.0, 0), 1)
                            st.progress(progress_value)
                            st.markdown("---")
                            
                            st.markdown("#### 📊 All Class Probabilities")
                            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                            df_probs = pd.DataFrame(sorted_probs, columns=['Class', 'Probability (%)'])
                            
                            fig_probs = go.Figure(data=[
                                go.Bar(
                                    x=df_probs['Probability (%)'],
                                    y=df_probs['Class'],
                                    orientation='h',
                                    marker=dict(
                                        color=df_probs['Probability (%)'],
                                        colorscale='Blues',
                                        showscale=False
                                    ),
                                    text=[f"{p:.2f}%" for p in df_probs['Probability (%)']],
                                    textposition='outside'
                                )
                            ])
                            
                            fig_probs.update_layout(
                                title="Class Probability Distribution",
                                xaxis_title="Probability (%)",
                                yaxis_title="Class",
                                template="plotly_white",
                                height=400,
                                xaxis=dict(range=[0, 105])
                            )
                            
                            st.plotly_chart(fig_probs, use_container_width=True)
                            st.dataframe(df_probs, use_container_width=True, hide_index=True)
                            
                        except Exception as e:
                            st.error(f"❌ Error during prediction: {e}")
                            import traceback
                            st.code(traceback.format_exc())
        
        else:
            st.info("👆 Please upload an image to start prediction")
            st.markdown("---")
            st.markdown("#### 📚 Supported Batik Classes")
            
            cols = st.columns(5)
            for idx, class_name in enumerate(class_names):
                with cols[idx % 5]:
                    st.markdown(f"**{idx + 1}. {class_name}**")


if __name__ == "__main__":
    main()
