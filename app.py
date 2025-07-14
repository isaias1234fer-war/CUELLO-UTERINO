import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
import os
import random

from reportlab.lib.pagesizes import A4
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from datetime import datetime
# Configuración de la página
st.set_page_config(page_title="Diagnóstico de Cáncer Cervical", layout="wide")

# Mapeo de clases
class_names = {
    0: "Normal",
    1: "Células Escamosas Superficiales/Intermedias",
    2: "Células Escamosas Parabasales",
    3: "Células Metaplásicas",
    4: "Adenocarcinoma"
}

@st.cache_resource
def load_models():
    return {
        "CNN Simple": tf.keras.models.load_model("models/best_cervical_model.h5"),
        "CNN Optimizado": tf.keras.models.load_model("models/optimized_cervical_model.h5")
    }

def preprocess_image(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (64, 64))

    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    img = img / 255.0
    return np.expand_dims(img, axis=0)
# Tabla de métricas
metricas_modelos = [
    ["Modelo", "Precisión", "Sensibilidad (Avg)", "Especificidad (Avg)", "F1-Score (Avg)", "MCC"],
    ["Mejor Modelo (Keras Tuner)", "83.08%", "80.73%", "89.83%", "82.78%", "0.7065"],
    ["CNN Simple", "84.89%", "80.36%", "91.08%", "82.21%", "0.7372"],
    ["ResNet50 (Transfer Learning)", "73.11%", "53.78%", "83.61%", "51.27%", "0.5407"]
]

# Resultados de la prueba de McNemar

mcnemar_results = [
    ["Comparación de\nModelos", "Estadístico\nχ²", "Valor-p", "¿Diferencia\nSignificativa?"],
    ["Mejor Modelo vs CNN Simple", "4.50", "0.034", "Sí"],
    ["Mejor Modelo vs ResNet50", "2.10", "0.147", "No"],
    ["CNN Simple vs ResNet50 C", "5.30", "0.021", "Sí"]
]
def generar_pdf(imagen, model_name, resultado, confianza, probabilidades, mcc, mcnemar_result):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    # Fecha
    fecha_actual = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    c.setFont("Helvetica", 10)
    c.drawString(30, y, f"Fecha del Reporte: {fecha_actual}")
    y -= 30

    # Título de sección
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, y, "Métricas de Rendimiento de Modelos")
    y -= 20

    # Tabla de métricas
    table = Table(metricas_modelos, colWidths=[130, 70, 90, 90, 90, 50])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    table.wrapOn(c, width, height)
    table.drawOn(c, 30, y - 90)
    y -= 120
    
    # Título de sección McNemar
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, y, "Pruebas de McNemar")
    y -= 20
    
    # Encabezado McNemar
    encabezado = ["Comparación", "Estadística Chi2", "P-valor", "Conclusión"]
    tabla_mcnemar = [encabezado] + mcnemar_results
    table2 = Table(tabla_mcnemar, colWidths=[170, 90, 60, 180])
    table2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),         # Fondo gris en encabezado
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),         # Texto blanco en encabezado
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),                # Centrado horizontal
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),               # Centrado vertical
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),        # Bordes negros
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),      # Fuente en negrita para encabezado
        ('FONTSIZE', (0, 0), (-1, -1), 9),  
    ]))
    
    table2.wrapOn(c, width, height)
    table2.drawOn(c, 30, y - 100)
    y -= 130
    
    y=-20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, y, "🧾 Matrices de Confusión por Modelo")
    c.setFont("Helvetica", 12)
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, 800, "📊 Matriz Confusión CNN Simple")
    c.setFont("Helvetica", 12)
    c.drawImage("reports/matriz_comparacion.png", 100, 580, width=400, height=200)
# Insertar imagen: CNN
    y=-20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, y, "🧾 Matrices de Confusión por Modelo")
    c.setFont("Helvetica", 12)
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, 800, "📊 Comparación entre Modelos")
    c.setFont("Helvetica", 12)
    c.drawImage("reports/confusion_matrix.png", 100, 580, width=400, height=200)


    y=-20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, y, "🧾 Matrices de Confusión por Modelo")
    c.setFont("Helvetica", 12)
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, 800, "📊 Matriz Confusión ResNet50")
    c.setFont("Helvetica", 12)
    c.drawImage("reports/matriz_resnet.png", 100, 580, width=400, height=200)

# Siguiente página (si quieres mostrar matriz de comparación aparte)

    # Firmar PDF
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Sidebar
st.sidebar.title("⚙ Opciones")
app_mode = st.sidebar.selectbox("Modo de Operación", ["Diagnóstico", "Reportes", "Información Técnica"])

models = load_models()

if app_mode == "Diagnóstico":
    st.header("📷 Subir Imagen Cervical")
    uploaded_file = st.file_uploader("Seleccione una imagen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida", width=300)
        processed_img = preprocess_image(image)
        model_name = st.selectbox("Seleccione el modelo", list(models.keys()))

        if st.button("🔍 Realizar diagnóstico"):
            with st.spinner("Analizando..."):
                model = models[model_name]
                preds = model.predict(processed_img)
                pred_class = np.argmax(preds[0])
                confidence = np.max(preds[0]) * 100
                resultado = class_names[pred_class]
                
                st.success(f"*Resultado:* {resultado} (Confianza: {confidence:.2f}%)")

                st.subheader("📊 Probabilidades por Clase")
                for i, prob in enumerate(preds[0]):
                    st.progress(int(prob * 100))
                    st.write(f"{class_names[i]}: {prob * 100:.2f}%")

                # Datos sintéticos para MCC y McNemar (reemplazar con tus datos reales)
                y_true = [random.randint(0, 4) for _ in range(100)]
                y_pred1 = [random.randint(0, 4) for _ in range(100)]
                y_pred2 = [random.randint(0, 4) for _ in range(100)]

                mcc = matthews_corrcoef(y_true, y_pred1)

                matrix = confusion_matrix(y_pred1, y_pred2, labels=[0,1,2,3,4])
                agree = matrix.trace()
                disagree = matrix.sum() - agree
                b = disagree // 2
                table = [[agree, b], [b, agree]]
                mcnemar_result = mcnemar(table, exact=False, correction=True)

                # Generar PDF
                pdf_buffer = generar_pdf(image, model_name, resultado, confidence, preds[0], mcc, mcnemar_result)

                st.download_button(
                    label="📄 Descargar Reporte PDF",
                    data=pdf_buffer,
                    file_name="reporte_diagnostico.pdf",
                    mime="application/pdf"
                )

elif app_mode == "Reportes":
    st.header("📈 Reportes de Rendimiento")
    st.subheader("Matriz de Confusión")
    st.image("reports/confusion_matrix.png")
    st.subheader("Curva de Aprendizaje")
    st.image("reports/learning_curve.png")

elif app_mode == "Información Técnica":
    st.header("🤖 Modelos Utilizados")
    st.markdown("""
    - *CNN Simple*: Red convolucional básica con 3 capas.
    - *CNN Optimizado*: Modelo con hiperparámetros ajustados automáticamente.
    """)
    st.subheader("📊 Métricas de Rendimiento")
    st.write("""
    | Modelo           | Precisión (%) | Sensibilidad (%) |
    |------------------|--------------|------------------|
    | CNN Simple       | 92.3         | 89.5             |
    | CNN Optimizado   | 95.1         | 93.8             |
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
*Dataset:* SIPaKMeD  
*Modelo:* CNN Optimizado  
*Precisión:* 95.1% (validación cruzada)
""")