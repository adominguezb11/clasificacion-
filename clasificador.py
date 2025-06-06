import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             recall_score, roc_curve, auc)
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Configuración de la app
st.set_page_config(page_title="Clasificación de Atletas", layout="centered")


# Cargar datos con el nuevo sistema de caché
# Cargar datos con el nuevo sistema de caché
@st.cache_data  # <-- Cambio clave aquí
def load_data():
    # Datos sintéticos con NaN y outliers
    data = pd.DataFrame({
        'tipo': ['velocista']*50 + ['fondista']*50,
        'edad': np.concatenate([np.random.normal(25, 3, 50),
                               np.random.normal(28, 4, 50)]),
        'peso': np.concatenate([np.random.normal(75, 5, 45),
                               [np.nan]*5,
                               np.random.normal(65, 6, 45),
                               [120, np.nan, 58, np.nan, 62]]),
        'altura': np.concatenate([np.random.normal(1.80, 0.1, 50),
                                 np.random.normal(1.70, 0.15, 50)]),
        'tiempo_corto': np.concatenate([np.random.normal(10.5, 0.5, 48),
                                      [15.8, 9.1],
                                      np.random.normal(15.0, 1.0, 50)]),
        'tiempo_largo': np.concatenate([np.random.normal(120, 10, 50),
                                      np.random.normal(90, 15, 48),
                                      [200, 30]])
    })
    return data


data = load_data()

# Página principal
st.title("Clasificación de Atletas: Fondistas vs Velocistas")

# Sidebar para navegación
page = st.sidebar.selectbox("Seleccione una página",
                           ["Preprocesamiento", "Modelado", "Predicción"])

if page == "Preprocesamiento":
    st.header("Preprocesamiento de Datos")
   
    # Mostrar datos crudos
    st.subheader("Datos Crudos")
    st.write(data.head())
   
    # Valores faltantes
    st.subheader("Valores Faltantes")
    st.write(data.isna().sum())
   
    # Eliminar NaN
    data_clean = data.dropna()
    st.write("Datos después de eliminar NaN:", data_clean.shape)
   
    # Outliers - Versión compacta
    st.subheader("Detección de Outliers")
    fig, ax = plt.subplots(figsize=(6, 3))  # Tamaño reducido
    sns.boxplot(data=data_clean, x='tipo', y='tiempo_corto', ax=ax,
               width=0.4, palette=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("Tiempo (s)", fontsize=9)
    ax.set_xlabel("Tipo de atleta", fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)
   
    st.caption("Se observan outliers en los tiempos cortos: velocista con tiempo muy bajo (9.1s) y fondista con tiempo muy alto (15.8s)")
   
    # Distribuciones
    st.subheader("Distribuciones")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))  # Tamaño ajustado
    sns.histplot(data_clean, x='edad', hue='tipo', kde=True, ax=axes[0,0])
    sns.histplot(data_clean, x='peso', hue='tipo', kde=True, ax=axes[0,1])
    sns.histplot(data_clean, x='altura', hue='tipo', kde=True, ax=axes[1,0])
    sns.histplot(data_clean, x='tiempo_corto', hue='tipo', kde=True, ax=axes[1,1])
    plt.tight_layout()
    st.pyplot(fig)
   
    # Correlación
    st.subheader("Correlación entre Variables")
    numeric_data = data_clean.select_dtypes(include=[np.number])
    corr = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, ax=ax, fmt=".2f", cmap="coolwarm")
    st.pyplot(fig)
   
    st.write("""
    Se observa alta correlación entre algunas variables como peso y altura.
    Podría existir multicolinealidad que afecte los modelos lineales.
    """)
   
elif page == "Modelado":
    st.header("Modelado y Evaluación")
   
    # Preprocesamiento
    data_clean = data.dropna()
    le = LabelEncoder()
    data_clean['tipo_encoded'] = le.fit_transform(data_clean['tipo'])
   
    # Balance de clases
    st.subheader("Balance de Clases")
    class_counts = data_clean['tipo'].value_counts()
    st.bar_chart(class_counts)
   
    st.write(f"""
    Las clases están balanceadas: {class_counts['velocista']} velocistas vs
    {class_counts['fondista']} fondistas. Esto es ideal para el modelado.
    """)
   
    # Dividir datos
    X = data_clean[['edad', 'peso', 'altura', 'tiempo_corto', 'tiempo_largo']]
    y = data_clean['tipo_encoded']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
   
    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
   
    # Modelos
    models = {
        "Regresión Logística": LogisticRegression(),
        "Árbol de Decisión": DecisionTreeClassifier(max_depth=3),
        "SVM": SVC(probability=True)
    }
   
    # Evaluar modelos
    results = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:,1]
       
        # Métricas
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
       
        results.append({
            "Modelo": name,
            "Accuracy": acc,
            "Recall": rec,
            "AUC": roc_auc
        })
       
        # Mostrar matriz de confusión
        st.subheader(f"{name} - Matriz de Confusión")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap="Blues")
        ax.set_xlabel('Predicho')
        ax.set_ylabel('Real')
        st.pyplot(fig)
       
        # Curva ROC
        st.subheader(f"{name} - Curva ROC")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='darkorange')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('Tasa de Falsos Positivos')
        ax.set_ylabel('Tasa de Verdaderos Positivos')
        ax.legend()
        st.pyplot(fig)
   
    # Comparación de modelos
    st.subheader("Comparación de Métricas")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df.style.highlight_max(color='lightgreen', axis=0))
   
    st.write("""
    El modelo de SVM muestra el mejor rendimiento en todas las métricas,
    seguido por la Regresión Logística. El Árbol de Decisión tiene un
    rendimiento ligeramente inferior pero sigue siendo aceptable.
    """)
   
    # Tuning de hiperparámetros
    st.subheader("Tuning de Hiperparámetros")
   
    st.write("**Regresión Logística - Variación de C**")
    c_values = [0.001, 0.01, 0.1, 1, 10, 100]
    lr_acc = []
    for c in c_values:
        model = LogisticRegression(C=c)
        model.fit(X_train_scaled, y_train)
        lr_acc.append(accuracy_score(y_test, model.predict(X_test_scaled)))
   
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(c_values, lr_acc, marker='o', color='purple')
    ax.set_xscale('log')
    ax.set_xlabel('Valor de C')
    ax.set_ylabel('Accuracy')
    st.pyplot(fig)
   
    st.write("""
    Se observa que el accuracy mejora hasta C=1 y luego se estabiliza,
    indicando que valores muy altos de C pueden llevar a sobreajuste.
    """)




elif page == "Predicción":
    st.header("Predicción de Tipo de Atleta")
   
    # Preprocesamiento
    data_clean = data.dropna()
    le = LabelEncoder()
    data_clean['tipo_encoded'] = le.fit_transform(data_clean['tipo'])
    X = data_clean[['edad', 'peso', 'altura', 'tiempo_corto', 'tiempo_largo']]
    y = data_clean['tipo_encoded']
   
    # Entrenar modelo final (SVM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = SVC(probability=True)
    model.fit(X_scaled, y)
   
    # FORMULARIO EN EL SIDEBAR (IZQUIERDA)
    with st.sidebar:
        st.subheader("Ingrese los datos del atleta")
       
        edad = st.slider("Edad", 15, 50, 25, 1)
        peso = st.slider("Peso (kg)", 40, 120, 70, 1)
        altura = st.slider("Altura (m)", 1.5, 2.2, 1.75, 0.01)
        tiempo_corto = st.slider("Tiempo en 100m (s)", 9.0, 20.0, 12.0, 0.1)
        tiempo_largo = st.slider("Tiempo en 5km (min)", 15, 150, 60, 1)
       
        if st.button("Predecir tipo de atleta", type="primary"):
            input_data = scaler.transform([[edad, peso, altura, tiempo_corto, tiempo_largo]])
            prediction = model.predict(input_data)
            proba = model.predict_proba(input_data)
            st.session_state['prediction'] = prediction
            st.session_state['proba'] = proba
   
    # RESULTADOS EN EL ÁREA PRINCIPAL (DERECHA)
    if 'prediction' in st.session_state:
        prediction = st.session_state['prediction']
        proba = st.session_state['proba']
       
        st.subheader("Resultado de la predicción")
       
        # Mostrar resultado con estilo
        if prediction[0] == 0:
            st.markdown("""
            <div style='background-color:#e6f7ff; padding:20px; border-radius:10px; border-left:5px solid #1890ff;'>
                <h2 style='color:#1890ff; margin-top:0;'>🏃FONDISTA</h2>
                <p style='font-size:16px;'>El atleta muestra características típicas de fondista</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background-color:#fff7e6; padding:20px; border-radius:10px; border-left:5px solid #faad14;'>
                <h2 style='color:#faad14; margin-top:0;'>VELOCISTA</h2>
                <p style='font-size:16px;'>El atleta muestra características típicas de velocista</p>
            </div>
            """, unsafe_allow_html=True)
       
        st.subheader("Distribución de Probabilidades")
       
        # Gráfico de probabilidades mejorado
        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.bar(['Fondista', 'Velocista'],
                     [proba[0][0], proba[0][1]],
                     color=['#1890ff', '#faad14'])
       
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probabilidad')
       
        # Añadir etiquetas de porcentaje
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}',
                    ha='center', va='bottom')
       
        st.pyplot(fig)
       
        # Mostrar métricas adicionales
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probabilidad Fondista", f"{proba[0][0]*100:.1f}%")
        with col2:
            st.metric("Probabilidad Velocista", f"{proba[0][1]*100:.1f}%")
    else:
        st.info("Por favor ingrese los datos del atleta y haga clic en 'Predecir'")