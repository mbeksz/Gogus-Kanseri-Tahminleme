import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import numpy as np

def get_clean_data():
    data = pd.read_csv('data/data.csv')
    
    data = data.drop(columns=['Unnamed: 32', 'id'])
    
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def add_sidebar():
    st.sidebar.header("Girilen Veriler")
    data = get_clean_data()
    
    slider_labels = [
        ("radius_mean", "Ortalama Yarıçap"),
        ("texture_mean", "Ortalama Doku"),
        ("perimeter_mean", "Ortalama Çevre"),
        ("area_mean", "Ortalama Alan"),
        ("smoothness_mean", "Ortalama Pürüzsüzlük"),
        ("compactness_mean", "Ortalama Sıkılık"),
        ("concavity_mean", "Ortalama İçbükeylik"),
        ("concave points_mean", "Ortalama İçbükey Noktalar"),
        ("symmetry_mean", "Ortalama Simetri"),
        ("fractal_dimension_mean", "Ortalama Fraktal Boyut"),
        ("radius_se", "Yarıçap Standart Hata"),
        ("texture_se", "Doku Standart Hata"),
        ("perimeter_se", "Çevre Standart Hata"),
        ("area_se", "Alan Standart Hata"),
        ("smoothness_se", "Pürüzsüzlük Standart Hata"),
        ("compactness_se", "Sıkılık Standart Hata"),
        ("concavity_se", "İçbükeylik Standart Hata"),
        ("concave points_se", "İçbükey Noktalar Standart Hata"),
        ("symmetry_se", "Simetri Standart Hata"),
        ("fractal_dimension_se", "Fraktal Boyut Standart Hata"),
        ("radius_worst", "En Büyük Yarıçap"),
        ("texture_worst", "En Büyük Doku"),
        ("perimeter_worst", "En Büyük Çevre"),
        ("area_worst", "En Büyük Alan"),
        ("smoothness_worst", "En Büyük Pürüzsüzlük"),
        ("compactness_worst", "En Büyük Sıkılık"),
        ("concavity_worst", "En Büyük İçbükeylik"),
        ("concave points_worst", "En Büyük İçbükey Noktalar"),
        ("symmetry_worst", "En Büyük Simetri"),
        ("fractal_dimension_worst", "En Büyük Fraktal Boyut")
    ]

    input_dict = {}
    
    for key, label in slider_labels:   
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(data[key].min()),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    
    return input_dict

def add_scaled_values(input_dict):
    data = get_clean_data()
    
    X = data.drop(columns=['diagnosis'], axis=1)
    scaled_dict = {}
    
    for key, value in input_dict.items():
        max_value = X[key].max()
        min_value = X[key].min()
        scaled_value = (value - min_value) / (max_value - min_value)
        scaled_dict[key] = scaled_value
        
    return scaled_dict

def get_radar_chart(input_data):
    input_data = add_scaled_values(input_data)
    
    categories = ['Yarıçap', 'Doku', 'Çevre', 'Alan', 
              'Pürüzsüzlük', 'Sıkılık', 
              'İçbükeylik', 'İçbükey Noktalar', 
              'Simetri', 'Fraktal Boyut']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Girilen Değerler'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standart Değerler'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='En kötü Değerler'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    return fig

def add_prediction(input_data):
    model = pickle.load(open('model/model.pkl', 'rb'))
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    input_df = pd.DataFrame(input_array, columns=list(input_data.keys()))

    
    input_array_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_array_scaled)
    
    st.subheader("Hücre Kümesi Tahmini")
    st.write("Tahmin Edilen Sonuç:")
    
    if prediction == 1:
        st.write("<span class= 'diagnosis malicious'>Kötü Huylu</span>", unsafe_allow_html=True)
        
    else:
        st.write("<span class= 'diagnosis benign'>İyi Huylu</span>", unsafe_allow_html=True)
    st.write("İyi huylu kanser olma olasılığı: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Kötü huylu kanser olma olasılığı: ", model.predict_proba(input_array_scaled)[0][1])
    st.write("Bu uygulama, tıbbi profesyonellere tanı koymada yardımcı olabilir, ancak profesyonel bir tanı yerine geçmemelidir.")



def main():
    st.set_page_config(page_title="Göğüs Kanseri Tahmin Uygulaması", page_icon=":guardsman:", layout="wide", initial_sidebar_state="expanded")
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    input_data = add_sidebar()

    with st.container():
        st.title("Göğüs Kanseri Tahmin Uygulaması")
        st.write("Lütfen bu uygulamayı sitoloji laboratuvarınıza bağlayarak doku örneklerinizden meme kanseri teşhisi koymanıza yardımcı olacak şekilde kullanın. Bu uygulama, laboratuvarınızdan alınan ölçümleri işleyerek, bir meme kütlesinin iyi huylu mu yoksa kötü huylu mu olduğunu makine öğrenimi modeliyle tahmin eder. Ayrıca, kenar çubuğundaki kaydırıcıları kullanarak ölçümleri manuel olarak da güncelleyebilirsiniz.")

    col1, col2 = st.columns([4, 1])
    
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart, use_container_width=True)
    
    with col2:
        add_prediction(input_data)
     
if __name__ == "__main__":
    main()
