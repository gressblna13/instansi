import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Langchain and Chroma integration
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import chroma
from langchain_community.embeddings import fastembed
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata


# Set page config as the first Streamlit command
st.set_page_config(page_title="Dashboard Hasil Evaluasi 2025", page_icon="✅", layout="wide")

# Styling untuk memperbaiki tampilan chat (chat berada di sebelah kanan)
st.markdown(""" 
    <style>
        .title {
            color: #003366;
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            padding: 20px;
        }

        .chat-container {
            max-height: 400px;
            overflow-y: auto;
            padding: 10px;
            margin-bottom: 20px;
            background-color: #f5f5f5;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            width: 300px;
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 999;
        }

        .chat-bubble {
            padding: 10px;
            border-radius: 15px;
            margin: 5px;
            max-width: 75%;
            font-size: 16px;
        }

        .user-bubble {
            background-color: #d1e7ff;
            align-self: flex-start;
        }

        .ai-bubble {
            background-color: #f1f1f1;
            align-self: flex-end;
        }

        .stButton button {
            background-color: #0061f2;
            color: white;
            border-radius: 10px;
            font-size: 16px;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .stButton button:hover {
            background-color: #0051c2;
        }

        .stSelectbox select {
            font-size: 16px;
            padding: 10px;
        }

        .stDataFrame table {
            width: 100%;
            border-collapse: collapse;
        }

        .stDataFrame th, .stDataFrame td {
            padding: 8px;
            text-align: left;
            border: 1px solid #ddd;
        }
    </style>
""", unsafe_allow_html=True)


# Menampilkan header yang lebih menarik
st.markdown('<div class="title">Dashboard Hasil Evaluasi 2025</div>', unsafe_allow_html=True)

# Menyambungkan dengan chromadb menggunakan client
client = chromadb.Client()

# Menggunakan Sentence Transformers untuk membuat embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Fungsi untuk menyimpan data ke dalam Chroma
def store_data_in_chroma(text_data):
    documents = list(text_data.values())  # Data teks untuk "Pemenuhan Aspek", "Kekuatan", "Kelemahan"
    
    # Membuat embeddings untuk setiap teks
    embeddings = model.encode(documents)
    
    # Cek apakah koleksi sudah ada
    if "tdp_data" not in client.list_collections():
        # Membuat koleksi Chroma untuk menyimpan data teks
        collection = client.create_collection("tdp_data")
    else:
        collection = client.get_collection("tdp_data")
    
    # Menambahkan data ke koleksi
    collection.add(
        documents=documents,
        metadatas=[{"source": key} for key in text_data.keys()],
        ids=[key for key in text_data.keys()],
        embeddings=embeddings  # Menyertakan embeddings
    )
    return collection

# Fungsi untuk chat dengan AI menggunakan Chromadb
def chat_with_ai(user_input, collection, general_data):
    # Cek apakah input user cocok dengan data umum (seperti salam)
    for entry in general_data:
        if entry['question'].lower() in user_input.lower():
            return entry['answer']
    
    # Jika bukan pertanyaan umum, lakukan pencarian pada Chroma
    query_vector = model.encode([user_input])[0]  # Membuat embedding vektor untuk query pengguna
    query_result = collection.query(query_embeddings=query_vector, n_results=1)
    
    if query_result['documents']:
        response = f"Jawaban berdasarkan data: {query_result['documents'][0]}"
    else:
        response = "Maaf, saya tidak dapat menemukan jawaban yang relevan."
    
    return response

# Membaca file JSON yang berisi pertanyaan umum
with open("data/questions.json", "r") as f:
    general_data = json.load(f)

# Set up untuk aplikasi Streamlit
st.title("Dashboard Hasil Evaluasi 2025")

# Data yang akan dimasukkan ke dalam database
text_data = {
    "Pemenuhan Aspek": "Menguraikan pemenuhan aspek dengan meninjau seluruh indikator yang telah memenuhi Tingkat Kematangan 2 – 5 dari masing-masing aspek penerapan SPBE",
    "Kekuatan": "Menggambarkan kondisi kekuatan indikator yang memenuhi Tingkat Kematangan 4-5 secara berjenjang dari nilai tertinggi; Mendeskripsikan fakta/ alasan/data dukung penerapan indikator tersebut.",
    "Kelemahan": "Menggambarkan kondisi kelemahan indikator yang memenuhi Tingkat Kematangan 1-2 secara berjenjang dari nilai terendah; Mendeskripsikan fakta/alasan tidak memenuhi/ menerapkan indikator tersebut."
}

# Store the text data into Chroma
vector_store = store_data_in_chroma(text_data)

# Upload File Excel
uploaded_file = st.file_uploader("Pilih file Excel untuk di-upload", type=["xlsx", "xls"])

def get_predikat(index):
    if index >= 4.2:
        return "Memuaskan"
    elif index >= 3.5:
        return "Sangat Baik"
    elif index >= 2.6:
        return "Baik"
    elif index >= 1.8:
        return "Cukup"
    else:
        return "Kurang"

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    instansi_list = df['nama_instansi'].unique().tolist()

    selected_instansi = st.selectbox("Pilih Instansi", instansi_list)

    instansi_data = df[df['nama_instansi'] == selected_instansi].drop_duplicates(subset=['nama_instansi'])
    st.subheader(f"Data untuk {selected_instansi}")
    st.dataframe(instansi_data)

    instansi_data['predikat'] = instansi_data['indeks'].apply(get_predikat)

    st.subheader("Nilai Indeks dan Predikat")
    st.write(instansi_data[['code', 'nama_instansi', 'indeks', 'predikat']])

    st.markdown("## Grafik Domain dan Aspek")
    
    col1, col2 = st.columns(2)

    with col1:
        try:
            domain_columns = ['d1', 'd2', 'd3', 'd4']
            domain_labels = ['Kebijakan Internal Tata Kelola SPBE', 'Perencanaan Strategis SPBE', 'Teknologi Informasi dan Komunikasi', 'Penyelenggaraan SPBE']
            domain_data = instansi_data[domain_columns].values[0]
            target_data = [2.6, 2.6, 2.6, 2.6]

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=target_data,
                theta=domain_labels,
                fill='toself',
                name='Target',
                fillcolor='rgba(0, 124, 212, 0.32)',
                line=dict(color='rgb(0, 124, 212)', width=2),
                opacity=0.5
            ))

            fig.add_trace(go.Scatterpolar(
                r=domain_data,
                theta=domain_labels,
                fill='toself',
                name='Capaian',
                fillcolor='rgba(255, 99, 132, 0.32)',
                line=dict(color='rgb(255, 99, 132)', width=2),
                opacity=0.5
            ))

            fig.update_layout(
                title="Evaluasi Domain",
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 5]),
                ),
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)
        except KeyError as e:
            st.error(f"Error: Kolom {e} tidak ditemukan dalam data.")

    with col2:
        try:
            aspek_columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']
            aspek_labels = ['Kebijakan Internal Tata Kelola SPBE', 'Perencanaan Strategis SPBE', 'Teknologi Informasi dan Komunikasi', 'Penyelenggaraan SPBE', 'Penerapan Manajemen SPBE', 'Audit TIK', 'Layanan Administrasi Pemerintahan Berbasis Elektronik', 'Layanan Publik Berbasis Elektronik']
            aspek_data = instansi_data[aspek_columns].values[0]
            target_data_aspek = [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6]

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=target_data_aspek,
                theta=aspek_labels,
                fill='toself',
                name='Target',
                fillcolor='rgba(0, 124, 212, 0.32)',
                line=dict(color='rgb(0, 124, 212)', width=2),
                opacity=0.5
            ))

            fig.add_trace(go.Scatterpolar(
                r=aspek_data,
                theta=aspek_labels,
                fill='toself',
                name='Capaian',
                fillcolor='rgba(255, 99, 132, 0.32)',
                line=dict(color='rgb(255, 99, 132)', width=2),
                opacity=0.5
            ))

            fig.update_layout(
                title="Evaluasi Aspek",
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 5]),
                ),
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)
        except KeyError as e:
            st.error(f"Error: Kolom {e} tidak ditemukan dalam data.")

    # Predikat SPBE - Di Samping Grafik
    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(""" 
                <div style="padding: 20px; background-color:rgb(185, 206, 249); border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                    <h3 style="color: #003366; font-weight: bold;">Predikat SPBE</h3>
                    <p style="font-size: 18px; color: #333;">Indeks SPBE: <strong>{}</strong></p>
                    <p style="font-size: 18px; color: #333;">Predikat: <strong>{}</strong></p>
                    <p style="font-size: 16px; color: #555;">Predikat ini menunjukkan bahwa instansi ini berada pada kategori <strong>{}</strong> untuk nilai SPBE.</p>
                </div>
            """.format(instansi_data['indeks'].values[0], get_predikat(instansi_data['indeks'].values[0]), get_predikat(instansi_data['indeks'].values[0])), unsafe_allow_html=True)

        with col2:
            st.markdown("## Fitur Chat AI")

            # Menggunakan session state untuk menyimpan status chat
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            # Menggunakan form untuk input
            with st.form(key='chat_form'):
                user_input = st.text_input("Tanyakan sesuatu kepada AI:")
                submit_button = st.form_submit_button(label='Kirim')

            if submit_button and user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                ai_response = chat_with_ai(user_input, vector_store, general_data)
                st.session_state.chat_history.append({"role": "ai", "content": ai_response})

            # Menampilkan riwayat chat
            for chat in st.session_state.chat_history:
                if chat["role"] == "user":
                    st.write(f"**You:** {chat['content']}")
                else:
                    st.write(f"**AI:** {chat['content']}") 
