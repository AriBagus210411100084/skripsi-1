import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Tampilan Streamlit
st.set_page_config(page_title="Deteksi Dini Penyakit Diabetes", layout="wide")

# Load model yang sudah disimpan
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        color: #4CAF50;
        margin-bottom: 30px;
        padding: 20px;
        background-color: #f0f8f0;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .info-section {
        background-color: #ffffff;
        padding: 25px;
        margin: 20px 0;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .footer {
        text-align: center;
        font-size: 1em;
        color: #777;
        margin-top: 30px;
        padding: 10px;
        background-color: #f8f8f8;
        border-radius: 10px;
    }
    .sidebar-menu {
        display: flex;
        flex-direction: column;
        background-color: #ffffff;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .sidebar-menu a {
        padding: 12px 24px;
        text-decoration: none;
        color: #333;
        border-radius: 8px;
        margin-bottom: 8px;
        background-color: #e1e1e1;
        text-align: center;
        transition: background-color 0.3s ease;
    }
    .sidebar-menu a:hover {
        background-color: #c7c7c7;
        color: #000;
    }
    .stButton>button {
        background-color: {'#00cc66' if is_active else '#f0f2f6'};
        color: {'white' if is_active else 'black'};
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)

# Inisialisasi state menu
if "menu" not in st.session_state:
    st.session_state.menu = "Home"

st.sidebar.markdown("## Main Menu")

# Tombol-tombol menu
if st.sidebar.button("Home"):
    st.session_state.menu = "Home"
if st.sidebar.button("Datasets"):
    st.session_state.menu = "Datasets"
if st.sidebar.button("Pre-Processing"):
    st.session_state.menu = "Pre-Processing"
if st.sidebar.button("Modelling"):
    st.session_state.menu = "Modelling"
if st.sidebar.button("Implementation"):
    st.session_state.menu = "Implementation"

# Ambil menu aktif
menu = st.session_state.menu

# # Navigasi menggunakan elemen HTML untuk menu
# menu_options = ["Home", "Datasets", "Pre-Processing", "Modelling", "Implementation"]
# menu = st.sidebar.radio("Main Menu", menu_options)

if menu == "Home":
    st.markdown("""<div class='title'>Web Klasifikasi Diabetes</div>""", unsafe_allow_html=True)
    st.markdown("""
        <div class='info-section'>
            <h2>Selamat Datang di Aplikasi Deteksi Dini Penyakit Diabetes</h2>
    """, unsafe_allow_html=True)

    # Menampilkan gambar dari folder lokal
    st.image("diabetes1.jpg", caption="Diabetes", use_column_width=True)

    st.markdown("""
            <p style="text-align: justify;">
                Diabetes melitus merupakan penyakit kronis yang menjadi ancaman serius bagi kesehatan global. Penyakit ini ditandai dengan kadar gula darah yang tinggi akibat gangguan produksi atau fungsi insulin dalam tubuh. Faktor risiko diabetes meliputi usia, jenis kelamin, riwayat keluarga, obesitas, pola makan tidak sehat, dan gaya hidup darah sewaktu ≥ 200 mg/dl atau kadar gula darah puasa ≥ 126 mg/dl. Berdasarkan data dari Jurnal yang ada, diperkirakan 537 juta orang dewasa berusia 20 hingga 79 tahun menderita diabetes pada tahun 2021. Angka ini diproyeksikan meningkat menjadi 643 juta pada tahun 2030 dan mencapai 783 juta pada tahun 2045. Dengan meningkatnya jumlah penderita diabetes, terutama di negara berpenghasilan rendah dan menengah, diperlukan langkah–langkah penanganan yang efektif, termasuk penggunaan teknologi untuk membantu diagnosis dan klasifikasi penyakit ini.
            </p>
            <p style="text-align: justify;">
                Pemeriksaan kadar glukosa darah sangat penting untuk mendiagnosis diabetes. Glukosa darah puasa normal adalah 80–110 mg/Dl, sementara kadar glukosa darah puasa yang melebihi 126 mg/Dl, atau kadar glukosa darah 2 jam setelah makan melebihi 200 mg/Dl, menandakan adanya diabetes melitus. Fitur yang akan digunakan pada website deteksi diabetes adalah jenis kelamin, umur, HB (Hemoglobin), HCT (Hematokrit), WBC (White Blood Cells), RBC (Red Blood Cells), PLT (Platelet), dan gula darah.
            </p>
            <h3>Pencegahan Diabetes</h3>
            <p style="text-align: justify;">
                Untuk mencegah diabetes, beberapa langkah yang dapat dilakukan antara lain:
                <ul>
                    <li>Menjaga pola makan sehat dengan mengurangi konsumsi gula dan karbohidrat berlebih.</li>
                    <li>Berolahraga secara teratur untuk menjaga berat badan ideal.</li>
                    <li>Menghindari kebiasaan merokok dan konsumsi alkohol berlebihan.</li>
                    <li>Memeriksa kadar gula darah secara berkala, terutama jika memiliki riwayat keluarga diabetes.</li>
                    <li>Mengelola stres dengan baik, karena stres dapat memengaruhi kadar gula darah.</li>
                </ul>
            </p>
            <h3>Klasifikasi dan Algoritma</h3>
            <p style="text-align: justify;">
                Klasifikasi merupakan salah satu teknik penting dalam data mining yang bertujuan untuk mengelompokkan data atau objek baru ke dalam kelas atau label berdasarkan atribut tertentu. Pada tahap pembuatan model, data yang tersedia digunakan untuk melatih model dalam mengenali pola tertentu. Selanjutnya, model tersebut diterapkan pada data baru untuk melakukan prediksi. Tahap evaluasi dilakukan untuk mengukur keakuratan dan performa model. Algoritma Support Vector Machine (SVM) merupakan salah satu metode pembelajaran mesin yang telah terbukti efektif dalam melakukan klasifikasi dan prediksi dengan akurasi yang tinggi.
            </p>
            <p style="text-align: justify;">
                Berdasarkan penjelasan di atas, penelitian ini bertujuan untuk mengembangkan sistem klasifikasi penyakit diabetes menggunakan algoritma Support Vector Machine (SVM). Dataset yang digunakan diperoleh dari data pasien UPT Puskesmas Bandaran, Kabupaten Pamekasan, Jawa Timur, yang terdiri dari 310 data lalu dilakukan proses balancing menggunakan metode SMOTE menjadi 434 data. Dengan memanfaatkan teknologi dan metode yang telah terbukti efektif, diharapkan penelitian ini dapat memberikan kontribusi dalam upaya diagnosis dan penanganan penyakit diabetes secara lebih akurat dan efisien.
            </p>
        </div>
    """, unsafe_allow_html=True)

elif menu == "Datasets":
    st.markdown("""<div class='title'>Datasets</div>""", unsafe_allow_html=True)
    st.markdown("""
        <div class='info-section'>
            <h2>Dataset yang Digunakan</h2>
            <p>Berikut adalah dataset yang digunakan dalam penelitian ini.</p>
    """, unsafe_allow_html=True)

    # Membaca file CSV
    try:
        df = pd.read_csv("Dataset.csv")
        st.dataframe(df)  # Menampilkan dataframe dalam bentuk tabel
    except FileNotFoundError:
        st.error("File CSV tidak ditemukan. Pastikan file 'nama_file.csv' ada di folder yang sama dengan kode ini.")

    st.markdown("""
            <h3>Penjelasan Fitur Penting</h3>
            <p style="text-align: justify;">
                Berikut adalah penjelasan mengenai fitur-fitur penting yang digunakan dalam dataset:
            </p>
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                <thead>
                    <tr style="background-color: #4CAF50; color: white;">
                        <th style="padding: 10px; border: 1px solid #ddd;">Fitur</th>
                        <th style="padding: 10px; border: 1px solid #ddd;">Deskripsi</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd;">Gender (Jenis Kelamin)</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">Mempertimbangkan perbedaan risiko diabetes antara laki-laki dan perempuan.</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd;">Age (Usia)</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">Usia merupakan faktor penting yang dapat mempengaruhi risiko diabetes, dengan kelompok usia tertentu lebih rentan.</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd;">HB (Hemoglobin)</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">Mengukur kadar hemoglobin dalam darah, yang dapat memberikan indikasi tentang kondisi kesehatan umum.</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd;">HCT (Hematokrit)</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">Persentase volume sel darah merah dalam darah, yang dapat berpengaruh terhadap sirkulasi dan kesehatan metabolisme.</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd;">RBC (Red Blood Cell)</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">Jumlah sel darah merah, yang penting untuk mengangkut oksigen ke seluruh tubuh dan dapat dipengaruhi oleh berbagai kondisi kesehatan.</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd;">WBC (White Blood Cell)</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">Jumlah sel darah putih, yang berfungsi dalam sistem kekebalan tubuh dan dapat memberikan petunjuk tentang adanya infeksi atau peradangan.</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd;">PLT (Platelet)</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">Jumlah keping darah yang berperan dalam pembekuan darah, penting untuk mengevaluasi kesehatan kardiovaskular.</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd;">GDA (Gula Darah)</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">Ukuran kadar glukosa dalam darah, yang merupakan indikator utama dalam mendiagnosis diabetes.</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd;">Label</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">
                            <strong>Diabetes Mellitus</strong>: Menandakan bahwa pasien terdiagnosis diabetes.<br>
                            <strong>Non Diabetes Mellitus</strong>: Menandakan bahwa pasien tidak terdiagnosis diabetes.
                        </td>
                    </tr>
                </tbody>
            </table>
            <p style="text-align: justify;">
                Label ini digunakan sebagai target klasifikasi dalam model machine learning. Nilainya adalah:
                <ul>
                    <li><strong>Diabetes Mellitus</strong>: Pasien terdiagnosis diabetes.</li>
                    <li><strong>Non Diabetes Mellitus</strong>: Pasien tidak terdiagnosis diabetes.</li>
                </ul>
            </p>
        </div>
    """, unsafe_allow_html=True)

elif menu == "Pre-Processing":
    st.markdown("""<div class='title'>Pre-Processing</div>""", unsafe_allow_html=True)
    st.markdown("""
        <div class='info-section'>
            <h2>Proses Pre-Processing Data</h2>
            <p>Proses pre-processing data meliputi pembersihan data, encoding, normalisasi, dan penyeimbangan data menggunakan SMOTE.</p>
        </div>
    """, unsafe_allow_html=True)

    # Menampilkan gambar dari folder lokal
    st.image("preprocessing.png", caption="Diagram Alur Preprocessing", use_column_width=True)

    # Tampilkan dataset sebelum preprocessing
    # Membaca file CSV
    try:
        df = pd.read_csv("Dataset.csv")
    except FileNotFoundError:
        st.error("File CSV tidak ditemukan. Pastikan file 'Dataset.csv' ada di folder yang sama dengan kode ini.")
        st.stop()  # Hentikan eksekusi jika file tidak ditemukan

    st.subheader("Dataset Sebelum Pre-Processing")
    st.write("Berikut adalah dataset sebelum dilakukan preprocessing:")
    st.write(df)  # Menampilkan dataset asli

    # Proses Preprocessing
    st.subheader("Langkah-langkah Preprocessing")

    # 1. Cleaning
    st.markdown("""
        **1. Cleaning Data**  
        - Menghapus kolom yang tidak diperlukan seperti `No` dan `Nama`.  
        - Membersihkan kolom `Umur` dengan menghapus satuan (jika ada).  
    """)
    # Contoh cleaning
    umur = []
    for data in df['Umur']:
        umur.append((data.split()[0]))
    df['Umur'] = umur
    df = df.drop(columns=['No', 'Nama'])
    st.write("Hasil setelah cleaning:")
    st.write(df)

    # 2. Encoding
    st.markdown("""
        **2. Encoding Data**  
        - Kolom `JK` diubah menjadi numerik:  
          - `Laki-laki` = 0  
          - `Perempuan` = 1  
        - Kolom `Label` diubah menjadi numerik:  
          - `Diabetes Mellitus` = -1  
          - `Non-Diabetes Mellitus` = 1  
    """)
    # Contoh encoding
    le = LabelEncoder()
    if 'JK' in df.columns:
        df['JK'] = le.fit_transform(df['JK'])
    if 'Label' in df.columns:
        df['Label'] = le.fit_transform(df['Label'])  # Hasil awalnya 0 dan 1
        df['Label'] = df['Label'] * 2 - 1  # Ubah 0 jadi -1, 1 tetap 1
    st.write("Hasil setelah encoding:")
    st.write(df)

    # 3. Balancing Data dengan SMOTE
    st.markdown("""
        **3. Penyeimbangan Data dengan SMOTE**  
        - SMOTE (Synthetic Minority Over-sampling Technique) digunakan untuk menyeimbangkan jumlah sampel antara kelas `Diabetes Mellitus` dan `Non-Diabetes Mellitus`.  
    """)
    # Contoh SMOTE
    X = df.drop(columns=['Label'])
    y = df['Label']
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    st.write("Sebelum SMOTE:")
    st.write(y.value_counts())
    st.write("Setelah SMOTE:")
    st.write(pd.Series(y_resampled).value_counts())

    # 4. Normalisasi Data
    st.markdown("""
        **4. Normalisasi Data**  
        - Normalisasi dilakukan menggunakan MinMaxScaler untuk mengubah nilai fitur ke dalam rentang 0 hingga 1.  
    """)
    # Contoh normalisasi
    scaler = MinMaxScaler()
    feature_smote = scaler.fit_transform(X_resampled)
    X_normalized_smote = pd.DataFrame(feature_smote, columns=X_resampled.columns)
    st.write("Hasil setelah normalisasi:")
    st.write(X_normalized_smote)

    # Simpan hasil normalisasi ke CSV
    X_normalized_smote.to_csv("X_normalized_smote.csv", index=False)
    st.write("Hasil normalisasi telah disimpan ke file `X_normalized_smote.csv`.")

    # Simpan variabel ke session state
    st.session_state.X_normalized_smote = X_normalized_smote
    st.session_state.y_resampled = y_resampled

    # Tampilkan dataset setelah preprocessing
    st.subheader("Dataset Setelah Pre-Processing")
    st.write("Berikut adalah dataset setelah dilakukan preprocessing:")
    st.write(X_normalized_smote)


elif menu == "Modelling":
    st.markdown("""<div class='title'>Modelling</div>""", unsafe_allow_html=True)
    st.markdown("""
        <div class='info-section'>
            <h2>Proses Modelling</h2>
            <p>Proses modelling meliputi pelatihan model menggunakan algoritma Support Vector Machine (SVM) dengan Stratified K-Fold Cross-Validation.</p>
        </div>
    """, unsafe_allow_html=True)

    # Menampilkan gambar dari folder lokal
    st.image("blok.png", caption="Diagram Alur Klasifikasi", use_column_width=True)

    # Proses Modelling
    st.subheader("Langkah-langkah Modelling")

    # Cek apakah variabel sudah didefinisikan
    if 'X_normalized_smote' not in st.session_state or 'y_resampled' not in st.session_state:
        st.error("Silakan jalankan menu Pre-Processing terlebih dahulu untuk mendefinisikan variabel.")
    else:
        # 1. Dataset Splitting
        st.markdown("""
            **1. Dataset Splitting**  
            - Dataset dibagi menjadi data training (80%) dan data testing (20%) menggunakan `train_test_split`.  
            - Pembagian dilakukan secara stratifikasi untuk memastikan proporsi kelas tetap seimbang.  
        """)
        # Dataset splitting
        X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(
            st.session_state.X_normalized_smote, st.session_state.y_resampled, test_size=0.2, random_state=42
        )
        st.write("Jumlah data training:", X_train_smote.shape[0])
        st.write("Jumlah data testing:", X_test_smote.shape[0])

        # 2. Training Model dengan K-Fold Cross-Validation
        st.markdown("""
            **2. Training Model dengan K-Fold Cross-Validation**  
            - Model SVM dilatih menggunakan Stratified K-Fold Cross-Validation dengan 5 fold.  
            - Setiap fold dievaluasi menggunakan metrik akurasi.  
        """)
        # Inisialisasi model SVM
        svm = SVC(probability=True, random_state=42)

        # Inisialisasi Stratified K-Fold
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Variabel untuk menyimpan akurasi dan model terbaik
        best_accuracy = 0
        best_model = None
        accuracy_per_fold = []

        # Evaluasi dengan Cross Validation
        st.write("Proses K-Fold Cross-Validation:")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_smote, y_train_smote)):
            # Split data
            X_train_fold, X_val = X_train_smote.iloc[train_idx], X_train_smote.iloc[val_idx]
            y_train_fold, y_val = y_train_smote.iloc[train_idx], y_train_smote.iloc[val_idx]

            # Train model
            svm.fit(X_train_fold, y_train_fold)

            # Predict dan hitung akurasi
            y_pred = svm.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            accuracy_per_fold.append(accuracy)
            st.write(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

            # Simpan model dengan akurasi tertinggi
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = svm

        # Rata-rata akurasi
        mean_accuracy = np.mean(accuracy_per_fold)
        st.write(f"\nRata-rata Akurasi: {mean_accuracy:.4f}")

        # Menampilkan grafik akurasi per fold
        st.subheader("Grafik Akurasi per Fold")
        plt.figure(figsize=(8, 5))
        plt.bar(range(1, len(accuracy_per_fold) + 1), accuracy_per_fold, color='skyblue', alpha=0.8, label='Accuracy per Fold')
        plt.axhline(mean_accuracy, color='red', linestyle='--', label=f'Mean Accuracy ({mean_accuracy:.4f})')
        plt.title('Accuracy per Fold (SVM with Stratified K-Fold)')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.xticks(range(1, len(accuracy_per_fold) + 1))
        plt.ylim(0, 1)  # Mengatur skala Y untuk akurasi (0-1)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(plt)

        # 3. Evaluasi Model dengan Confusion Matrix
        st.markdown("""
            **3. Evaluasi Model dengan Confusion Matrix**  
            - Model terbaik diuji pada data testing.  
            - Confusion matrix digunakan untuk mengevaluasi performa model.  
        """)
        # Prediksi dengan model terbaik
        y_pred = best_model.predict(X_test_smote)

        # Hitung confusion matrix
        cm = confusion_matrix(y_test_smote, y_pred)

        # Hitung akurasi
        accuracy = accuracy_score(y_test_smote, y_pred)
        st.write(f"Akurasi pada Data Testing: {accuracy:.4f}")

        # Tampilkan confusion matrix
        st.subheader("Confusion Matrix")
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test_smote), yticklabels=np.unique(y_test_smote))
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        st.pyplot(plt)

        # Tampilkan classification report
        st.subheader("Classification Report")
        report = classification_report(y_test_smote, y_pred, output_dict=True)
        st.write(pd.DataFrame(report).transpose())

        # Simpan model terbaik
        with open('best_model.pkl', 'wb') as file:
            pickle.dump(best_model, file)
        st.write("Model terbaik telah disimpan sebagai `best_model.pkl`.")

elif menu == "Implementation":
    st.markdown("""<div class='title'>Implementation</div>""", unsafe_allow_html=True)
    st.markdown("""
        <div class='info-section'>
            <h2>Implementasi Model</h2>
            <p>Implementasi model untuk prediksi diabetes berdasarkan input pengguna.</p>
        </div>
    """, unsafe_allow_html=True)

    # Input fitur dari pengguna
    st.header("Input Fitur Pasien")

    jk = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    umur = st.number_input("Umur", min_value=0, max_value=120, value=25)
    hb = st.number_input("HB (Hemoglobin)", min_value=0.0, value=12.0)
    hct = st.number_input("HCT (Hematokrit)", min_value=0.0, value=38.0)
    wbc = st.number_input("WBC (White Blood Cells)", min_value=0.0, value=7000.0)
    rbc = st.number_input("RBC (Red Blood Cells)", min_value=0.0, value=4.5)
    plt = st.number_input("PLT (Platelet)", min_value=0.0, value=250000.0)
    gda = st.number_input("GDA (Glucose Darah Acak)", min_value=0.0, value=100.0)

    # Preprocessing input
    # Encoding untuk Jenis Kelamin (0 = Laki-laki, 1 = Perempuan)
    jk_encoded = 0 if jk == "Laki-laki" else 1

    # Membuat DataFrame dari input
    input_data = pd.DataFrame({
        'JK': [jk_encoded],
        'Umur': [umur],
        'HB': [hb],
        'HCT': [hct],
        'WBC': [wbc],
        'RBC': [rbc],
        'PLT': [plt],
        'GDA': [gda]
    })

    # Normalisasi data
    scaler = MinMaxScaler()
    input_normalized = scaler.fit_transform(input_data)

    # Tombol untuk prediksi
    if st.button("Prediksi"):
        # Aturan bisnis: Jika GDA ≥ 126, otomatis Diabetes Mellitus (-1)
        if gda >= 126:
            prediction = -1  # Diabetes Mellitus
            st.subheader("Hasil Prediksi")
            st.write("**Hasil Prediksi:** Diabetes Mellitus")
        else:
            # Jika GDA < 126, gunakan model untuk prediksi
            prediction = model.predict(input_normalized)[0]
            st.subheader("Hasil Prediksi")
            if prediction == -1:
                st.write("**Hasil Prediksi:** Diabetes Mellitus")
            else:
                st.write("**Hasil Prediksi:** Non-Diabetes Mellitus")

    # Menampilkan data input
    st.subheader("Data Input yang Dimasukkan")
    st.write(input_data)

# Footer
st.markdown("""
    <div class='footer'>
        <p>©Ari Bagus Firmansyah 2025 Deteksi Dini Penyakit Diabetes. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)