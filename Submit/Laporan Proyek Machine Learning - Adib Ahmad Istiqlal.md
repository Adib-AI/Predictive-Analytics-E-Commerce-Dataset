# Laporan Proyek Machine Learning - Adib Ahmad Istiqlal
## Domain Proyek

Indonesia sebagai negara berkembang memiliki banyak parameter yang berkaitan mengenai stabilitas perkonomian maupun keuangan. Parameter yang bekaitan tersebut salah satunya adalah pendapatan e-commerce. Pada penelitian yang dilakukan Rianty dan Rahayu dengan judul [Pengaruh E-commerce Terhadap Pendapatan UMKM Yang Bermitra Gojek Dalam Masa Pandemi Covid-19](https://akuntansi.pnp.ac.id/jam/index.php/jam/article/download/159/115/), menyatakan bahwa e-commerce memiliki pendapatan yang baik bagi negara khususnya dari segi UMKM dengan peningkatan total transaksi hingga 5%.
Peningkatan yang cukup tinggi, pada permasalahan ini penulis ingin mengetahui bagaimana pendapatan e-commerce dari parameter penggunaaan sistem dan lamanya membership terhadap pendapatan e-commerce dengan menggunakan model machine learning classic menggunakan *support vector regression* yang merupakan kembangan *support vector machine* yang diperkenalkan oleh Vapnik pada tahun 1992. Pada penelitian [Support Vector Regression (SVR) Dalam Memprediksi Harga Minyak Kelapa Sawit di Indonesia dan Nilai Tukar Mata Uang EUR/USD](http://jcosine.if.unram.ac.id/index.php/jcosine/article/download/403/79/) yang dilakukan oleh Saadah, dkk pada tahun 2021 menghasilkan akurasi yang hampir mendekati 100% Terumata pada penggunaan kernel RBF.

## Business Understanding
**Pada pernyataan yang telah dijelaskan, sehingga masalah yang diangkat adalah**
- Bagaimana pengaruh penggunaan sistem e-commerce dan lamanya membership user terhadapa pendapatan oleh e-commerce.
- Bagaimana akurasi pendapatan e-commerce dengan sistem e-commerce dan lamanya membership user menggunakan kernel linear.

**Tujuan dari masalah yang diangkat adalah**
- Mengetahui pengaruh korelasi terhadap parameter tersebut terhadap pendapatan oleh e-commer
- Mengetahui akurasi kernel linear terhadap prediksi pendapatan e-commerce dengan parameter penggunaan sistem e-commerce dan lamanya membership user

**Solusi Statements**
Solusi yang dapat dilakukan
- Menggunakan korelasi dengan bantuan visualisasi heatmap dengan library seaborns
- Mengevaluasi hasil kernel linear *dengan mean squared error*
- Melakukan optimasi parameter kernel linear dengan parameter aslinya yaitu C, untuk meningkatkan hasil akurasi.
- 
## Data Understanding
Dataset yang digunakan pada penelitian ini adalah dataset pakaian secara online yang dapat dilakukan dari website atau app. Sumber dataset ini berasal dari [Kaggle.com](https://www.kaggle.com/datasets/kolawale/focusing-on-mobile-app-or-website?resource=download). Adapun kolom-kolom pada dataset ini, antara lain.
- E-mail : Alamat surat elektronik pengguna yang dapat digunakan sebagai ID.
- Address : Alamat tempat tinggal dari pengguna
- Avatar : Foto pengguna
- Avg. Session Length : Lamanya Session pengguna pada sistem yang tercatat
- Time on App : Lamanya penggunaan aplikasi perusahaan oleh pengguna
- Time on Website : Lamanya penggunaan aplikasi perusahaan oleh pengguna
- Length of Membership : Lamanya pengguna terdaftar
- Yearly Amount Spent : Pendapatan dari pengguna terhadap perusahaan.

Pada kolom diatas, **label** yang digunakan adalah kolom Yearly Amount Spent dan total dataset dari dataset ini berjumlah 500 baris.

Tahapan yang dilakukan untuk memahami data adalah.
- Teknik Visualisasi menggunakan matplotlib dan seaborn
- Statistik data menggunkan pandas

## Data Preparation
Tahapan yang dilakukan
- **Melakukan EDA**
   **1. Cek Null**
   ![This in an image](https://github.com/Adib-AI/Data_Science/blob/main/Predictive%20Analysis/Images/Cek_null.PNG?raw=True)

    **2. Cek Outlier**
    ![This in an image](https://github.com/Adib-AI/Data_Science/blob/main/Predictive%20Analysis/Images/Cek_outlier.PNG?raw=True)
    Disini saya tetap menggunakan outlier, meskipun data yang dimiliki sangat kecil. Saya tidak mengganti data pada nilai outliernya. Pada label yang digunakan, saya akan memprediksi nilai pada kolom Yearly Amount Spent. Menghasilkan total dataset baru sebear 476 baris.
        ```
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        
        IQR = Q3 - Q1
        
        df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis = 1)]
        
        df.shape
        ```
        
    **3. Korelasi parameter terhadap label**
    ![This in an image](https://github.com/Adib-AI/Data_Science/blob/main/Predictive%20Analysis/Images/Korelasi_kolom.PNG?raw=True)

- **Preprocessing** 
    **1. Reduction feature**
    Disini saya tidak menggunakan PCA dikarenakan tidak adanya korelasi yang tinggi antar fitur yang sama. Menurut perkiraan saya, Time On Website dengan Avg Session dapat dilakukan PCA. Namun dengan korelasi yang cukup rendah. Hal tersebut tidak perlu dilakukan dan yang saya gunakan hanyalah korelasi dengan rentang mendekati -1 dan +1

        ```
        X = df[['Avg. Session Length', 'Time on App', 'Length of Membership']]
        y = df['Yearly Amount Spent']
        ```
    **2. plit data (75%:25%)**
    Untuk pembagian dataset, saya menggunakan 75% (Train) : 25% (Test) karena mengingat dataset yang kecil
    
        ```
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 250)
        print(f'Total of Dataset : {len(X)}')
        print(f'Total of Train Dataset : {len(X_train)}')
        print(f'Total of Test Dataset : {len(X_test)}')
        ```
        
    **3. Standardization**
    Jenis standadization yang digunakan adalah StandardScaler milik sklearn.
        ```
        kolom = ['Avg. Session Length', 'Time on App', 'Length of Membership']
        scaler = StandardScaler()
        scaler.fit(X_train[kolom])
        X_train[kolom] = scaler.transform(X_train.loc[:, kolom])
        X_train[kolom].head()
        ```

## Modeling
Pada proses modeling, model yang digunakan SVR dikarenakan permasalahan regresi dengan jenis kernel linear. Pada tahapan ini terdapat dua tahapan, yaitu tanpa optimasi parameter dan menggunakan optimasi parameter dari kernel linear itu sendiri (nilai C) dengan rentang nilai 1-20. Pada penelitian yang dilakukan Noviana Pratiwi dan Yudi Setyawan berjudul [ANALISIS AKURASI DARI PERBEDAAN FUNGSI KERNEL DAN COST PADA SUPPORT VECTOR  MACHINE STUDI KASUS KLASIFIKASI CURAH HUJAN DI JAKARTA](https://ejournal2.undip.ac.id/index.php/jfma/article/download/11691/6606), menjelaskan bahwa parameter C merupakan parameter untuk mengontrol nilai error yang berpengaruh pada margin yang terbentuk.
Tahapan yang dilakukan ialah:
1. Mengimport library SVR dari sklearn dan membuat variable yang berisi SVR
    ```
    #Tanpa Optimasi
    model = SVR(kernel = 'linear')
    model.fit(X_train, y_train)
    ```
2. Mengimport library SVR dari sklearn dan membuat variable yang berisi SVR dan optimasi parameter C
    ```
    #Menggunakan Optimasi C
    for c in range(1, 21):
    models = SVR(kernel = 'linear', C = c)
    models.fit(X_train, y_train)
    ```
##### Adapun keunggulan dan kekurangan dari model SVR.

##### **Keunggulan**
- SVR mampu menghindari overfiting
- SVR efektif untuk menggeneralisasi sampel data yang sedikit
- SVR mampu melakukan penyelesaian norm error pada saat 
  pinalti outlier selama fase pelatihan. Hal ini yang diketahui 
  dengan kernel trick

##### **Kekurangan**
-  kinerja SVR sangat bergantung  terhadap parameter di dalamnya

## Evaluation
Evaluasi yang digunakan pada hasil model ialah mean squared error. Alasan mengapa menggunakan metrik tersebut karena permasalahan yang diangkat mengenai regresi. Menurut Iwa Sungkawa dan Ries Tri Megasari pada penelitian [PENERAPAN UKURAN KETEPATAN NILAI RAMALAN DATA DERET WAKTU DALAM SELEKSI MODEL PERAMALAN VOLUME PENJUALAN PT SATRIAMANDIRI CITRAMULIA](https://media.neliti.com/media/publications/165961-ID-penerapan-ukuran-ketepatan-nilai-ramalan.pdf) menyatakan bahwa MSE merupakan salah satu model evaluasi terbaik pada masalah regresi. MSE sendiri bekerja melakukan perhitungan error antara nilai hasil prediksi dengan nilai sebesarnya. Berikut formula dari MSE.
![a](https://github.com/Adib-AI/Data_Science/blob/main/Predictive%20Analysis/Images/MSE_Formula.png?raw=True)
Hasil MSE yang didapatkan ialah
    1. Tanpa optimasi
    ![to](https://github.com/Adib-AI/Data_Science/blob/main/Predictive%20Analysis/Images/tanpa_optimasi.PNG?raw=True)
    2. Menggunakan Optimasi
    Train
    ![o](https://github.com/Adib-AI/Data_Science/blob/main/Predictive%20Analysis/Images/Optimasi_1.PNG?raw=True)
    Test
    ![02](https://github.com/Adib-AI/Data_Science/blob/main/Predictive%20Analysis/Images/Optimasi_2.PNG?raw=True)
    3. Hasil prediksi dengan nilai Real
    ![hpr](https://github.com/Adib-AI/Data_Science/blob/main/Predictive%20Analysis/Images/Hasil_prediksi_nilai_rill.PNG?raw=True)

Pada hasil diatas dapat disimpulkan bahwa, kernel linear tanpa nilai C dan menggunakan nilai C hasil MSE tidak cukup berbeda jauh. Namun hasil prediksi yang didapatkan pada index ke-1 pada tanpa nilai C dan menggunakan nilai C mengalami perbedaan yang signifikan sekitar 7%. Hal ini menyatakan bahwa kernel linear dengan permasalahan regresi masih belum cukup baik dan dapat dilakukan percobaan kernel RBF seperti pada penelitian [Support Vector Regression (SVR) Dalam Memprediksi Harga Minyak Kelapa Sawit di Indonesia dan Nilai Tukar Mata Uang EUR/USD](http://jcosine.if.unram.ac.id/index.php/jcosine/article/download/403/79/) yang dilakukan oleh Saadah, dkk