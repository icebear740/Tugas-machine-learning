# Tugas3 Machine Learning

**Nama**: Reival Muhamad Asyari Uzzukru  
**NPM**: 41155050210018  
**Kelas**: A1 
## 1.1.	Pengenalan komponen Decision Tree: root, node, leaf
**Decision tree adalah alat visual yang digunakan dalam pengambilan keputusan, terutama dalam bidang ilmu komputer dan statistik. Pohon ini membantu kita untuk membuat keputusan berdasarkan serangkaian pertanyaan atau kondisi.
Komponen Utama Pohon Keputusan:**
- Node (simpul): Setiap kotak dalam diagram disebut node.
- Root node: Node paling atas, merupakan titik awal pengambilan keputusan. Dalam contoh ini, pertanyaan pertama adalah "Fear technology?".
- Internal node: Node di antara root node dan leaf node, mewakili pertanyaan-pertanyaan selanjutnya. Contohnya, "Your dad rich?" dan "Care about privacy?".
- Leaf node: Node paling bawah, mewakili hasil akhir atau keputusan yang diambil. Dalam contoh ini, leaf node adalah ikon-ikon sistem operasi (Apple, Chrome, Windows, Linux, Android).
## 1.2.	Pengenalan Gini Impurity
**Gini impurity adalah salah satu metrik yang paling umum digunakan dalam algoritma pohon keputusan untuk mengukur kemurnian (atau ketidakmurnian) dari sebuah node. Semakin murni sebuah node, semakin baik pemisahan data yang dilakukan oleh node tersebut.**
## 1.3.	Pengenalan Information Gain
**Information Gain adalah metrik yang digunakan dalam algoritma pohon keputusan untuk mengukur seberapa banyak informasi baru yang diberikan oleh suatu atribut saat membagi dataset. Dengan kata lain, Information Gain menunjukkan seberapa efektif suatu atribut dalam mengurangi ketidakpastian (impurity) pada data.**
## 1.4.	Membangun Decision Tree
![image](https://github.com/user-attachments/assets/7dd27cbc-cff7-48a1-9b7d-aac7f3f03171)
-	Dataset: Kita memiliki dataset kecil yang berisi informasi tentang buah, yaitu warna (Color) dan diameter (Diameter), serta label buah (Label).
-	Pohon Keputusan: Gambar menunjukkan permulaan pembuatan pohon keputusan. Node akar (root node) belum dipecah menjadi cabang-cabang.
-	Pertanyaan Pemisahan: Terdapat beberapa pertanyaan potensial untuk membagi data, seperti "Color == Green?", "Color == Yellow?", dan sebagainya.
-	Gini Impurity: Nilai Gini Impurity dihitung untuk mengukur ketidakmurnian dari dataset. Semakin kecil nilai Gini Impurity, semakin murni suatu node.
**Rumus
Gini Impurity = 1 - Σ (p_i)^2 ( p_i adalah proporsi data yang termasuk dalam kelas )
Hitung proporsi setiap kelas:**
- Apple: 2/5
-	Grape: 2/5
-	Lemon: 1/5
**Gini Impurity = 1 - ((2/5)^2 + (2/5)^2 + (1/5)^2)
Hitung : Gini Impurity = 1 - (4/25 + 4/25 + 1/25) = 1 - 9/25 = 0.63
interpretasi:
Nilai Gini Impurity sebesar 0.63 menunjukkan bahwa dataset pada node akar masih cukup tidak murni. Artinya, data masih tercampur cukup banyak antara kelas Apple, Grape, dan Lemon.
Tujuan dari pembuatan pohon keputusan adalah untuk terus membagi data sehingga setiap node memiliki Gini Impurity yang semakin kecil (semakin murni). Dengan kata lain, kita ingin memisahkan data sedemikian rupa sehingga setiap node berisi data yang dominan berasal dari satu kelas saja**

## 1.5.	Persiapan dataset: Iris Dataset
![image](https://github.com/user-attachments/assets/bdf19eb8-cde7-4fc9-96f5-24ab335a7d89)
## 1.6.	Training model Decision Tree Classifier
![image](https://github.com/user-attachments/assets/aa4fb778-635e-48d5-b05a-53f606549751)
## 1.7.	Visualisasi model Decision Tree
![image](https://github.com/user-attachments/assets/563aeb78-a57e-4a6d-9f32-11d05dd449f7)
![image](https://github.com/user-attachments/assets/62e24b0a-8108-43c0-a4a9-e48f0f7f92d3)
## 1.8.	Evaluasi model Decision Tree
![image](https://github.com/user-attachments/assets/c4e9d442-cb1b-4ef2-83a0-f9fc5905ec24)
## 2.1	Proses training model Machine Learning secara umum 
![image](https://github.com/user-attachments/assets/fe2b4d9b-725a-42bc-8212-7253945cf7ac)
-	X_train: Ini adalah kumpulan data fitur (independent variables) yang digunakan untuk melatih model. Fitur-fitur ini adalah karakteristik data yang akan digunakan model untuk membuat prediksi.
-	y_train: Ini adalah kumpulan label atau target (dependent variables) yang sesuai dengan setiap data fitur di X_train. Label ini adalah hasil yang ingin kita prediksi oleh model.

-	Model:
Ini adalah algoritma Machine Learning yang dipilih (misalnya, Regresi Linear, Decision Tree, Neural Network). Algoritma ini akan belajar dari data pelatihan (X_train dan y_train) untuk menemukan pola hubungan antara fitur dan target.
-	Pelatihan Model:
Data pelatihan (X_train dan y_train) dimasukkan ke dalam model.
Model akan menyesuaikan parameter internalnya untuk meminimalkan kesalahan antara prediksi model dan label sebenarnya (y_train). Proses ini disebut sebagai pelatihan atau fitting.
-	Trained Model:
Setelah pelatihan selesai, kita mendapatkan model yang telah "belajar" dari data pelatihan. Model ini sekarang dapat digunakan untuk membuat prediksi pada data baru.
-	X_new:
Ini adalah data baru yang belum pernah dilihat oleh model sebelumnya. Data ini memiliki fitur yang sama dengan data pelatihan.
-	y_pred:
Ini adalah prediksi yang dihasilkan oleh model yang telah dilatih ketika diberikan data baru (X_new). Prediksi ini adalah hasil yang diharapkan dari model.

## 2.2	 Pengenalan Ensemble Learning
![image](https://github.com/user-attachments/assets/b4130099-6c56-4d1c-83f2-1d82549d83ac)
**Konsep Homogeneous dalam konteks Ensemble Learning merujuk pada penggunaan beberapa model dasar yang memiliki jenis yang sama.**
- Training Set:
	**X_train: Data fitur atau atribut yang digunakan untuk melatih model.**
	**y_train: Label atau target yang sesuai dengan setiap data fitur.**
-  Model Dasar:
	**KNN (K-Nearest Neighbors): Sebuah algoritma yang mengklasifikasikan data baru berdasarkan k data terdekatnya.**
	**SVM (Support Vector Machine): Algoritma yang mencari hyperplane terbaik untuk memisahkan data ke dalam kelas yang berbeda.**
  **Decision Tree: Algoritma yang membuat pohon keputusan untuk mengklasifikasikan data.**
- Pelatihan Model:
	Setiap model dasar dilatih secara independen menggunakan data pelatihan yang sama (X_train dan y_train).
- Prediksi:
	Setiap model dasar membuat prediksi (y_pred) untuk data baru (X_new).
- Penggabungan Prediksi:
	Prediksi dari semua model dasar digabungkan untuk menghasilkan prediksi akhir (y_pred final). Dalam diagram ini, metode penggabungan yang digunakan adalah mean (rata-rata) atau mode (nilai yang paling sering muncul).
## 2.3	Pengenalan Bootstrap Aggregating | Bagging
![image](https://github.com/user-attachments/assets/5d465c1f-de1f-46de-8496-c96de8c8127a)
- Training Set: Dataset asli yang akan digunakan untuk melatih model. 
- Random Sampling with Replacement: Proses pengambilan sampel data secara acak dengan pengembalian untuk membuat beberapa bag. 
- Bag 1, Bag 2, Bag 3: Setiap bag berisi sampel data yang berbeda yang diambil dari dataset asli. 
- Model: Model yang dilatih pada setiap bag. 
- Trained Model 1, Trained Model 2, Trained Model 3: Model yang telah dilatih dan siap digunakan untuk membuat prediksi. 
- X_new: Data baru yang ingin diprediksi. 
- y_pred: Prediksi akhir yang diperoleh dari menggabungkan prediksi semua model.
## 2.4	Pengenalan Random Forest | Hutan Acak
![image](https://github.com/user-attachments/assets/07502efa-bb89-4ef8-aa90-47e6e3f7700b)
- Persiapan Training Set:
Data pelatihan terdiri dari fitur (X_train) dan label (y_train).
Data ini digunakan untuk melatih beberapa decision tree secara terpisah.
- Bagging + Keacakan Fitur:
Random Forest menerapkan Bagging (Bootstrap Aggregating). Ini membuat beberapa subset acak dari data asli (disebut bag) dengan penggantian (with replacement), artinya beberapa titik data dapat muncul beberapa kali dalam satu subset.
Selain itu, setiap pohon hanya menggunakan subset acak dari fitur, sehingga mengurangi korelasi antar pohon dan meningkatkan kemampuan generalisasi model.
- Decision Tree:
Setiap bag digunakan untuk melatih satu Decision Tree secara terpisah. Dalam contoh ini, terdapat 3 bag, menghasilkan 3 model Decision Tree terlatih (Model 1, Model 2, dan Model 3).
Masing-masing pohon belajar pola secara independen dari subset datanya.
- Prediksi dengan Data Baru:
Saat input baru (X_new) diberikan, data tersebut diproses melalui masing-masing model (Decision Tree).
Setiap pohon memberikan prediksinya sendiri terhadap data baru tersebut.
- Menggabungkan Prediksi:
Untuk klasifikasi, prediksi dari semua pohon digabungkan menggunakan mode (voting mayoritas), di mana kelas yang paling sering muncul dipilih sebagai hasil akhir.
Untuk regresi, prediksi dari semua pohon dihitung rata-ratanya (mean) untuk memberikan output akhir.
- Prediksi Akhir:
Hasil gabungan dari semua pohon adalah prediksi akhir, yang disebut y_pred.
## 2.5	 Persiapan dataset | Iris Flower Dataset
![image](https://github.com/user-attachments/assets/acce1640-8b78-4ff2-bd11-fe63548c0c72)
## 2.6	Implementasi Random Forest Classifier dengan Scikit Learn
![image](https://github.com/user-attachments/assets/13477d08-9b20-487b-9643-d063a95470e1)
## 2.7	Evaluasi model  dengan Classification Report
![image](https://github.com/user-attachments/assets/d31350d7-92c1-41ed-b383-8e46e46fdb46)
