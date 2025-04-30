# 🚢 Titanic Survival Prediction with ANN

Proyek ini merupakan implementasi prediksi kelangsungan hidup penumpang Titanic menggunakan Artificial Neural Network (ANN) dengan TensorFlow, yang dilatih pada dataset dari kompetisi [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data).

---

## 📁 Fitur Utama

- ✅ Preprocessing data (handling missing values, encoding, scaling)
- ✅ Training model ANN dengan TensorFlow
- ✅ Evaluasi model dengan confusion matrix & classification report
- ✅ Export model ke format `.tflite` untuk deployment
- ✅ Simulasi prediksi menggunakan model TFLite
- ✅ Kompatibel untuk dijalankan di **Google Colab**

---

## 📂 Struktur Output Model

Setelah proses pelatihan dan deployment, kamu akan mendapatkan 3 file utama:

| File | Deskripsi |
|------|-----------|
| `titanic_model.tflite` | Model hasil pelatihan dalam format TensorFlow Lite |
| `scaler.pkl` | Scaler untuk transformasi fitur numerik (Age & Fare) |
| `label_encoder.pkl` *(opsional)* | Encoder untuk fitur kategorikal seperti Sex/Embarked jika diperlukan |

---

## 🚀🚀🚀

### 1. **Clone Repository**

```python
!git clone https://github.com/jasminemumtaz444/UTS_ML2_221351062.git
