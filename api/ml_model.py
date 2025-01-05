import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pelatihan model
def train_model():
    data = pd.read_csv(os.path.join(BASE_DIR, 'api/dataset/batik_data.csv'))
    print(data.head())
    data = data.dropna(subset=['name', 'suitable_for', 'description'])
    # Pisahkan kolom suitableFor menjadi daftar
    data['suitable_for'] = data['suitable_for'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])

    # Binarisasi label
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(data['suitable_for'])

    # Encode fitur (motif dan lainnya)
    X = pd.get_dummies(data[['name', 'description']])

    # Split data menjadi train dan test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Latih model Decision Tree
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Evaluasi model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")

    # Pastikan direktori untuk menyimpan model ada
    model_dir = os.path.join(BASE_DIR, 'api/models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Simpan model dan binarizer
    joblib.dump(model, os.path.join(model_dir, 'batik_classifier.pkl'))
    joblib.dump(mlb, os.path.join(model_dir, 'label_binarizer.pkl'))

# Prediksi kategori acara
def predict_category(features):
    # Load model dan binarizer
    model = joblib.load(os.path.join(BASE_DIR, 'api/models/batik_classifier.pkl'))
    mlb = joblib.load(os.path.join(BASE_DIR, 'api/models/label_binarizer.pkl'))

    # Encode fitur input
    features = pd.get_dummies(pd.DataFrame([features]))
    features = features.reindex(columns=model.feature_names_in_, fill_value=0)

    # Prediksi dan konversi kembali ke label
    predictions = model.predict(features)
    return mlb.inverse_transform(predictions)[0]

    # Fungsi untuk mencari data berdasarkan kolom 'suitable_for'
def search_by_suitable_for(suitable_for_value):
    data = pd.read_csv(os.path.join(BASE_DIR, 'api/dataset/batik_data.csv'))
    data = data.dropna(subset=['name', 'suitable_for', 'description'])
    data['suitable_for'] = data['suitable_for'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
    
    # Mencari data yang sesuai dengan nilai suitable_for_value
    result = data[data['suitable_for'].apply(lambda x: suitable_for_value in x)]
    return result

# Untuk pelatihan model (jalankan hanya jika diperlukan)
if __name__ == "__main__":
    train_model()