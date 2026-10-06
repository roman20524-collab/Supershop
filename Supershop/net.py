import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='templates')

# --- КОНФИГУРАЦИЯ И ДАННЫЕ ---
CATEGORY_NAMES = ['Электроника', 'Одежда', 'Дом и уют', 'Спорт', 'Книги']

recommendation_db = {
    'Электроника': [
        '🎧 Беспроводные наушники AirBeat (скидка 15%)',
        '⌚ Смарт-часы Chrono Sport (новинка)',
        '🎵 Портативная колонка BassBox (компактная)'
    ],
    'Одежда': [
        '👖 Джинсы Regular Fit (классика)',
        '👕 Футболка Cotton Classic (разные цвета)',
        '🎒 Рюкзак городской Stealth (вместительный)'
    ],
    'Дом и уют': [
        '💨 Увлажнитель воздуха MistMaster (тихий)',
        '💡 Настольная LED лампа (регулируемая яркость)',
        '☕ Кофемашина Barista Home (автоматический капучинатор)'
    ],
    'Спорт': [
        '🧘 Фитнес-коврик толстый (антискользящий)',
        '🥤 Бутылка для воды sport 1L (с фильтром)',
        '💪 Эспандеры резиновые (3 уровня)'
    ],
    'Книги': [
        '📖 Паттерны проектирования (бестселлер)',
        '📚 Психология влияния (Р. Чалдини) (новое издание)',
        '💻 Грокаем алгоритмы (А. Бхаргава) (с примерами на Python)'
    ]
}

train_data = np.array([
    # Электроника
    'Смартфон Nexus Pro 5G', 'Беспроводные наушники AirBeat', 'Ультрабук Aero 14"',
    'Смарт-часы Chrono Sport', 'Планшет Tab Ultra 11"', 'Портативная колонка BassBox',
    # Одежда
    'Худи Oversize Basic', 'Кроссовки Urban Runner', 'Джинсы Regular Fit',
    'Куртка-бомбер утепленная', 'Футболка Cotton Classic', 'Рюкзак городской Stealth',
    # Дом и уют
    'Кофемашина Barista Home', 'Робот-пылесос CleanBot', 'Настольная LED лампа',
    'Увлажнитель воздуха MistMaster', 'Набор керамической посуды', 'Электрочайник SmartTemp',
    # Спорт
    'Набор гантелей разборных', 'Фитнес-коврик толстый', 'Велосипед горный Trail 29',
    'Скакалка со счетчиком калорий', 'Бутылка для воды sport 1L', 'Эспандеры резиновые (набор)',
    # Книги
    'Чистый код (Р. Мартин)', 'Паттерны проектирования', 'Искусство войны (Сунь Цзы)',
    'Алгебра разума (Нейросети)', 'Психология влияния (Р. Чалдини)', 'Грокаем алгоритмы (А. Бхаргава)'
], dtype=object)

train_labels = np.array([0]*6 + [1]*6 + [2]*6 + [3]*6 + [4]*6)

MODEL_PATH = 'recommender_model.keras'

# --- ФУНКЦИИ МОДЕЛИ ---
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        print(f"✅ Загружаем готовую модель из {MODEL_PATH}")
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    else:
        print("🧠 Модель не найдена. Обучаем с нуля...")
        vectorizer = layers.TextVectorization(max_tokens=1000, output_sequence_length=10)
        vectorizer.adapt(train_data)

        model = tf.keras.Sequential([
            vectorizer,
            layers.Embedding(input_dim=1000, output_dim=32),
            layers.GlobalAveragePooling1D(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(CATEGORY_NAMES), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_data, train_labels, epochs=50, verbose=0)
        model.save(MODEL_PATH)
        print(f"✅ Модель обучена и сохранена в {MODEL_PATH}")
        return model

# Глобальная переменная для модели
model = load_or_train_model()

def get_recommendation(user_text: str):
    text = np.array([user_text], dtype=object)
    pred = model.predict(text, verbose=0)
    category_idx = int(np.argmax(pred))
    category = CATEGORY_NAMES[category_idx]
    recs = recommendation_db.get(category, )[:3]

    if not recs:
        return {"category": category, "recommendations": []}

    return {
        "category": category,
        "recommendations": recs
    }


# --- ROUTES (МАРШРУТЫ) ---

# 1. Отдаём HTML файл
@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

# 2. API для рекомендаций (вызывается из JS)
@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    data = request.json
    user_text = data.get('query', '')
    if not user_text:
        return jsonify({"error": "Нет запроса"}), 400

    result = get_recommendation(user_text)
    return jsonify(result)

if __name__ == '__main__':
    print("🚀 Сервер запущен: http://127.0.0.1:5000")
    app.run(debug=True)
