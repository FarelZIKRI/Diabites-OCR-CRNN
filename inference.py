import tensorflow as tf
import numpy as np
import cv2
import os
import json

# === KONFIGURASI ===
IMG_HEIGHT = 64
IMG_WIDTH = 480
MODEL_PATH = "saved_model/crnn_model.keras"
VOCAB_PATH = "saved_model/vocab.json"
TEST_PATH = "test-data-baru"        # Sumber data
OUTPUT_PATH = "hasil_data_baru"    # Folder hasil

# === COMPATIBILITY PATCH ===
from keras.src.layers.core.dense import Dense
original_dense_init = Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_dense_init(self, *args, **kwargs)
Dense.__init__ = patched_dense_init

def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    num_to_char = {int(k): v for k, v in data['num_to_char'].items()}
    return num_to_char

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None: return None, None
    
    # Simpan original untuk visualisasi (resize agar seragam)
    orig_viz = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    ratio = min(IMG_WIDTH / w, IMG_HEIGHT / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    img_resized = cv2.resize(img_rgb, (new_w, new_h))

    padded = np.ones((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8) * 255
    y_offset = (IMG_HEIGHT - new_h) // 2
    padded[y_offset:y_offset+new_h, 0:new_w] = img_resized
    
    normalized = padded.astype(np.float32) / 255.0
    return normalized, orig_viz

def ctc_decode_predictions(y_pred, num_to_char):
    input_len = np.ones(y_pred.shape[0]) * y_pred.shape[1]
    decoded, _ = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
        sequence_length=input_len.astype(np.int32)
    )
    dense = tf.sparse.to_dense(decoded[0], default_value=-1)
    results = []
    for seq in dense.numpy():
        text = ''.join([num_to_char.get(idx, '') for idx in seq if idx > 0])
        results.append(text)
    return results

def run_inference():
    # Buat folder output jika belum ada
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
        print("Error: Model/Vocab tidak ditemukan.")
        return

    print("Loading model...")
    num_to_char = load_vocab(VOCAB_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    if os.path.isfile(TEST_PATH):
        image_files = [TEST_PATH]
    elif os.path.isdir(TEST_PATH):
        image_files = [os.path.join(TEST_PATH, f) for f in os.listdir(TEST_PATH) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        print(f"Path tidak valid: {TEST_PATH}")
        return

    print(f"Memproses {len(image_files)} gambar...")
    for img_path in image_files:
        processed, viz_img = preprocess_image(img_path)
        if processed is None: continue
            
        batch = np.expand_dims(processed, axis=0)
        preds = model.predict(batch, verbose=0)
        result = ctc_decode_predictions(preds, num_to_char)[0]
        
        # Buat canvas hitam sedikit lebih tinggi untuk teks
        h, w = viz_img.shape[:2]
        canvas = np.zeros((h + 40, w, 3), dtype=np.uint8)
        canvas[40:, :] = viz_img
        
        # Tambahkan teks prediksi
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, f"Pred: {result}", (10, 30), font, 0.7, (0, 255, 0), 2)
        
        # Simpan hasil
        output_filename = f"res_{os.path.basename(img_path)}"
        save_path = os.path.join(OUTPUT_PATH, output_filename)
        cv2.imwrite(save_path, canvas)
        
        print(f"Tersimpan: {output_filename} -> '{result}'")

if __name__ == "__main__":
    run_inference()
