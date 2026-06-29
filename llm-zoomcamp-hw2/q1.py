import numpy as np
from huggingface_hub import hf_hub_download
import onnxruntime as ort
from tokenizers import Tokenizer

# 1. تحميل ملفات النموذج
print("جاري تحميل نموذج التضمين (Embedding Model)...")
model_path = hf_hub_download(repo_id="Xenova/all-MiniLM-L6-v2", filename="onnx/model.onnx")
tokenizer_path = hf_hub_download(repo_id="Xenova/all-MiniLM-L6-v2", filename="tokenizer.json")

# 2. إعداد الـ Tokenizer والمحرك
tokenizer = Tokenizer.from_file(tokenizer_path)
session = ort.InferenceSession(model_path)

# 3. النص المطلوب تضمينه
query = "How does approximate nearest neighbor search work?"

# 4. تحويل النص وتجهيز الـ 3 مدخلات المطلوبة
encoded = tokenizer.encode(query)
input_ids = np.array([encoded.ids], dtype=np.int64)
attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
# إضافة الجزء الناقص (Token Type IDs) عبارة عن أصفار بنفس الطول
token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

# تشغيل النموذج بجميع المدخلات المطلوبة
outputs = session.run(None, {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "token_type_ids": token_type_ids
})

# 5. عمل Mean Pooling والـ Normalization
token_embeddings = outputs[0]
input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
sum_embeddings = np.sum(token_embeddings * input_mask_expanded, 1)
sum_mask = np.clip(input_mask_expanded.sum(1), a_min=1e-9, a_max=None)
sentence_embedding = sum_embeddings / sum_mask

norm = np.linalg.norm(sentence_embedding, axis=1, keepdims=True)
v = (sentence_embedding / norm)[0]

# 6. طباعة القيمة النهائية
print("\n" + "="*40)
print(f"قيمة أول عنصر v[0] هي بدقة: {v[0]:.2f}")
print("="*40)