import numpy as np
from huggingface_hub import hf_hub_download
import onnxruntime as ort
from tokenizers import Tokenizer
from gitsource import GithubRepositoryDataReader, chunk_documents

# 1. جلب البيانات وتقطيعها للحصول على أول Chunk كما فعلنا سابقاً
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d"
)
documents = reader.read()

filtered_docs = []
for doc in documents:
    if doc.filename.endswith('.md') and '01-agentic-rag' in doc.filename:
        filtered_docs.append({"filename": doc.filename, "content": doc.content})

chunks = chunk_documents(documents=filtered_docs, size=2000, step=1000)
first_chunk_text = chunks[0]['content']

# 2. تحميل وإعداد نموذج التضمين ONNX
model_path = hf_hub_download(repo_id="Xenova/all-MiniLM-L6-v2", filename="onnx/model.onnx")
tokenizer_path = hf_hub_download(repo_id="Xenova/all-MiniLM-L6-v2", filename="tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)
session = ort.InferenceSession(model_path)

# دالة مساعدة لتوليد الـ Embedding وعمل Normalization له مباشرة
def get_embedding(text):
    encoded = tokenizer.encode(text)
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
    
    outputs = session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids
    })
    
    # Mean Pooling
    token_embeddings = outputs[0]
    input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = np.clip(input_mask_expanded.sum(1), a_min=1e-9, a_max=None)
    sentence_embedding = sum_embeddings / sum_mask
    
    # Normalization
    norm = np.linalg.norm(sentence_embedding, axis=1, keepdims=True)
    return (sentence_embedding / norm)[0]

# 3. حساب الـ Embeddings للنصين
query_text = "How does approximate nearest neighbor search work?"
v_query = get_embedding(query_text)
v_chunk = get_embedding(first_chunk_text)

# 4. حساب Cosine Similarity
# بما أن المتجهات معمل لها Normalization بالفعل، فإن حاصل الضرب القياسي (Dot Product) هو نفسه الـ Cosine Similarity
similarity = np.dot(v_query, v_chunk)

print("\n" + "="*40)
print(f"قيمة تشابه جيب التمام (Cosine Similarity) هي: {similarity:.2f}")
print("="*40)
for i in range(min(5, len(chunks))):
    v_c = get_embedding(chunks[i]['content'])
    sim = np.dot(v_query, v_c)
    print(f"التشابه مع Chunk رقم {i}: {sim:.2f}")