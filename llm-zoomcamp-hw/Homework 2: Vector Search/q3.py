from gitsource import GithubRepositoryDataReader

# 1. تحميل الملفات
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d"
)
documents = reader.read()

target_files = [
    "02-vector-search/lessons/03-embeddings-dataset.md",
    "02-vector-search/lessons/06-rag-vector.md",
    "02-vector-search/lessons/07-sqlitesearch-vector.md",
    "02-vector-search/lessons/09-onnx-embedder.md"
]

print("--- استعراض العناوين ومحتويات الملفات الأربعة ---")
for doc in documents:
    for target in target_files:
        if target in doc.filename:
            print(f"\n" + "="*50)
            print(f"📁 اسم الملف: {doc.filename}")
            print("="*50)
            # طباعة السطور التي تبدأ بـ # (العناوين) لمعرفة محتوى الملف
            headings = [line for line in doc.content.split('\n') if line.strip().startswith('#')]
            print("🔍 العناوين الرئيسية في الملف:")
            for h in headings[:5]:
                print(f"   {h}")