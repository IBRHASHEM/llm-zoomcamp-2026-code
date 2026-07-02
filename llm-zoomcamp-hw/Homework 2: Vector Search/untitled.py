from gitsource import GithubRepositoryDataReader, chunk_documents

# 1. تعريف القارئ وتحميل البيانات من جيتهاب (تأكد أن هذا السطر موجود في البداية)
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d"
)

print("جاري تحميل الملفات من GitHub...")
documents = reader.read()
print(f"إجمالي الملفات المحملة: {len(documents)}")

# 2. تصفية الملفات وتحويلها إلى الصيغة المطلوبة Dict[str, str]
filtered_docs = []
for doc in documents:
    if doc.filename.endswith('.md') and '01-agentic-rag' in doc.filename:
        filtered_docs.append({
            "filename": doc.filename,
            "content": doc.content
        })

print(f"عدد ملفات الأسبوع الأول المفلترة: {len(filtered_docs)}")

# 3. تشغيل عملية التقطيع (Chunking)
chunks = chunk_documents(
    documents=filtered_docs,
    size=2000, 
    step=1000,
    content_field_name='content'
)

# 4. طباعة النتيجة لمعرفة إجابة السؤال الأول في الواجب!
print("\n" + "="*40)
print(f"إجابة السؤال الأول (Total Chunks): {len(chunks)}")
print("="*40)