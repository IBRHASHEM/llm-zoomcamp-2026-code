import os
from gitsource import GitSource

# تحديد المستودع والنسخة المحددة المطلوبة في الواجب
repo_url = "https://github.com/DataTalksClub/llm-zoomcamp.git"
commit_hash = "8c1834d" # الـ Commit المحدد في الواجب لتوحيد البيانات

# تحميل الملفات
source = GitSource(repo_url, commit=commit_hash)
# هنا ستقوم بكتابة الكود الخاص بقراءة ملفات الـ Markdown (.md) من مجلد الأسبوع الأول