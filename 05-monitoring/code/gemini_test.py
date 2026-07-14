# from google import genai
# from dotenv import load_dotenv
# import os

# load_dotenv()

# print("API Key:", os.getenv("GEMINI_API_KEY")[:8] + "...")

# client = genai.Client(
#     api_key="AQ.Ab8RN6K3aynA7FgMw0h7LWLhgo-nm5fIsL-3siQuktwWeL21Ow"
# )

# response = client.models.generate_content(
#     model="gemini-2.5-flash",
#     contents="Reply with exactly: Hello from Gemini!"
# )

# print(response.text)

#=================================
from dotenv import load_dotenv
import os

load_dotenv(override=True)

print("GEMINI_API_KEY =", repr(os.getenv("GEMINI_API_KEY")))
print("Current directory =", os.getcwd())