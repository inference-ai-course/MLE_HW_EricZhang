import requests

URL = "http://localhost:8000/api/voice-query"

for text in [
    "What is 2^10 + 3*7?",
    "Search arXiv for retrieval-augmented generation",
    "Tell me a fun fact about black holes",
]:
    r = requests.post(URL, json={"text": text}, timeout=30)
    print("\n---")
    print("Text:", text)
    print("Status:", r.status_code)
    try:
        print("JSON:", r.json())
    except Exception:
        print("Body:", r.text)
