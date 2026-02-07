from dotenv import load_dotenv
load_dotenv()
import os
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
import nest_asyncio
nest_asyncio.apply()
from tutor_rag_app import build_index, load_all_docs
if __name__ == "__main__":
    print("Loading documents...")
    docs = load_all_docs()
    print(f"Loaded {len(docs)} documents")
    if docs:
        print("Building index...")
        result = build_index()
        print(f"Build index result: {result}")
    else:
        print("No documents loaded")
