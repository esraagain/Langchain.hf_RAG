from fastapi import FastAPI, File, UploadFile
import shutil
import os
import pdfplumber
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from fastapi.responses import HTMLResponse

app = FastAPI()

# Configuration
PDF_STORAGE_PATH = "uploads/"
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

# Ensure upload directory exists
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
    <head>
        <title>PDF.LM</title>
        <script src="https://unpkg.com/htmx.org@1.9.2"></script>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 20px; background: #333; color: white; }
            form { margin: 20px auto; width: 50%; padding: 20px; border: 1px solid #555; border-radius: 10px; background: #444; }
            input, button { padding: 10px; margin: 10px; }
            button { background: blue; color: white; border: none; cursor: pointer; transition: 0.3s; border-radius: 15px; }
            button:hover { background: pink; }
            button:active { background: green; }
            #loading-bar { width: 0%; height: 5px; background: white; transition: width 0.5s; display: none; }
        </style>
    </head>
    <body>
        <h1>PDF.LM</h1>
        <form hx-post="/upload" hx-encoding="multipart/form-data" hx-target="#result" hx-on::before-request="
            document.getElementById('upload-btn').style.background = 'green';
            document.getElementById('upload-btn').innerText = 'Processing...';
            document.getElementById('loading-bar').style.display = 'block';
            document.getElementById('loading-bar').style.width = '50%';
        ">
            <input type="file" name="file" accept=".pdf" required>
            <button id="upload-btn" type="submit">Upload and Process</button>
        </form>
        <div id="loading-bar"></div>
        <div id="result"></div>
    </body>
    </html>
    """

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(PDF_STORAGE_PATH, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load and process document
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    DOCUMENT_VECTOR_DB.add_texts([chunk.page_content for chunk in chunks])

    return """
    <script>
        document.getElementById('loading-bar').style.width = '100%';
        setTimeout(() => document.getElementById('loading-bar').style.display = 'none', 1000);
        document.getElementById('upload-btn').innerText = 'Processed';
    </script>
    <p style='color:green;'>File processed successfully!</p>
    <form hx-get='/ask' hx-target='#answer' hx-on::before-request="
        document.getElementById('ask-btn').style.background = 'green';
        document.getElementById('ask-btn').innerText = 'Answering...';
    ">
        <input type='text' name='query' placeholder='Ask a question...' required>
        <button id="ask-btn" type='submit' style="border-radius: 8px; width: 120px;">Ask</button>
    </form>
    <div id='answer'></div>
    """

@app.get("/ask")
async def ask(query: str):
    retrieved_docs = DOCUMENT_VECTOR_DB.similarity_search(query, k=3)
    context = " ".join([doc.page_content for doc in retrieved_docs])
    prompt = PROMPT_TEMPLATE.format(user_query=query, document_context=context)
    
    # Fix: Use a list for `generate()`
    response = LANGUAGE_MODEL.generate([prompt]).generations[0][0].text.strip().replace('\n', ' ')

    return f"""
    <script>document.getElementById('ask-btn').innerText = 'Processed';</script>
    <p><strong>Answer:</strong> {response}</p>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)