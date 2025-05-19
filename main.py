# main.py — FastAPI + LangGraph chatbot with upload check

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
import shutil
import os
from dotenv import load_dotenv

load_dotenv()  
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5501"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    
)


vectorstore = None
retriever = None

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
parser = StrOutputParser()

prompt_template = ChatPromptTemplate.from_template("""
Use the following context to answer the question. If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}
""")

def retrieve_node(state):
    global retriever
    if not retriever:
        return {"context": "", "output": "❌ No document uploaded. Please upload a PDF first."}
    docs = retriever.invoke(state["input"])
    context = "\n\n".join(doc.page_content for doc in docs)
    return {"context": context, "input": state["input"]} 

def prompt_node(state):
    prompt = prompt_template.format(question=state["input"], context=state["context"])
    return {"prompt": prompt}

def llm_node(state):
    response = llm.invoke(state["prompt"])
    return {"raw_response": response}

def parse_node(state):
    if "output" in state:
        return {"output": state["output"]}
    return {"output": parser.invoke(state["raw_response"])}

builder = StateGraph(input=str, output=str)
builder.add_node("retrieve", retrieve_node)
builder.add_node("prompt", prompt_node)
builder.add_node("llm", llm_node)
builder.add_node("parse", parse_node)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "prompt")
builder.add_edge("prompt", "llm")
builder.add_edge("llm", "parse")
builder.add_edge("parse", END)

graph = builder.compile()

class ChatRequest(BaseModel):
    input: str

@app.post("/chat")
def chat(req: ChatRequest):
    if not retriever:
        return JSONResponse(content={"output": "❌ Please upload a PDF before asking questions."}, status_code=200)
    input_state = {"input": req.input}
    result = graph.invoke(input_state)
    return {"output": result["output"]}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    print("📥 Upload endpoint triggered")

    # Save the file
    os.makedirs("temp_uploads", exist_ok=True)
    file_path = f"temp_uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print("📝 File saved. Returning success response early.")

    # Fire and forget (in background)
    import threading
    threading.Thread(target=process_file, args=(file_path,)).start()

    return JSONResponse(content={"message": "✅ File uploaded. Processing in background."})


def process_file(file_path):
    try:
        print("📄 Background: Loading PDF")
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        print(f"📚 Background: {len(chunks)} chunks created")

        global vectorstore, retriever
        vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        print("🧠 Background: Retriever ready")

    except Exception as e:
        print("❌ Background processing failed:", e)