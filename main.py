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
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from typing import TypedDict, List, Optional

import shutil
from pydantic import BaseModel
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

class ChatRequest(BaseModel):
    message: str
    history: list[dict]

class ChatState(TypedDict):
    messages: List
    context: Optional[str]
    prompt: Optional[str]
    raw_response: Optional[str]
    output: Optional[str]

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

def retrieve_node(state: ChatState):
    global retriever
    print("🔍 Inside retrieve_node, keys:", list(state.keys()))
    
    if not retriever:
        return {
            "messages": state["messages"],
            "context": "",
            "output": "❌ No document uploaded. Please upload a PDF first."
        }

    latest_question = state["messages"][-1].content
    docs = retriever.invoke(latest_question)
    context = "\n\n".join(doc.page_content for doc in docs)
    print(f"📚 Retrieved context with {len(docs)} documents")
    
    return {
        "messages": state["messages"],
        "context": context
    }


def prompt_node(state: ChatState):
    print("👣 Inside prompt_node, keys:", list(state.keys()))
    
    # If we already have an output, skip processing
    if "output" in state:
        print("🚪 Skipping prompt_node, output already exists")
        return state

    # Get the question from the latest message
    question = state["messages"][-1].content
    context = state.get("context", "")
    
    # Create the prompt
    prompt = prompt_template.format(question=question, context=context)
    
    print(f"📝 Created prompt with context length: {len(context)}")
    
    return {
        "messages": state["messages"],
        "context": context,
        "prompt": prompt
    }


def llm_node(state: ChatState):
    print("🧠 Inside llm_node, keys:", list(state.keys()))
    
    # If we already have an output, skip processing
    if "output" in state:
        print("🚪 Skipping llm_node, output already exists")
        return state

    # Check if we have a prompt
    if "prompt" not in state:
        print("❌ No prompt found in state")
        return {
            "messages": state.get("messages", []),
            "context": state.get("context", ""),
            "output": "❌ Error: No prompt available for processing."
        }

    try:
        response = llm.invoke(state["prompt"])
        print("✅ LLM response received")
        
        return {
            "messages": state["messages"],
            "context": state.get("context", ""),
            "prompt": state["prompt"],
            "raw_response": response
        }
    except Exception as e:
        print(f"❌ LLM error: {e}")
        return {
            "messages": state.get("messages", []),
            "context": state.get("context", ""),
            "output": f"❌ Error processing request: {str(e)}"
        }


def parse_node(state: ChatState):
    print("📝 Inside parse_node, keys:", list(state.keys()))
    
    # If we already have an output, return it
    if "output" in state:
        print("🚪 Returning existing output")
        return {
            "messages": state.get("messages", []),
            "context": state.get("context", ""),
            "output": state["output"]
        }

    # Check if we have a raw response to parse
    if "raw_response" not in state:
        print("❌ No raw_response found in state")
        return {
            "messages": state.get("messages", []),
            "context": state.get("context", ""),
            "output": "❌ Error: No response to parse."
        }

    try:
        ai_text = parser.invoke(state["raw_response"])
        updated_messages = state["messages"] + [AIMessage(content=ai_text)]
        
        print(f"✅ Parsed response: {ai_text[:100]}...")

        return {
            "messages": updated_messages,
            "context": state.get("context", ""),
            "output": ai_text
        }
    except Exception as e:
        print(f"❌ Parse error: {e}")
        return {
            "messages": state.get("messages", []),
            "context": state.get("context", ""),
            "output": f"❌ Error parsing response: {str(e)}"
        }


builder = StateGraph(ChatState)
builder.add_node("retrieve_docs", retrieve_node)
builder.add_node("create_prompt", prompt_node)
builder.add_node("generate_response", llm_node)
builder.add_node("parse_output", parse_node)

builder.set_entry_point("retrieve_docs")
builder.add_edge("retrieve_docs", "create_prompt")
builder.add_edge("create_prompt", "generate_response")
builder.add_edge("generate_response", "parse_output")
builder.add_edge("parse_output", END)

graph = builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))

# class ChatRequest(BaseModel):
#     input: str

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        user_input = HumanMessage(content=req.message)
        prior_messages = [
            HumanMessage(content=m["content"]) if m["role"] == "user"
            else AIMessage(content=m["content"])
            for m in req.history
        ]

        input_state = {"messages": prior_messages + [user_input]}
        print("📤 Invoking graph with state:", input_state)

        result = graph.invoke(input_state)

        print("✅ Graph output:", result)

        # Check if we have an output
        if "output" not in result:
            print("❌ No output in result, providing fallback response")
            return {
                "response": "❌ Sorry, I couldn't process your request. Please try again.",
                "messages": [
                    {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
                    for m in result.get("messages", [])
                ]
            }

        return {
            "response": result["output"],
            "messages": [
                {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
                for m in result["messages"]
            ]
        }

    except Exception as e:
        print("❌ Exception in /chat:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


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