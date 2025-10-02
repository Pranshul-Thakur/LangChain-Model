# LangChain PDF Q&A System

Simple RAG implementation using LangChain, Google Gemini, and FAISS for PDF document question-answering and summarization.

## Features

- PDF document loading and parsing
- Text chunking with overlap for context preservation
- FAISS vector store for efficient similarity search
- Google Generative AI embeddings
- Gemini 1.5 Flash for responses
- RetrievalQA chain for context-aware answers

## Installation

```bash
pip install langchain langchain-google-genai google-generativeai faiss-cpu pypdf
```

## Setup

Set your Google API key:
```python
os.environ["GOOGLE_API_KEY"] = "your_api_key_here"
```

## Usage

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load and chunk PDF
loader = PyPDFLoader("document.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Create vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Setup QA chain
llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Query document
result = qa("Summarize the document in bullet points")
print(result["result"])
```

## Configuration

- **Chunk size**: 1000 characters
- **Chunk overlap**: 200 characters
- **Model**: Gemini 1.5 Flash
- **Temperature**: 0 (deterministic)
- **Embedding model**: models/embedding-001
