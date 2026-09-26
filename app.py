
import os
from flask import Flask, request, jsonify, render_template_string

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


app = Flask(__name__)


# Get Gemini API key from environment variable
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")


# Load KT document
with open("rag_kt_document.txt", "r", encoding="utf-8") as f:
    kt_content = f.read()


# Create document
kt_document = Document(
    page_content=kt_content,
    metadata={"source": "rag_kt_document.txt"}
)

documents = [kt_document]


# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)


# Create Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)


# Create FAISS vector store
vector_store = FAISS.from_documents(
    chunks,
    embeddings
)


# Create retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# Create Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# RAG prompt
rag_prompt = ChatPromptTemplate.from_template("""
You are the InnovateCorp Knowledge Transfer Assistant.

Answer the user's question using ONLY the information provided
in the retrieved KT context.

Do not use outside knowledge.

If the answer cannot be found in the retrieved context, say:

"I could not find this information in the InnovateCorp KT."

Do not invent information.

Retrieved KT Context:
{context}

User Question:
{question}

Answer:
""")


# Format retrieved documents
def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )


# Create RAG chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)


# Home page
@app.route("/")
def home():

    return render_template_string("""
    <!DOCTYPE html>

    <html>

    <head>

        <title>InnovateCorp RAG Assistant</title>

        <style>

            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
            }

            h1 {
                margin-bottom: 10px;
            }

            input {
                width: 70%;
                padding: 12px;
                font-size: 16px;
            }

            button {
                padding: 12px 20px;
                font-size: 16px;
                cursor: pointer;
            }

            #answer {
                margin-top: 25px;
                padding: 20px;
                background: #f2f2f2;
                border-radius: 8px;
                white-space: pre-wrap;
            }

        </style>

    </head>


    <body>

        <h1>InnovateCorp Knowledge Assistant</h1>

        <p>
            Ask a question about InnovateCorp.
        </p>


        <input
            id="question"
            type="text"
            placeholder="Enter your question"
        >


        <button onclick="askQuestion()">
            Ask
        </button>


        <div id="answer"></div>


        <script>

            async function askQuestion() {

                const question =
                    document.getElementById("question").value;


                if (!question) {

                    document.getElementById("answer").innerText =
                        "Please enter a question.";

                    return;
                }


                document.getElementById("answer").innerText =
                    "Searching the InnovateCorp KT...";


                const response = await fetch("/ask", {

                    method: "POST",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify({
                        question: question
                    })

                });


                const data = await response.json();


                document.getElementById("answer").innerText =
                    data.answer;

            }

        </script>

    </body>

    </html>
    """)


# Ask endpoint
@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = data.get("question", "")

        if not question:
            return jsonify({
                "answer": "Please enter a question."
            })

        print("Question received:", question)
        print("Running RAG chain...")

        answer = rag_chain.invoke(question)

        print("RAG answer generated successfully.")

        return jsonify({
            "answer": answer
        })

    except Exception as e:
        print("ERROR IN /ask:")
        print(type(e).__name__)
        print(str(e))

        return jsonify({
            "answer": "An error occurred while processing your question.",
            "error": str(e)
        }), 500


# Run application
if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
