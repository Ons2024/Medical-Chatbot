from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
API_KEY=os.environ.get('API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["API_KEY"] = API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medical-bot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


chatModel = ChatOpenAI(
    openai_api_base=os.getenv("API_KEY"),
    openai_api_key="sk-no-key-needed",
    model_name="deepseek-coder:6.7b",  # Vérifie le nom exact dans LM Studio
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})

    answer = response["answer"]
    clean_answer = answer.split("</think>")[-1].strip()

    return (clean_answer)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)