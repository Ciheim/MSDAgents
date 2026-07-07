from pymilvus import MilvusClient
from langchain_huggingface import HuggingFaceEmbeddings
from fms.chatbot import retrieve_documents, MILVUS_DB_PATH

client = MilvusClient(uri=str(MILVUS_DB_PATH))
emb = HuggingFaceEmbeddings(
    model_name='BAAI/bge-small-en-v1.5',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
)
print("Enter question to check embedding retrieval (type 'exit' to quit):")
while True:
    question = input("Question: ")
    if question.lower() == 'exit':
        break
    rows = retrieve_documents(client, emb, question, limit=3)
    print(len(rows))
    for doc, score in rows:
        print(round(score, 4), doc.metadata.get('source'), doc.metadata.get('name'))
if hasattr(client, 'close'):
    client.close()