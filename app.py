import uuid
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import ollama

COLLECTION_NAME = "pdf_chunks"


# PDF Loader
def load_pdf(path):
    reader = PdfReader(path)

    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t

    # clean formatting
    text = text.replace("\n", " ")
    text = " ".join(text.split())

    return text


# Token-based Chunking (aligned to embedding model's tokenizer)
def create_chunks(text, tokenizer, chunk_size=200, overlap=32):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk = tokenizer.decode(chunk_tokens)
        chunks.append(chunk)

    return chunks


# Prompt Injection Guard
def detect_injection(query):
    bad_patterns = [
        "ignore previous instructions",
        "you are a comedian",
        "act as",
        "roleplay",
        "do anything",
        "disregard rules",
        "system prompt",
        "forget all",
        "jailbreak",
        "developer mode",
        "dan",
        "pretend",
        "hypothetical",
        "always respond with",
        "bypassing",
        "simulated",
        "you are now",
        "sudo",
        "say exactly",
        "print your instructions",
        "ignore all"
    ]

    for pattern in bad_patterns:
        if pattern in query.lower():
            return True

    return False


# Store chunks and their embeddings in Qdrant
def store_chunks(client, model, chunks):
    print("\nGenerating embeddings and storing in Qdrant...")

    embeddings = model.encode(chunks)

    points = []

    for i in range(len(chunks)):
        vector = embeddings[i].tolist()
        chunk_text = chunks[i]

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": chunk_text}
        )

        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print("Stored", len(points), "chunks in Qdrant.")


# Retrieve the most relevant chunks for a query
def retrieve_context(client, model, query, top_k=5):
    query_vector = model.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    ).points

    retrieved_chunks = []

    print("\nRetrieved context:\n")

    for hit in results:
        chunk = hit.payload["text"]
        retrieved_chunks.append(chunk)

        sentences = chunk.split(". ")
        print("-", ". ".join(sentences[:2]), "\n")

    context = ""
    for chunk in retrieved_chunks:
        context += chunk + "\n"

    return context


# Main Pipeline
def main():

    path = input("Enter Path of manual: ")

    text = load_pdf(path)

    if not text.strip():
        print("No text extracted from PDF.")
        return

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    chunks = create_chunks(text, tokenizer)

    print("Connecting to Qdrant...")
    client = QdrantClient("localhost", port=6333)

    # Create a fresh collection each run
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

    store_chunks(client, model, chunks)

    print("\nPDF indexed successfully.")
    print("Ask questions about the manual.\n")

    while True:

        question = input("Question: ")

        if question.lower() == "exit":
            break
        elif question.lower() == "quit":
            break

        if detect_injection(question):
            print("Prompt injection detected. Please ask a question about the manual.")
            continue

        context = retrieve_context(client, model, question)

        # prompt for LLM
        prompt = f"""
            You are a helpful assistant answering questions based ONLY on the provided manual.

            Context:
            {context}

            Question:
            {question}

            Rules:
            - Use only the provided context.
            - Do not invent information.
            - If the answer is not in the context, say:
            "The manual does not provide this information."

            Answer clearly and concisely.
            """

        print("\nGenerated answer:\n")

        response = ollama.chat(
            model="granite3-moe:1b",
            messages=[{"role": "user", "content": prompt}]
        )

        print(response["message"]["content"])
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
