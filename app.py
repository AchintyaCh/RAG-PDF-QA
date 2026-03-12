from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

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


# Chunking
def create_chunks(text, chunk_size=300, overlap=80):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# Main Pipeline
def main():

    path = input("Enter Path of manual: ")

    text = load_pdf(path)

    if not text.strip():
        print("No text extracted from PDF.")
        return

    chunks = create_chunks(text)

    print("\nCreating embeddings...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    chunk_embeddings = model.encode(chunks)

    dimension = chunk_embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(chunk_embeddings))

    print("\nPDF indexed successfully.")
    print("Ask questions about the manual.\n")

    while True:

        question = input("Question: ")

        if question.lower() in ["exit", "quit"]:
            break

        # embed query
        q_embedding = model.encode([question])

        # retrieve
        distances, indices = index.search(np.array(q_embedding), k=5)

        retrieved_chunks = []

        print("\nRetrieved context:\n")

        for i in indices[0]:

            chunk = chunks[i]
            retrieved_chunks.append(chunk)

            sentences = chunk.split(". ")

            print("-", ". ".join(sentences[:2]), "\n")

        context = "\n".join(retrieved_chunks)

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