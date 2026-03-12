from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_pdf(path):
    reader = PdfReader(path)

    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t

    text = text.replace("\n", " ")
    text = " ".join(text.split())

    return text


def create_chunks(text, chunk_size=120, overlap=30):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks


# Load document
text = load_pdf(input("Enter Path of manual: "))

if not text.strip():
    print("No text extracted from PDF.")
    exit()

chunks = create_chunks(text)

vectorizer = TfidfVectorizer(stop_words="english")
chunk_vectors = vectorizer.fit_transform(chunks)
sentences = []
print("PDF loaded. Ask questions.\n")

while True:
    question = input("Question: ")

    if question.lower() in ["exit", "quit"]:
        break

    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, chunk_vectors).flatten()

    top_indices = scores.argsort()[-3:][::-1]

    print("\nMost relevant passages:\n")

    results = [i for i in top_indices if scores[i] >= 0.05]

    if not results:
        print("No relevant passages found.\n")
        continue

    for i in results:
        sentences = chunks[i].split(". ")
        print("-", ". ".join(sentences[:2]), "\n")
        print(f"[Chunk {i} | score {scores[i]:.2f}]")