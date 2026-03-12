# RAG-PDF-QA

A simple Retrieval-Augmented Question Answering (RAG-style) system for PDF manuals.

This project allows a user to load a PDF document (for example: car manuals, appliance manuals, etc.) and ask natural language questions about the content. The system retrieves the most relevant passages from the document using TF-IDF vectorization and cosine similarity.

This implementation focuses on the **retrieval component of a RAG pipeline**, without using any Large Language Models.

---

## Features

* Extracts text from PDF manuals
* Cleans and preprocesses extracted text
* Splits documents into overlapping chunks
* Converts chunks into TF-IDF vectors
* Uses cosine similarity to retrieve relevant passages
* Interactive CLI for asking questions

---

## Architecture

```
PDF Document
      │
      ▼
Text Extraction (PyPDF2)
      │
      ▼
Text Cleaning
      │
      ▼
Chunking (120 words with overlap)
      │
      ▼
TF-IDF Vectorization
      │
      ▼
User Question
      │
      ▼
Cosine Similarity Search
      │
      ▼
Top Relevant Passages Returned
```

---

## Installation

Clone the repository:

```
git clone https://github.com/AchintyaCh/RAG-PDF-QA.git
cd RAG-PDF-QA
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Usage

Run the application:

```
python app.py
```

Enter the path of the PDF manual when prompted:

```
Enter Path of manual: C:\manuals\washing_machine_manual.pdf
```

Ask questions about the document:

```
Question: How to install the washing machine?
Question: Safety precautions?
Question: Where to put detergent?
```

Exit the program with:

```
exit
```

---

## Example Output

```
Most relevant passages:

- Install the washing machine on a solid level floor to avoid vibration.
- Do not install the washing machine in a humid place such as a bathroom.

[Chunk 17 | score 0.21]
```

---

## Technologies Used

* Python
* PyPDF2
* Scikit-learn
* TF-IDF Vectorization
* Cosine Similarity

---

## Project Structure

```
RAG-PDF-QA
│
├── app.py              # Main CLI application
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── sample_pdfs/        # Example manuals (optional)
```

---

## Example CLI Demo

```
$ python app.py
Enter Path of manual: hitachimanual.pdf

PDF loaded. Ask questions.

Question: Safety precautions?

Most relevant passages:
- Do not dismantle, repair or modify the washing machine. This could cause malfunction, fire, electric shock or injury.

[Chunk 12 | score 0.21]
```

---

