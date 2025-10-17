# PrivGuard

PrivGuard is a **Python CLI tool** that automatically detects and redacts **Personally Identifiable Information (PII)** from **PDFs** and **images**.
It uses **spaCy**, **Tesseract OCR**, **PyMuPDF**, **OpenCV**, and **Pillow** to identify and hide sensitive text, blur faces, and remove metadata.

---

## Features

* Detects names, emails, phone numbers, dates, and Aadhaar numbers
* Redacts sensitive text in PDFs and applies black bar overlays
* Redacts text and blurs faces in images
* Extracts text from scanned PDFs and images using OCR
* Generates an audit report summarizing detected PII
* Removes embedded metadata from images and PDFs

---

## Tech Stack

| Component          | Purpose                             |
| ------------------ | ----------------------------------- |
| **spaCy**          | NLP for entity recognition          |
| **pytesseract**    | OCR for text extraction             |
| **opencv-python**  | Image processing and face detection |
| **pillow (PIL)**   | Image cleanup and EXIF removal      |
| **PyMuPDF (fitz)** | PDF parsing and redaction           |

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/bencysandra2006/privguard.git
   cd privguard
   ```

2. **(Optional) Create and activate a virtual environment**

   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   # or
   source venv/bin/activate   # macOS/Linux
   ```

3. **Install required libraries**

   ```bash
   pip install spacy pytesseract opencv-python pillow PyMuPDF
   ```

4. **Download spaCy’s English model**

   ```bash
   python -m spacy download en_core_web_sm
   ```

---

## Usage

### Redact a PDF

```bash
python privguard.py samples/sample.pdf
```

### Redact an Image

```bash
python privguard.py samples/photo.png
```

**Output files:**

* `redacted_<filename>.pdf` or `.png` — redacted version
* `report.txt` — summary report

---

## Example Report

```
=== PRIVGUARD AUDIT REPORT ===

File: sample.pdf
  Faces redacted: 0
  Text PII found: 3
    - EMAIL: john.doe@email.com
    - PHONE: 9876543210
    - DATE: 12/09/2023
```

---

## How It Works

1. Extracts text from documents or images using OCR and PyMuPDF
2. Detects PII using spaCy and regular expressions
3. Redacts sensitive text and applies black rectangles or blur masks
4. Detects faces and applies Gaussian blur
5. Generates a detailed audit report

---

## Use Cases

* Redacting personal data from official forms and resumes
* Preparing documents for secure sharing or publication
* Removing identifying details from scanned IDs or images
* Supporting privacy compliance in data pipelines

---

## License

This project is licensed under the **MIT License**.

