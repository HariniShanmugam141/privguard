import sys
import os
import re
import io
import pytesseract
import fitz  # PyMuPDF
import spacy
import cv2
from PIL import Image

TESSERACT_CONFIG = r"--oem 3 --psm 6"

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print("[!] spaCy model not found. Run: python -m spacy download en_core_web_sm")
    raise e


# PDF TEXT EXTRACTION
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF; OCR image-only pages."""
    text_parts = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[!] Could not open PDF {pdf_path}: {e}")
        return ""

    for i, page in enumerate(doc):
        try:
            page_text = page.get_text("text").strip()
        except Exception:
            page_text = ""
        if page_text:
            text_parts.append(page_text)
        else:
            try:
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                ocr_text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
            except Exception as e:
                print(f"[!] OCR error on page {i}: {e}")
                ocr_text = ""
            text_parts.append(ocr_text)
    doc.close()
    return "\n\n".join(text_parts)


# PII DETECTION (ENHANCED)
def detect_pii_entities(text):
    """
    Detect ONLY personally identifiable information (PII),
    keeping all medical/clinical info intact.
    """
    entities = []
    if not text:
        return entities

    # --- Regex patterns for expanded PII coverage ---
    regex_patterns = {
        # Core identifiers
        "NAME": r"(?i)(?<=Name[:\s])([A-Z][a-z]+(?:\s[A-Z]\.)?\s[A-Z][a-z]+)",
        "DOB": r"(?i)(?:DOB[:\s]*)(?:\d{1,2}[-/\s]\d{1,2}[-/\s]\d{2,4}|[A-Z][a-z]+\s\d{1,2},\s\d{4})",
        "GENDER": r"(?i)(?<=Gender[:\s])(Male|Female|Other|Non[-\s]?binary|Unknown)",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}",
        "PHONE": r"(?:\+?\d{1,2}[\s-]?)?(?:\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})",

        # Addresses
        "ADDRESS": r"\d{1,5}\s+[A-Za-z0-9\s]+(Street|St|Road|Rd|Avenue|Ave|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Circle|Way|Place|Trail|Terrace|Parkway|Pkwy|Loop|Highway|Hwy)\b.*",
        "CITY_STATE_ZIP": r"[A-Z][a-z]+,\s?[A-Z]{2}\s?\d{5}(?:-\d{4})?",

        # Insurance identifiers
        "MEMBER_ID": r"(?i)(?<=Member ID[:\s]*)[A-Z0-9-]{6,15}",
        "GROUP_NUMBER": r"(?i)(?<=Group Number[:\s]*)[A-Z0-9-]{3,15}",
        "POLICY_ID": r"(?i)(?<=Policy Effective Date[:\s]*)\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",
        "PAYER_ID": r"(?i)(?<=Payer ID[:\s]*)\d{3,10}",
        "INSURANCE_ID": r"\b\d{6,15}\b(?=.*(Insurance|Insurer|Provider|ID))",

        # Government / institutional IDs
        "DRIVER_LICENSE": r"(?i)(?<=License[:\s]*)[A-Z0-9-]{5,15}",
        "PASSPORT": r"(?i)(?<=Passport[:\s]*)[A-Z0-9-]{6,15}",
        "TAX_ID": r"\b\d{2}-\d{7}\b(?=.*Tax)",
        "NATIONAL_ID": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        "NPI_ID": r"(?i)NPI[:\s]*\d{8,10}\b",
        "MRN": r"(?i)MRN[:\s]*[A-Z0-9-]{6,15}\b",

        # Misc numeric or alphanumeric identifiers
        "ACCOUNT_ID": r"\b[A-Z]{2,3}\d{6,}\b",
        "LONG_ID": r"\b[A-Z0-9]{8,}\b",
        "ZIP": r"\b\d{5}(?:-\d{4})?\b",
        "DATE_NUMERIC": r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
    }

    for label, pattern in regex_patterns.items():
        for match in set(re.findall(pattern, text)):
            if isinstance(match, tuple):
                match = match[0]
            if match.strip():
                entities.append((match.strip(), label))

    # spaCy NER for additional personal names or locations
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "GPE", "LOC"}:
            et = ent.text.strip()
            if et and len(et.replace(" ", "")) > 3:
                entities.append((et, ent.label_))

    # Deduplicate
    seen = set()
    uniq = []
    for t, l in entities:
        key = (t.strip().lower(), l)
        if key not in seen:
            seen.add(key)
            uniq.append((t.strip(), l))
    return uniq


# REDACT TEXT
def redact_text_replacements(text, entities):
    """Return text with all PII replaced by [REDACTED]."""
    redacted = text
    for ent_text, _ in entities:
        if not ent_text:
            continue
        pattern = re.escape(ent_text)
        redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)
    return redacted

# REDACT PDF
def create_redacted_pdf(original_pdf_path, pii_entities, output_path):
    """Redact only detected PII fields from PDF."""
    try:
        doc = fitz.open(original_pdf_path)
    except Exception as e:
        print(f"[!] Could not open PDF: {e}")
        return False

    try:
        doc.set_metadata({})
    except Exception:
        pass

    unique_entities = list({(t, l) for (t, l) in pii_entities})

    for page in doc:
        for ent_text, _ in unique_entities:
            ent_text = ent_text.strip()
            if not ent_text or len(ent_text.replace(" ", "")) < 3:
                continue
            try:
                rects = page.search_for(ent_text)
            except Exception:
                rects = []
            for r in rects:
                try:
                    page.add_redact_annot(r, fill=(0, 0, 0))
                except Exception:
                    pass
        try:
            page.apply_redactions()
        except Exception:
            pass

    try:
        doc.save(output_path)
        doc.close()
        print(f"[+] Saved redacted PDF → {output_path}")
        return True
    except Exception as e:
        print(f"[!] Failed to save redacted PDF: {e}")
        try:
            doc.close()
        except Exception:
            pass
        return False


# IMAGE REDACTION
def preprocess_for_ocr_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th
    except Exception:
        return gray


def redact_image_with_text_and_faces(image_path, output_path):
    """Redact PII from image and blur faces."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] Could not read image: {image_path}")
        return [], 0

    proc = preprocess_for_ocr_image(img)
    ocr_data = pytesseract.image_to_data(proc, output_type=pytesseract.Output.DICT, config=TESSERACT_CONFIG)

    words = [w for w in ocr_data.get("text", []) if w and w.strip()]
    full_text = " ".join(words)
    pii_entities = detect_pii_entities(full_text)

    for i, word in enumerate(ocr_data["text"]):
        if not word.strip():
            continue
        for ent_text, _ in pii_entities:
            if word.lower() in ent_text.lower():
                x, y, w, h = (
                    ocr_data["left"][i],
                    ocr_data["top"][i],
                    ocr_data["width"][i],
                    ocr_data["height"][i],
                )
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
                break

    # Face blur
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img[y:y + h, x:x + w] = cv2.GaussianBlur(img[y:y + h, x:x + w], (99, 99), 30)

    cv2.imwrite(output_path, img)
    return pii_entities, len(faces)

# REPORT GENERATION
def generate_report_per_file(report_entries, output_path="report.txt"):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=== PRIVGUARD AUDIT REPORT ===\n\n")
            for entry in report_entries:
                f.write(f"File: {entry['file']}\n")
                f.write(f"  Faces redacted: {entry.get('faces', 0)}\n")
                f.write(f"  Text PII found: {len(entry.get('pii', []))}\n")
                for (t, l) in entry.get('pii', []):
                    f.write(f"    - {l}: {t}\n")
                f.write("\n")
        print(f"[+] Audit report written to {output_path}")
    except Exception as e:
        print(f"[!] Failed to write report: {e}")


# MAIN
def main(argv):
    if len(argv) < 2:
        print("Usage: python privguard.py <file1> <file2> ...")
        return 1

    report_entries = []

    for filepath in argv[1:]:
        if not os.path.exists(filepath):
            print(f"[!] File not found: {filepath}")
            continue

        basename = os.path.basename(filepath)
        name_noext = os.path.splitext(basename)[0]

        if filepath.lower().endswith(".pdf"):
            print(f"[*] Processing PDF: {filepath}")
            text = extract_text_from_pdf(filepath)
            pii = detect_pii_entities(text)
            redacted_text = redact_text_replacements(text, pii)

            txt_out = f"redacted_{name_noext}.txt"
            with open(txt_out, "w", encoding="utf-8") as f:
                f.write(redacted_text)
            print(f"[+] Saved redacted text -> {txt_out}")

            pdf_out = f"redacted_{name_noext}.pdf"
            create_redacted_pdf(filepath, pii, pdf_out)
            report_entries.append({"file": basename, "pii": pii, "faces": 0})

        elif filepath.lower().endswith((".jpg", ".jpeg", ".png")):
            print(f"[*] Processing Image: {filepath}")
            img_out = f"redacted_{name_noext}{os.path.splitext(filepath)[1]}"
            pii_list, face_count = redact_image_with_text_and_faces(filepath, img_out)
            report_entries.append({"file": basename, "pii": pii_list, "faces": face_count})
            print(f"[+] Saved redacted image -> {img_out} (faces blurred: {face_count})")
        else:
            print(f"[!] Unsupported file type: {filepath}")
            continue

    if report_entries:
        generate_report_per_file(report_entries)
    else:
        print("[*] No files processed — nothing to report.")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
