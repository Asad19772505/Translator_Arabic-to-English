import os
import io
import tempfile
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder
from groq import Groq
import httpx

# New imports for document handling
import re
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from PIL import Image

# Optional OCR (requires system tesseract installed)
# On Linux (e.g., Streamlit Cloud), you may need:
#   sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-ara
# For Docker, add those to your image.
try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# -----------------------------
# Setup
# -----------------------------
st.set_page_config(page_title="Arabic ↔ English Translator (Groq)", layout="centered")
st.title("🗣️/📄 Arabic → English Translator (Groq)")

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

with st.sidebar:
    st.header("Settings")
    timeout_sec = st.slider("Request timeout (seconds)", 30, 6000, 180, step=30)
    st.caption("Increase if you process longer inputs.")
    model_name = st.selectbox(
        "Groq model for text translation",
        [
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ],
        index=0,
        help="Used for document/text translation (not audio)."
    )
    polishing = st.checkbox(
        "Polish English tone (formal, concise, CFO-ready)",
        value=True,
        help="Keeps meaning faithful while improving clarity and tone."
    )
    keep_layout = st.checkbox(
        "Preserve light structure (paragraphs & bullet hints)",
        value=True,
        help="Adds mild structure cues; not full formatting."
    )

    if not API_KEY:
        st.warning("Add your GROQ_API_KEY to .env or Streamlit Secrets.", icon="⚠️")

# Create Groq client (supports httpx.Timeout)
client = Groq(
    api_key=API_KEY,
    timeout=httpx.Timeout(timeout_sec, read=timeout_sec, write=timeout_sec, connect=30.0),
)

st.write(
    "Choose **Audio** or **Document**. For audio, you can record or upload. "
    "For documents, upload **PDF, DOCX, or images**. OCR is available for scans."
)

top_mode = st.radio(
    "What do you want to translate?",
    ["Audio (Speech → Text/Translation)", "Document (PDF / Word / Scanned)"],
    horizontal=True,
)

# =========================
# Common helpers
# =========================

def clean_whitespace(s: str) -> str:
    return re.sub(r'\s+\n', '\n', re.sub(r'[ \t]+', ' ', s)).strip()

def chunk_text(text: str, max_chars: int = 6000) -> List[str]:
    """Chunk text by paragraphs without breaking sentences too harshly."""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks

def translate_with_groq(text: str, model: str, polishing: bool, keep_layout: bool) -> str:
    """Use Groq chat.completions to translate Arabic → English, chunk-aware call."""
    if not text.strip():
        return ""

    style_instructions = []
    style_instructions.append("Translate from Arabic to English faithfully.")
    if polishing:
        style_instructions.append("Polish the English for clarity, brevity, and professional tone (CFO-ready).")
    if keep_layout:
        style_instructions.append("Preserve basic paragraph structure and convert obvious lists to bullets if appropriate (no heavy formatting).")
    style_instructions.append("Do NOT summarize or omit content. Keep all data, numbers, names, and legal wording intact.")
    style_instructions.append("Do NOT include any translator notes; output English only.")

    system_prompt = (
        "You are a professional Arabic→English translator for legal/financial/business documents. "
        + " ".join(style_instructions)
    )

    chunks = chunk_text(text, max_chars=6000)
    outputs = []
    progress = st.progress(0, text="Translating…")
    for i, ch in enumerate(chunks, start=1):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ch}
            ],
            temperature=0.2,
        )
        outputs.append(resp.choices[0].message.content.strip())
        progress.progress(i/len(chunks), text=f"Translating chunk {i}/{len(chunks)}")
    progress.empty()
    return "\n\n".join(outputs).strip()

def write_temp_file(data: bytes, suffix: str = ".wav") -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)

# =========================
# AUDIO PATH (Original)
# =========================
if top_mode == "Audio (Speech → Text/Translation)":
    mode = st.radio(
        "Audio Mode",
        ["Translate Arabic → English", "Transcribe Arabic (Arabic text)"],
        horizontal=True,
    )

    col1, col2 = st.columns(2, vertical_alignment="center")
    with col1:
        st.subheader("🎧 Microphone (browser)")
        st.caption("Click once to start, click again to stop. Auto-stops after silence.")
        audio_bytes = audio_recorder(pause_threshold=2.0)

    with col2:
        st.subheader("📤 Or upload audio")
        uploaded_audio = st.file_uploader(
            "Supported: wav, mp3, m4a, ogg, webm", type=["wav", "mp3", "m4a", "ogg", "webm"]
        )

    prompt = st.text_input(
        "Optional context (names, terms, spellings)",
        help="Helps the model with unusual words or spellings.",
    )

    go = st.button("Process Audio", type="primary", use_container_width=True)

    def process_with_groq_audio(audio_path: Path, mode: str, prompt: str | None = None) -> str:
        # Translation endpoint (audio → English)
        if mode.startswith("Translate"):
            with open(audio_path, "rb") as f:
                res = client.audio.translations.create(
                    file=(audio_path.name, f.read()),
                    model="whisper-large-v3",
                    response_format="json",
                    temperature=0.0,
                    prompt=prompt or None,
                )
            return res.text

        # Transcription endpoint (audio → Arabic text)
        with open(audio_path, "rb") as f:
            res = client.audio.transcriptions.create(
                file=f,
                model="whisper-large-v3-turbo",
                response_format="json",
                temperature=0.0,
                language="ar",
                prompt=prompt or None,
            )
        return res.text

    if go:
        if not API_KEY:
            st.error("GROQ_API_KEY not found. Set it in .env or Streamlit Secrets.", icon="🚫")
            st.stop()

        temp_path = None
        try:
            if audio_bytes:
                temp_path = write_temp_file(audio_bytes, suffix=".wav")
            elif uploaded_audio is not None:
                suffix = "." + uploaded_audio.name.split(".")[-1].lower()
                temp_path = write_temp_file(uploaded_audio.getbuffer(), suffix=suffix)
            else:
                st.warning("Please record or upload an audio file.", icon="ℹ️")
                st.stop()

            with st.spinner("Transcribing/Translating with Groq Whisper…"):
                text = process_with_groq_audio(temp_path, mode, prompt)

            st.success("Done.")
            st.subheader("Result")
            st.text_area("Output", value=text, height=220)

            st.download_button(
                "⬇️ Download as .txt",
                data=text.encode("utf-8"),
                file_name="transcript.txt",
                mime="text/plain",
                use_container_width=True,
            )

        except httpx.ReadTimeout:
            st.error("Request timed out. Increase the timeout slider and try again.")
        except Exception as e:
            st.exception(e)
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

# =========================
# DOCUMENT PATH (NEW)
# =========================
else:
    st.subheader("📄 Upload Document (Arabic → English)")
    st.caption("Supported: PDF, DOCX, Images (PNG/JPG/JPEG). Use OCR for scans.")
    file = st.file_uploader(
        "Upload a file", type=["pdf", "docx", "png", "jpg", "jpeg", "tif", "tiff", "bmp"]
    )

    use_ocr = st.toggle(
        "Enable OCR (for scanned PDFs/images)",
        value=False,
        help="Uses Tesseract if installed. Not needed for text-based PDFs."
    )

    lang_hint = st.text_input(
        "Optional terms/names to preserve (comma-separated)",
        help="e.g., company or people names, brand terms, product codes.",
    )

    go_doc = st.button("Translate Document", type="primary", use_container_width=True)

    # -------- Document extractors --------
    def extract_text_from_pdf(stream: bytes, use_ocr: bool) -> Tuple[str, List[Image.Image]]:
        """Return (text, page_images_for_ocr_if_any)."""
        text_parts: List[str] = []
        ocr_images: List[Image.Image] = []

        with fitz.open(stream=stream, filetype="pdf") as doc:
            for page in doc:
                txt = page.get_text("text") or ""
                txt_clean = clean_whitespace(txt)
                if txt_clean:
                    text_parts.append(txt_clean)
                elif use_ocr:
                    # Render page to raster and OCR
                    pix = page.get_pixmap(dpi=200, alpha=False)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_images.append(img)
                # else: empty page gets ignored if no OCR
        joined = "\n\n".join(text_parts).strip()
        return joined, ocr_images

    def extract_text_from_docx(stream: bytes) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(stream)
            tmp.flush()
            tmp_path = tmp.name

        try:
            doc = DocxDocument(tmp_path)
            paras = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
            return "\n".join(paras)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def ocr_images_to_text(images: List[Image.Image], lang: str = "ara") -> str:
        if not OCR_AVAILABLE:
            st.warning("OCR libraries not available. Install pytesseract & tesseract to enable OCR.", icon="⚠️")
            return ""
        texts = []
        for i, im in enumerate(images, start=1):
            st.write(f"OCR page {i}…")
            txt = pytesseract.image_to_string(im, lang=lang)
            texts.append(clean_whitespace(txt))
        return "\n\n".join([t for t in texts if t])

    def extract_text_from_image(stream: bytes, use_ocr: bool) -> str:
        img = Image.open(io.BytesIO(stream)).convert("RGB")
        if use_ocr:
            return ocr_images_to_text([img], lang="ara")
        else:
            st.info("Image provided. Enable OCR to read Arabic text from image scans.")
            return ""

    # -------- Process Document --------
    if go_doc:
        if not API_KEY:
            st.error("GROQ_API_KEY not found. Set it in .env or Streamlit Secrets.", icon="🚫")
            st.stop()
        if not file:
            st.warning("Please upload a document.", icon="ℹ️")
            st.stop()

        name = file.name.lower()
        data = file.getbuffer()

        try:
            with st.spinner("Extracting text…"):
                arabic_text = ""
                if name.endswith(".pdf"):
                    pdf_text, imgs_for_ocr = extract_text_from_pdf(data, use_ocr=use_ocr)
                    arabic_text = pdf_text
                    if not arabic_text and use_ocr and imgs_for_ocr:
                        arabic_text = ocr_images_to_text(imgs_for_ocr, lang="ara")
                elif name.endswith(".docx"):
                    arabic_text = extract_text_from_docx(data)
                elif any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]):
                    arabic_text = extract_text_from_image(data, use_ocr=use_ocr)
                else:
                    st.error("Unsupported file type.")
                    st.stop()

                arabic_text = clean_whitespace(arabic_text)

            if not arabic_text:
                st.error("No text found. If this is a scan or image-based PDF, enable OCR.")
                st.stop()

            # Optionally inject glossary hints
            hints = ""
            if lang_hint.strip():
                hints = f"\n\nGlossary/Hints (keep spellings as-is): {lang_hint.strip()}"

            with st.spinner("Translating with Groq…"):
                english_text = translate_with_groq(arabic_text + hints, model=model_name, polishing=polishing, keep_layout=keep_layout)

            st.success("Translation complete.")
            st.subheader("English Translation")
            st.text_area("Output", value=english_text, height=360)

            # Downloads
            st.download_button(
                "⬇️ Download translation (.txt)",
                data=english_text.encode("utf-8"),
                file_name=f"{Path(file.name).stem}_EN.txt",
                mime="text/plain",
                use_container_width=True,
            )

            # Create a DOCX with simple source/target pairing
            if st.checkbox("Generate bilingual DOCX (source + translation)"):
                docx_buf = io.BytesIO()
                doc = DocxDocument()
                doc.add_heading(f"Translation of: {file.name}", level=1)
                doc.add_paragraph("Source Language: Arabic")
                doc.add_paragraph("Target Language: English")
                doc.add_paragraph("-" * 30)
                doc.add_heading("English Translation", level=2)
                for para in english_text.split("\n"):
                    doc.add_paragraph(para)
                doc.save(docx_buf)
                st.download_button(
                    "⬇️ Download bilingual DOCX",
                    data=docx_buf.getvalue(),
                    file_name=f"{Path(file.name).stem}_EN.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )

        except httpx.ReadTimeout:
            st.error("Request timed out. Increase the timeout slider and try again.")
        except Exception as e:
            st.exception(e)

# -----------------------------
# Tips
# -----------------------------
st.markdown(
    """
---
**Tips**
- For long audio or documents, raise the timeout in the sidebar.  
- **PDFs**: Text-based PDFs extract directly. For scans, enable **OCR**.  
- **OCR setup**: Install system package `tesseract-ocr` (and `tesseract-ocr-ara`) and Python libs `pytesseract`, `Pillow`.  
- **Token limits**: The app automatically chunks large texts before translation.  
"""
)

