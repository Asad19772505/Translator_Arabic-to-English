import os
import io
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder
from groq import Groq
import httpx

# -----------------------------
# Setup
# -----------------------------
st.set_page_config(page_title="Arabic ↔ English Speech Translator (Groq)", layout="centered")
st.title("🎙️ Arabic Speech → Text / English Translation")

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

with st.sidebar:
    st.header("Settings")
    timeout_sec = st.slider("Request timeout (seconds)", 30, 6000, 180, step=30)
    st.caption("Increase if you process longer audio files.")

    if not API_KEY:
        st.warning("Add your GROQ_API_KEY to .env or Streamlit Secrets.", icon="⚠️")

# Create Groq client (supports httpx.Timeout)
client = Groq(
    api_key=API_KEY,
    timeout=httpx.Timeout(timeout_sec, read=timeout_sec, write=timeout_sec, connect=30.0),
)

st.write(
    "Record audio in the browser or upload a file. "
    "Choose **Transcribe Arabic** or **Translate Arabic → English**."
)

mode = st.radio(
    "Mode",
    ["Translate Arabic → English", "Transcribe Arabic (Arabic text)"],
    horizontal=True,
)

col1, col2 = st.columns(2, vertical_alignment="center")
with col1:
    st.subheader("🎧 Microphone (browser)")
    st.caption("Click once to start, click again to stop. Auto-stops after silence.")
    audio_bytes = audio_recorder(pause_threshold=2.0)  # simple defaults

with col2:
    st.subheader("📤 Or upload audio")
    uploaded = st.file_uploader(
        "Supported: wav, mp3, m4a, ogg, webm", type=["wav", "mp3", "m4a", "ogg", "webm"]
    )

prompt = st.text_input(
    "Optional context (names, terms, spellings)",
    help="Helps the model with unusual words or spellings.",
)

go = st.button("Process", type="primary", use_container_width=True)

def write_temp_file(data: bytes, suffix: str = ".wav") -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)

def process_with_groq(audio_path: Path, mode: str, prompt: str | None = None) -> str:
    # For translation, pass (filename, bytes) per Groq STT docs
    if mode.startswith("Translate"):
        with open(audio_path, "rb") as f:
            res = client.audio.translations.create(
                file=(audio_path.name, f.read()),
                model="whisper-large-v3",
                response_format="json",
                temperature=0.0,
                # language='en' is optional for translations; endpoint outputs English
                prompt=prompt or None,
            )
        return res.text

    # For transcription (Arabic), pass a file object; supply language hint
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

    # Decide input source
    temp_path = None
    try:
        if audio_bytes:
            temp_path = write_temp_file(audio_bytes, suffix=".wav")
        elif uploaded is not None:
            suffix = "." + uploaded.name.split(".")[-1].lower()
            temp_path = write_temp_file(uploaded.getbuffer(), suffix=suffix)
        else:
            st.warning("Please record or upload an audio file.", icon="ℹ️")
            st.stop()

        with st.spinner("Transcribing/Translating with Groq Whisper…"):
            text = process_with_groq(temp_path, mode, prompt)

        st.success("Done.")
        st.subheader("Result")
        st.text_area("Output", value=text, height=220)

        # Downloads
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

st.markdown(
    """
---  
**Tips**
- For long files, raise the timeout in the sidebar.  
- Keep uploads under 25–100MB per Groq plan; for longer audio, chunk before upload.  
"""
)
