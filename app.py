import os
import tempfile
import streamlit as st

# =============== Translation core (same as before, with fallback) ===============
def translate_text(text, src_lang="auto", tgt_lang="en"):
    try:
        from googletrans import Translator
        translator = Translator(service_urls=["translate.googleapis.com", "translate.google.com"])
        result = translator.translate(text, src=src_lang, dest=tgt_lang)
        return result.text, "googletrans"
    except Exception as e_google:
        try:
            from deep_translator import GoogleTranslator
            result = GoogleTranslator(source=src_lang, target=tgt_lang).translate(text)
            return result, "deep-translator"
        except Exception as e_deep:
            raise RuntimeError(
                f"Both translators failed.\n"
                f"googletrans error: {e_google}\n"
                f"deep-translator error: {e_deep}"
            )

# =============== ASR (speech-to-text) using faster-whisper on CPU ===============
@st.cache_resource
def load_asr():
    # 'base' is a good CPU default; you can try 'small' if your instance is stronger
    from faster_whisper import WhisperModel
    # int8 is light for CPU; change to "float32" if you prefer accuracy over speed
    return WhisperModel("base", device="cpu", compute_type="int8")

def transcribe_audio_file(path, lang_hint=None):
    """
    Transcribe an audio file using faster-whisper.
    lang_hint: None (auto-detect) or language code like 'ar', 'en'
    """
    model = load_asr()
    segments, info = model.transcribe(path, language=lang_hint, beam_size=1)
    text = "".join(seg.text for seg in segments).strip()
    detected = info.language if hasattr(info, "language") else lang_hint or "auto"
    return text, detected

# =============== UI =================
st.set_page_config(page_title="Arabic ↔ English Translator (Text + Voice)", layout="wide")
st.title("Arabic ↔ English Translator")

tabs = st.tabs(["Text", "Voice (Mic / Upload)"])

# ---------------- TEXT TAB ----------------
with tabs[0]:
    with st.expander("Settings", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            src_text = st.selectbox("Source language", ["auto", "ar", "en", "fr", "ur"], index=0)
        with c2:
            dest_text = st.selectbox("Target language", ["en", "ar", "fr", "ur"], index=0)

    text_in = st.text_area("Enter text to translate", height=160, placeholder="اكتب النص العربي هنا…")
    if st.button("Translate Text", type="primary"):
        if not text_in.strip():
            st.warning("Please enter some text to translate.")
        else:
            with st.spinner("Translating…"):
                try:
                    out, engine = translate_text(text_in.strip(), src_lang=src_text, tgt_lang=dest_text)
                    st.success(f"Translated with **{engine}**")
                    st.text_area("Translation", value=out, height=160)
                except Exception as e:
                    st.error(f"Translation failed.\n\n{e}")

# ---------------- VOICE TAB ----------------
with tabs[1]:
    st.write("Record with your browser mic **or** upload an audio file (wav/mp3/m4a).")
    left, right = st.columns([1, 1])

    # Settings
    with st.expander("Voice Settings", expanded=True):
        v1, v2, v3 = st.columns(3)
        with v1:
            src_voice = st.selectbox("Source speech language", ["auto", "ar", "en", "fr", "ur"], index=1)
        with v2:
            dest_voice = st.selectbox("Target language", ["en", "ar", "fr", "ur"], index=0)
        with v3:
            auto_translate = st.checkbox("Auto-translate after transcription", value=True)

    # ---- Option A: Browser microphone (streamlit-audiorecorder) ----
    with left:
        st.subheader("🎙️ Record")
        mic_ok = True
        audio_bytes = None
        try:
            from audiorecorder import audiorecorder  # component
            audio = audiorecorder("Start recording", "Stop recording")
            # The component returns a pydub.AudioSegment when audio is recorded.
            if audio and len(audio) > 0:
                st.audio(audio.export(format="wav").read(), format="audio/wav")
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    audio.export(tmp.name, format="wav")
                    audio_bytes = tmp.name
        except Exception as e:
            mic_ok = False
            st.info(
                "Mic recorder component unavailable in this environment.\n"
                "You can still upload an audio file on the right."
            )

        if audio_bytes and st.button("Transcribe Recording", use_container_width=True):
            with st.spinner("Transcribing…"):
                try:
                    hint = None if src_voice == "auto" else src_voice
                    transcript, detected = transcribe_audio_file(audio_bytes, lang_hint=hint)
                    st.success(f"Transcribed (detected: **{detected}**)")
                    st.text_area("Transcript", transcript, height=140, key="tx_mic")

                    if auto_translate and transcript.strip():
                        out, engine = translate_text(transcript, src_lang=src_voice, tgt_lang=dest_voice)
                        st.success(f"Translated with **{engine}**")
                        st.text_area("Translation", out, height=140, key="tr_mic")
                except Exception as e:
                    st.error(f"Transcription failed.\n\n{e}")
                finally:
                    try:
                        os.unlink(audio_bytes)
                    except Exception:
                        pass

    # ---- Option B: File upload ----
    with right:
        st.subheader("📤 Upload")
        up = st.file_uploader("Upload audio (wav/mp3/m4a)", type=["wav", "mp3", "m4a"], accept_multiple_files=False)
        if up is not None:
            st.audio(up, format="audio/wav")
            if st.button("Transcribe Upload", use_container_width=True):
                with st.spinner("Transcribing…"):
                    try:
                        with tempfile.NamedTemporaryFile(suffix=f".{up.name.split('.')[-1]}", delete=False) as tmp:
                            tmp.write(up.read())
                            tmp_path = tmp.name
                        hint = None if src_voice == "auto" else src_voice
                        transcript, detected = transcribe_audio_file(tmp_path, lang_hint=hint)
                        st.success(f"Transcribed (detected: **{detected}**) from uploaded file")
                        st.text_area("Transcript", transcript, height=140, key="tx_up")

                        if auto_translate and transcript.strip():
                            out, engine = translate_text(transcript, src_lang=src_voice, tgt_lang=dest_voice)
                            st.success(f"Translated with **{engine}**")
                            st.text_area("Translation", out, height=140, key="tr_up")
                    except Exception as e:
                        st.error(f"Transcription failed.\n\n{e}")
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

st.caption("Note: googletrans/deep-translator use unofficial Google endpoints; rate limits may apply. ASR runs locally via faster-whisper.")
