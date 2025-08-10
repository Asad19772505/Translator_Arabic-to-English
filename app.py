import os
import tempfile
import streamlit as st

# ---------- translation + ASR helpers (same as your current file) ----------
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

@st.cache_resource
def load_asr():
    from faster_whisper import WhisperModel
    return WhisperModel("base", device="cpu", compute_type="int8")

def transcribe_audio_file(path, lang_hint=None):
    model = load_asr()
    segments, info = model.transcribe(path, language=lang_hint, beam_size=1)
    text = "".join(seg.text for seg in segments).strip()
    detected = getattr(info, "language", None) or lang_hint or "auto"
    return text, detected

# ---------- UI ----------
st.set_page_config(page_title="Arabic ↔ English Translator (Text + Voice)", layout="wide")
st.title("Arabic ↔ English Translator")

tab_text, tab_voice = st.tabs(["Text", "Voice (Mic / Upload)"])

# ---------------- TEXT TAB (unchanged) ----------------
with tab_text:
    with st.expander("Settings", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            src_text = st.selectbox("Source language (text)", ["auto", "ar", "en", "fr", "ur"], index=0, key="src_text_sel")
        with c2:
            dest_text = st.selectbox("Target language (text)", ["en", "ar", "fr", "ur"], index=0, key="dest_text_sel")
    text_in = st.text_area("Enter text to translate", height=160, placeholder="اكتب النص العربي هنا…", key="text_input_area")
    if st.button("Translate Text", type="primary", key="btn_translate_text"):
        if not text_in.strip():
            st.warning("Please enter some text to translate.")
        else:
            with st.spinner("Translating…"):
                out, engine = translate_text(text_in.strip(), src_lang=src_text, tgt_lang=dest_text)
                st.success(f"Translated with **{engine}**")
                st.text_area("Translation (text tab)", value=out, height=160, key="text_output_area")

# ---------------- VOICE TAB (now with mic button) ----------------
with tab_voice:
    st.write("Record with your browser mic **or** upload an audio file (wav/mp3/m4a).")

    with st.expander("Voice Settings", expanded=True):
        v1, v2, v3 = st.columns(3)
        with v1:
            src_voice = st.selectbox("Source speech language (voice)", ["auto", "ar", "en", "fr", "ur"], index=1, key="src_voice_sel")
        with v2:
            dest_voice = st.selectbox("Target language (voice)", ["en", "ar", "fr", "ur"], index=0, key="dest_voice_sel")
        with v3:
            auto_translate = st.checkbox("Auto-translate after transcription", value=True, key="auto_translate_chk")

    left, right = st.columns(2)

    # ===== A) MIC RECORDER with BUTTON =====
    with left:
        st.subheader("🎙️ Record (Mic Button)")
        wav_path = None

        # Try streamlit-mic-recorder first (has Start/Stop button)
        mic_loaded = False
        try:
            from streamlit_mic_recorder import mic_recorder
            mic_loaded = True
            rec = mic_recorder(
                start_prompt="🎙️ Record",
                stop_prompt="■ Stop",
                just_once=False,
                format="wav",             # returns WAV bytes
                key="mic_btn_widget"
            )
            if rec:
                # rec may be bytes or a dict with "bytes"
                wav_bytes = rec.get("bytes") if isinstance(rec, dict) else rec
                if isinstance(wav_bytes, (bytes, bytearray)):
                    st.audio(wav_bytes, format="audio/wav")
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(wav_bytes)
                        wav_path = tmp.name
        except Exception:
            mic_loaded = False

        # Fallback to older audiorecorder component if available
        if not mic_loaded and wav_path is None:
            try:
                from audiorecorder import audiorecorder
                audio = audiorecorder("Start recording", "Stop recording", key="audiorecorder_widget")
                if audio and len(audio) > 0:
                    wav_bytes = audio.export(format="wav").read()
                    st.audio(wav_bytes, format="audio/wav")
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(wav_bytes)
                        wav_path = tmp.name
            except Exception:
                st.info("Mic recorder not available in this browser. Use the Upload panel on the right.")

        if wav_path and st.button("Transcribe Recording", use_container_width=True, key="btn_transcribe_mic"):
            with st.spinner("Transcribing…"):
                try:
                    hint = None if src_voice == "auto" else src_voice
                    transcript, detected = transcribe_audio_file(wav_path, lang_hint=hint)
                    st.success(f"Transcribed (detected: **{detected}**)")
                    st.text_area("Transcript (mic)", transcript, height=140, key="tx_mic_area")
                    if auto_translate and transcript.strip():
                        out, engine = translate_text(transcript, src_lang=src_voice, tgt_lang=dest_voice)
                        st.success(f"Translated with **{engine}**")
                        st.text_area("Translation (mic)", out, height=140, key="tr_mic_area")
                finally:
                    try:
                        os.unlink(wav_path)
                    except Exception:
                        pass

    # ===== B) FILE UPLOAD =====
    with right:
        st.subheader("📤 Upload")
        up = st.file_uploader("Upload audio (wav/mp3/m4a)", type=["wav", "mp3", "m4a"], accept_multiple_files=False, key="uploader_voice")
        if up is not None:
            st.audio(up, format="audio/wav", start_time=0)
            if st.button("Transcribe Upload", use_container_width=True, key="btn_transcribe_upload"):
                with st.spinner("Transcribing…"):
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=f".{up.name.split('.')[-1]}", delete=False) as tmp:
                            tmp.write(up.read()); tmp_path = tmp.name
                        hint = None if src_voice == "auto" else src_voice
                        transcript, detected = transcribe_audio_file(tmp_path, lang_hint=hint)
                        st.success(f"Transcribed (detected: **{detected}**) from uploaded file")
                        st.text_area("Transcript (upload)", transcript, height=140, key="tx_up_area")
                        if auto_translate and transcript.strip():
                            out, engine = translate_text(transcript, src_lang=src_voice, tgt_lang=dest_voice)
                            st.success(f"Translated with **{engine}**")
                            st.text_area("Translation (upload)", out, height=140, key="tr_up_area")
                    finally:
                        if tmp_path:
                            try: os.unlink(tmp_path)
                            except Exception: pass

st.caption("Note: googletrans/deep-translator use unofficial Google endpoints; rate limits may apply. ASR runs locally via faster-whisper.")
