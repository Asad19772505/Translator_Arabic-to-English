# Arabic-to-English Translator (Voice + Text) - Ready to Use Streamlit App

import streamlit as st
import speech_recognition as sr
from googletrans import Translator
import tempfile
import os

# Setup
st.set_page_config(page_title="Arabic to English Translator", layout="centered")
st.title("🗣️ Arabic to English Translator")
translator = Translator()

# Initialize session state for translation output
if 'translated_text' not in st.session_state:
    st.session_state['translated_text'] = ''

# Text Translation
st.header("📄 Translate Arabic Text")
arabic_text = st.text_area("Enter Arabic Text:")
if st.button("Translate Text") and arabic_text:
    translated = translator.translate(arabic_text, src='ar', dest='en')
    st.session_state['translated_text'] = translated.text
    st.success("Translation:")
    st.write(translated.text)

# Voice Translation
st.header("🎤 Translate Arabic Voice")
uploaded_audio = st.file_uploader("Upload an Arabic Audio File (WAV/MP3)", type=['wav', 'mp3'])
if uploaded_audio:
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_audio.read())
        tmp_file_path = tmp_file.name

    with sr.AudioFile(tmp_file_path) as source:
        audio = recognizer.record(source)
        try:
            arabic_speech = recognizer.recognize_google(audio, language='ar-SA')
            st.audio(tmp_file_path)
            st.info(f"Arabic Transcript: {arabic_speech}")
            translated_voice = translator.translate(arabic_speech, src='ar', dest='en')
            st.session_state['translated_text'] = translated_voice.text
            st.success("English Translation:")
            st.write(translated_voice.text)
        except sr.UnknownValueError:
            st.error("Could not understand the audio.")
        except sr.RequestError:
            st.error("Could not request results from the speech recognition service.")

# Download Translation
if st.session_state['translated_text']:
    st.download_button(
        "Download Translated Text",
        data=st.session_state['translated_text'],
        file_name="translation.txt",
        mime="text/plain"
    )
