import os
import streamlit as st
from streamlit_ace import st_ace
from groq import Groq
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import assemblyai as aai
# import whisper
import time
from audio_recorder_streamlit import audio_recorder

# Set API keys from Streamlit secrets
groq_client = Groq(api_key="YOUR_API_KEY")
# set_api_key(st.secrets["ELEVENLABS_API_KEY"])
client = ElevenLabs(api_key = "YOUR_API_KEY")

aai.settings.api_key = "YOUR_API_KEY"

# Initialize transcriber
transcriber = aai.Transcriber()


def generate_questions(job_role, tech_stack, experience):
    prompt = f"""You are a technical recruiter at a major MNC conducting interviews for {job_role} position. 
    The candidate has {experience} years of experience with technology stack: {tech_stack}.
    Generate 5 recent technical interview questions that would be asked in actual MNC interviews.
    Questions should be specific to mentioned technologies and experience level.
    Format questions as a Python list of strings. Give only the question and answer in python list and nothing else"""

    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )
    
    try:
        questions = eval(chat_completion.choices[0].message.content)
        return questions[:5]
    except:
        return ["Error generating questions. Please try again."]

def evaluate_answer(question, answer, job_role, tech_stack, experience):
    prompt = f"""As an experienced {job_role} technical interviewer at an MNC evaluating a candidate with {experience} years experience in {tech_stack}:
    Interview Question: {question}
    Candidate Answer: {answer}
    
    Provide:
    1. Brief evaluation of answer quality (technical accuracy, relevance, structure)
    2. Specific improvements needed
    3. Sample ideal answer (concise)
    Format response under 200 words."""

    evaluation = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.5
    )
    return evaluation.choices[0].message.content

def text_to_speech(text):
    audio = client.text_to_speech.convert(text=text, voice_id="pNInz6obpgDQGcFmaJgB", model_id="eleven_multilingual_v2")
    play(audio)

def main():
    st.title("AI Mock Interview Platform")
    
    # Sidebar inputs
    with st.sidebar:
        job_role = st.text_input("Job Role: ",placeholder="Ex. Full Stack Developer")
        tech_stack = st.text_input("Tech Stack",placeholder="Ex. React, Angular, Node.Js, MySQL etc")
        experience = st.slider("Years of Experience", 0, 20, 0)
        
        if st.button("Start Interview"):
            st.session_state.questions = generate_questions(job_role, tech_stack, experience)
            st.session_state.current_question = 0
            st.session_state.feedback = []
            st.session_state.answers = []
            

    # Main interview area
    if "questions" in st.session_state:
        if st.session_state.current_question < len(st.session_state.questions):
            current_q = st.session_state.questions[st.session_state.current_question]
            
            st.subheader(f"Question {st.session_state.current_question + 1}/5")
            st.write(current_q)
            
            # Text-to-speech for question
            if st.button("🔊 Play Question"):
                text_to_speech(current_q)
            
            # Audio recording and processing
            st.write("## Record Your Answer")
            audio_bytes = audio_recorder(
                text="Click to record",
                recording_color="#e8b62c",
                neutral_color="#6aa36f",
                icon_name="microphone",
                icon_size="2x",
                pause_threshold=3.0
            )

            
            if audio_bytes:
                with st.spinner("transcribing your answer..."):
                    # Save and transcribe audio
                    with open("temp_audio.wav", "wb") as f:
                        f.write(audio_bytes)
                    
                    transcript = transcriber.transcribe("temp_audio.wav")
                    if transcript.status == aai.TranscriptStatus.error:
                        st.error("Transcription failed")
                    else:
                        answer = transcript.text
                        st.write(answer)
                        st.session_state.answers.append(answer)
                    # answer = transcript["text"]
                    # st.write(answer)
                    # st.session_state.answers.append(answer)
            
            if st.button("Evaluate"):
                with st.spinner("Analyzing your answer..."):
                    # Get evaluation
                    evaluation = evaluate_answer(current_q, answer, job_role, 
                                              tech_stack, experience)
                    st.session_state.feedback.append(evaluation)
                    
                    # Show feedback
                    st.subheader("Feedback")
                    st.write(evaluation)
                    
                    # Convert feedback to speech
                    text_to_speech(evaluation)
                    
                    # Move to next question
                    st.session_state.current_question += 1
                    st.rerun()
        else:
            st.success("Interview Completed!")
            st.write("## Final Feedback Summary")
            for i, feedback in enumerate(st.session_state.feedback):
                st.write(f"### Question {i+1}")
                st.write(feedback)

if __name__ == "__main__":
    main()