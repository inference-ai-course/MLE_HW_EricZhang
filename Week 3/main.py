from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import whisper
from transformers import pipeline
from cozyvoice import CozyVoice

app = FastAPI()

asr_model = whisper.load_model("small")

def transcribe_audio(audio_bytes):
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    result = asr_model.transcribe("temp.wav")
    return result["text"]

llm = pipeline("text-generation", model="meta-llama/Llama-3-8B")

conversation_history = []

def generate_response(user_text):
    conversation_history.append({"role": "user", "text": user_text})
    # Construct prompt from history
    prompt = ""
    for turn in conversation_history[-5:]:
        prompt += f"{turn['role']}: {turn['text']}\n"
    outputs = llm(prompt, max_new_tokens=100)
    bot_response = outputs[0]["generated_text"]
    conversation_history.append({"role": "assistant", "text": bot_response})
    return bot_response

tts_engine = CozyVoice()

def synthesize_speech(text, filename="response.wav"):
    tts_engine.generate(text, output_file=filename)
    return filename

@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    user_text = transcribe_audio(audio_bytes)
    print(user_text)
    bot_text = generate_response(user_text)
    # pythonaudio_path = synthesize_speech(bot_text)
    # # TODO: ASR → LLM → TTS
    # return FileResponse("response.wav", media_type="audio/wav")
    audio_path = synthesize_speech(bot_text)
    return FileResponse(audio_path, media_type="audio/wav")