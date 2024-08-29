import os
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
import asyncio
import logging
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow CORS (if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold models and other configurations
whisper_model = None
sentence_model = None
faiss_index = None
context_data = {"transcribed_text": "", "conversation_history": []}

# Groq API configuration
GROQ_API_KEY = "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"
groq_client = Groq(api_key=GROQ_API_KEY)

class SummarizeRequest(BaseModel):
    texts: list[str]

class ChatRequest(BaseModel):
    user_input: str

# Corrected load_model function
def load_model():
    global whisper_model, sentence_model, faiss_index
    if whisper_model is None:
        logger.info("Loading Whisper model...")
        whisper_model = whisper.load_model("base")
        logger.info("Whisper model loaded successfully")
    
    if sentence_model is None:
        logger.info("Loading Sentence Transformer model...")
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Sentence Transformer model loaded successfully")
    
    if faiss_index is None:
        logger.info("Initializing FAISS index...")
        faiss_index = faiss.IndexFlatL2(384)  # 384 is the dimension of the sentence embeddings
        logger.info("FAISS index initialized successfully")

async def convert_speech_to_text(audio_path: str) -> str:
    """Convert speech to text using Whisper."""
    logger.info("Converting speech to text with Whisper...")
    result = whisper_model.transcribe(audio_path)
    return result['text']

async def summarize_text(text: str) -> str:
    """Summarize the given text using Groq."""
    try:
        logger.info("Summarizing text using Groq...")
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Please provide a concise summary of the following text:\n\n{text}",
                }
            ],
            model="llama3-8b-8192",
            max_tokens=150,
            temperature=0.5,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Event that runs on startup to load models."""
    load_model()

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the index.html file."""
    with open("index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    """Endpoint to process uploaded audio file."""
    if not file.filename.lower().endswith(('wav', 'mp3', 'm4a', 'flac')):
        logger.warning("Invalid file type uploaded")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")

    audio_content = await file.read()
    audio_path = "temp_audio.wav"

    # Save the audio file temporarily
    with open(audio_path, "wb") as f:
        f.write(audio_content)

    try:
        # Convert audio to text using Whisper
        text = await convert_speech_to_text(audio_path)

        # Create embedding for the transcribed text
        embedding = sentence_model.encode([text])[0]

        # Add the embedding to the FAISS index
        faiss_index.add(embedding.reshape(1, -1))

        # Summarize the transcribed text using Groq
        summary = await summarize_text(text)

        # Store context in memory
        context_data["transcribed_text"] = text
        context_data["conversation_history"] = [{"role": "system", "content": summary}]

        return JSONResponse(content={"text": text, "summary": summary})

    except Exception as e:
        logger.error(f"Error during audio processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up the temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

@app.post("/chat")
async def chat_with_model(request: ChatRequest):
    user_input = request.user_input
    context_data["conversation_history"].append({"role": "user", "content": user_input})

    try:
        # Encode the user input
        query_embedding = sentence_model.encode([user_input])[0]

        # Perform similarity search
        k = 1  # Number of similar contexts to retrieve
        distances, indices = faiss_index.search(query_embedding.reshape(1, -1), k)

        # Retrieve the most similar context
        similar_context = context_data["transcribed_text"] if indices[0][0] < faiss_index.ntotal else ""

        # Prepare the messages for the Groq API
        messages = context_data["conversation_history"] + [
            {"role": "system", "content": f"Relevant context: {similar_context}"}
        ]

        # Send the conversation history and relevant context to Groq API
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
            max_tokens=150,
            temperature=0.5,
        )

        bot_response = chat_completion.choices[0].message.content
        context_data["conversation_history"].append({"role": "assistant", "content": bot_response})

        return JSONResponse(content={"response": bot_response})

    except Exception as e:
        logger.error(f"Error during chat processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
