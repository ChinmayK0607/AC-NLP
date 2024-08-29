# Audio Processing and Conversational AI API

This project provides an API for processing audio files to transcribe speech, summarize the text, and perform chat-based interactions using advanced machine learning models. The application leverages FastAPI for the backend, using state-of-the-art models like Whisper for transcription, Sentence Transformers for text embedding, FAISS for similarity search, and Groq for text summarization and chatbot responses.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technologies and Libraries Used](#technologies-and-libraries-used)
4. [Project Structure](#project-structure)
5. [Setup and Installation](#setup-and-installation)
6. [Usage](#usage)
7. [Methodology](#methodology)
8. [Models Used](#models-used)

## Project Overview

This API provides endpoints for uploading audio files, processing them to extract and summarize speech, and interacting with a chatbot. It uses machine learning models to transcribe speech from audio files, generate summaries, and respond to user inputs in a conversational context.

## Features

- Upload audio files for speech transcription
- Text summarization from transcribed speech
- Chat-based interaction using context-aware responses
- Audio file format validation (supports WAV, MP3, M4A, FLAC)
- Real-time text processing and response generation

## Technologies and Libraries Used

- **Backend**:
  - FastAPI: For creating the web server and API endpoints
  - Whisper: For speech-to-text transcription
  - Sentence Transformers: For creating text embeddings
  - FAISS: For similarity search in text embeddings
  - Groq: For text summarization and chatbot responses
  - Uvicorn: ASGI server for running the FastAPI application

- **Utilities**:
  - Logging: For application logs and debugging

## Project Structure

```
Audio-Processing-API/
├── main.py
├── index.html
├── requirements.txt
└── README.md
```

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/Audio-Processing-API.git
   cd Audio-Processing-API
   ```

2. **Install the requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the Groq API Key:**
   - Replace the `GROQ_API_KEY` in `main.py` with your Groq API key.

## Usage

1. **Start the API server:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Open a web browser and navigate to** `http://localhost:8000` to use the web interface.

3. **API Endpoints:**
   - `POST /process_audio`: Upload an audio file to process and get the transcribed text and summary.
   - `POST /chat`: Send a user input to the chatbot and receive a context-aware response.

## Methodology

1. **Audio Processing**: The user uploads an audio file through the `/process_audio` endpoint. The audio file is validated for supported formats, saved temporarily, and then transcribed into text using the Whisper model.

2. **Text Embedding and Similarity Search**: The transcribed text is converted into a vector embedding using Sentence Transformers. The embedding is added to a FAISS index for efficient similarity search.

3. **Text Summarization**: The transcribed text is summarized using the Groq API, which generates a concise summary based on the input.

4. **Chat Interaction**: For the `/chat` endpoint, the user input is embedded and matched against the existing context in the FAISS index. Relevant context is used to generate a response using the Groq API.

5. **Context Management**: All conversations and interactions are stored in memory to maintain context across multiple chat exchanges.

## Models Used

1. **Whisper**: An advanced speech-to-text model capable of handling various audio formats and delivering high-accuracy transcription.

2. **Sentence Transformers**: A model for generating dense vector representations (embeddings) of sentences, enabling efficient similarity search.

3. **Groq**: A cloud-based AI model for text summarization and chatbot responses, capable of understanding context and generating coherent replies.

## Additional Notes

- The application is optimized for running on machines with a GPU for faster model inference, though it can also be run on CPU.
- Ensure the Groq API key is valid and has sufficient quota for the intended operations.
- Place `index.html` in the same directory as `main.py` to serve the frontend correctly.

## Example Directory Structure

```plaintext
Audio-Processing-API/
├── main.py
├── index.html
├── requirements.txt
└── README.md
``` 

This structure ensures that all required files are accessible, and the application runs smoothly across different environments.
