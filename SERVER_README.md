# OpenVoice Server

This server provides voice cloning, tone color conversion, and MeloTTS text-to-speech services via HTTP API, isolating OpenVoice and MeloTTS functionality in its own conda environment.

## Features

- **Speaker Embedding Extraction**: Extract speaker embeddings from audio files
- **Tone Color Conversion**: Convert audio tone color using speaker embeddings
- **Voice Cloning**: Complete voice cloning process with tone color conversion
- **MeloTTS Integration**: Text-to-speech generation with multiple language support
- **Health Monitoring**: Server health and status endpoints

## Server Endpoints

### Health Check
```
GET /health
```
Returns server status and model information including loaded MeloTTS models.

### MeloTTS Speakers
```
GET /melo-speakers?language=EN_NEWEST
```
Returns available speaker IDs for a given MeloTTS language.

**Query Parameters:**
- `language`: MeloTTS language code (default: EN_NEWEST)

### MeloTTS Audio Generation
```
POST /generate-melo-base-audio
```
Generate base audio using MeloTTS with specified speaker and language.

**Request Body:**
```json
{
    "text": "Text to convert to speech",
    "speaker_key": "speaker_name",
    "output_path": "/path/to/output.wav",
    "language": "EN_NEWEST"
}
```

### Speaker Embedding Extraction
```
POST /extract-se
```
Extract speaker embedding from audio file.

**Request Body:**
```json
{
    "audio_path": "/path/to/audio.wav",
    "output_path": "/path/to/save/se.pth",
    "vad": true,
    "parameters": {}
}
```

### Tone Color Conversion
```
POST /tone-convert
```
Convert tone color of audio using speaker embeddings.

**Request Body:**
```json
{
    "audio_src_path": "/path/to/source.wav",
    "src_se_path": "/path/to/source_se.pth",
    "tgt_se_path": "/path/to/target_se.pth",
    "output_path": "/path/to/output.wav",
    "tau": 0.3,
    "message": "default"
}
```

### Voice Cloning
```
POST /voice-clone
```
Complete voice cloning process with tone color conversion.

**Request Body:**
```json
{
    "subtitle_text": "Text to clone",
    "target_speaker_path": "/path/to/target_speaker.wav",
    "output_path": "/path/to/output.wav",
    "tau": 0.3,
    "language": "en"
}
```

### Generic Processing
```
POST /process
```
Generic endpoint that routes to appropriate processing based on request type.

### Shutdown
```
POST /shutdown
```
Graceful server shutdown endpoint.

## Setup

1. **Create conda environment:**
```bash
conda create -n openvoice python=3.9
conda activate openvoice
```

2. **Install dependencies:**
```bash
pip install -r requirements_server.txt
```

3. **Download required NLTK data:**
```python
import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')
```

4. **Start server:**
```bash
python server.py
```

The server will start on port 8009.

## Integration with Main Application

The main application (`srt_to_speech.py`) automatically starts the OpenVoice server when needed and communicates with it via HTTP requests. The server provides:

- **Isolated Environment**: OpenVoice and MeloTTS run in their own conda environment
- **Resource Management**: Automatic server lifecycle management
- **Error Handling**: Graceful error handling and logging
- **Temporary File Management**: Automatic cleanup of temporary files
- **Dynamic Model Loading**: MeloTTS models are loaded on-demand based on language requirements

## MeloTTS Language Support

The server supports multiple MeloTTS languages including:
- EN_NEWEST (default)
- Other language codes as supported by MeloTTS

Models are loaded dynamically when first requested for a specific language.

## Logs

Server logs are stored in:
- `../logs/openvoice_server_stdout.log`
- `../logs/openvoice_server_stderr.log`

## Port Configuration

The server runs on port 8009 by default. This can be modified in `server.py` if needed.

## Dependencies

Key dependencies include:
- FastAPI for HTTP API
- PyTorch for deep learning models
- Librosa for audio processing
- Soundfile for audio I/O
- OpenVoice core libraries
- MeloTTS for text-to-speech generation
- NLTK for natural language processing