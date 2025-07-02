# OpenVoice Server

This server provides voice cloning and tone color conversion services via HTTP API, isolating OpenVoice functionality in its own conda environment.

## Features

- **Speaker Embedding Extraction**: Extract speaker embeddings from audio files
- **Tone Color Conversion**: Convert audio tone color using speaker embeddings
- **Voice Cloning**: Complete voice cloning process with tone color conversion
- **Health Monitoring**: Server health and status endpoints

## Server Endpoints

### Health Check
```
GET /health
```
Returns server status and model information.

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

3. **Start server:**
```bash
python server.py
```

The server will start on port 8009.

## Integration with Main Application

The main application (`srt_to_speech.py`) automatically starts the OpenVoice server when needed and communicates with it via HTTP requests. The server provides:

- **Isolated Environment**: OpenVoice runs in its own conda environment
- **Resource Management**: Automatic server lifecycle management
- **Error Handling**: Graceful error handling and logging
- **Temporary File Management**: Automatic cleanup of temporary files

## Testing

Run the test script to verify server functionality:
```bash
python test_server.py
```

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

## Download
ReadMe.MD + Below Things
import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')