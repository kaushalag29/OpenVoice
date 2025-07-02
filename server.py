import os
import sys
import torch
import torchaudio
import soundfile as sf
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import tempfile
import shutil

# Add OpenVoice to path
sys.path.append(os.path.dirname(__file__))

from openvoice.api import ToneColorConverter
from openvoice import se_extractor
from melo.api import TTS as MELOTTS

app = FastAPI(title="OpenVoice API Server")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Detect device (Mac with M1/M2/M3/M4)
# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
logger.info(f"Using device: {device}")

# Load OpenVoice models at startup
tone_color_converter = None
melo_models: Dict[str, MELOTTS] = {}  # Dictionary to hold language-specific models
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
ckpt_converter = os.path.join(SERVER_DIR, 'checkpoints_v2', 'converter')

def load_melo_model(language: str = 'EN_NEWEST') -> Optional[MELOTTS]:
    """Loads a MeloTTS model for a specific language if not already loaded."""
    global melo_models
    if language not in melo_models:
        logger.info(f"--- Loading MeloTTS Model for language: {language} ---")
        try:
            model = MELOTTS(language=language, device=device)
            melo_models[language] = model
            logger.info(f"--- MeloTTS Model for {language} Loaded Successfully ---")
        except Exception:
            logger.error(f"--- FATAL: Error loading MeloTTS model for {language} ---", exc_info=True)
            return None
    return melo_models.get(language)

def load_models():
    """Load core OpenVoice models"""
    global tone_color_converter
    try:
        logger.info("--- Starting Model Loading ---")
        logger.info(f"Current Working Directory: {os.getcwd()}")
        logger.info(f"SERVER_DIR variable: {SERVER_DIR}")
        logger.info(f"Calculated checkpoint converter path: {ckpt_converter}")
        
        # Check if checkpoint files exist
        config_path = os.path.join(ckpt_converter, 'config.json')
        checkpoint_path = os.path.join(ckpt_converter, 'checkpoint.pth')
        
        logger.info(f"Attempting to load config from: {config_path}")
        if not os.path.exists(config_path):
            logger.error(f"Config file NOT FOUND at: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
        logger.info("Config file found.")
        
        logger.info(f"Attempting to load checkpoint from: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file NOT FOUND at: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        logger.info("Checkpoint file found.")
        
        # Initialize tone color converter
        logger.info(f"Initializing ToneColorConverter with device: {device}")
        tone_color_converter = ToneColorConverter(config_path, device=device)
        logger.info("ToneColorConverter initialized. Loading checkpoint...")
        tone_color_converter.load_ckpt(checkpoint_path)
        
        logger.info("--- OpenVoice Models Loaded Successfully ---")
        return True
    except Exception as e:
        logger.error(f"--- FATAL: Error during model loading ---", exc_info=True)
        return False

# Load models at startup
logger.info("Attempting to load all models...")
models_loaded = load_models()
if not models_loaded:
    logger.error("Failed to load OpenVoice models. Server may not function properly.")
    logger.error("This is a critical error - the server will not be able to process requests.")
else:
    logger.info("All models loaded successfully!")

# Define request models
class MeloTTSRequest(BaseModel):
    text: str
    speaker_key: str
    output_path: str
    language: str = 'EN_NEWEST'

class VoiceCloneRequest(BaseModel):
    """Request for voice cloning with tone color conversion"""
    subtitle_text: str
    target_speaker_path: str
    output_path: str
    tau: float = 0.3
    language: str = "en"
    parameters: Dict[str, Any] = {}

class SExtractionRequest(BaseModel):
    """Request for speaker embedding extraction"""
    audio_path: str
    output_path: Optional[str] = None
    vad: bool = True
    parameters: Dict[str, Any] = {}

class ToneConversionRequest(BaseModel):
    """Request for tone color conversion"""
    audio_src_path: str
    src_se_path: str
    tgt_se_path: str
    output_path: str
    tau: float = 0.3
    message: str = "default"
    parameters: Dict[str, Any] = {}

@app.get("/melo-speakers")
async def get_melo_speakers(language: str = 'EN_NEWEST'):
    """Returns the available MeloTTS speaker IDs for a given language"""
    melo_model = load_melo_model(language)
    if melo_model is None:
        raise HTTPException(status_code=500, detail=f"MeloTTS model for language '{language}' could not be loaded.")
    return melo_model.hps.data.spk2id

@app.post("/generate-melo-base-audio")
async def generate_melo_base_audio(request: MeloTTSRequest):
    """Generate base audio using MeloTTS"""
    try:
        melo_model = load_melo_model(request.language)
        if melo_model is None:
            raise HTTPException(status_code=500, detail=f"MeloTTS model for language '{request.language}' could not be loaded.")
            
        speaker_ids = melo_model.hps.data.spk2id
        if request.speaker_key not in speaker_ids:
            raise HTTPException(status_code=400, detail=f"Speaker key '{request.speaker_key}' not found for language '{request.language}'.")
        speaker_id = speaker_ids[request.speaker_key]

        logger.info(f"Generating MeloTTS audio for text: '{request.text}' with speaker: '{request.speaker_key}' in language: '{request.language}'")
        melo_model.tts_to_file(request.text, speaker_id, request.output_path)

        if not os.path.exists(request.output_path) or os.path.getsize(request.output_path) == 0:
            raise HTTPException(status_code=500, detail="MeloTTS failed to generate audio file.")

        return {"status": "success", "message": f"MeloTTS audio saved to {request.output_path}"}

    except Exception as e:
        logger.error(f"Error generating MeloTTS audio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating MeloTTS audio: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if tone_color_converter is not None else "unhealthy",
        "model": "openvoice",
        "device": device,
        "model_loaded": tone_color_converter is not None,
        "melo_models_loaded": list(melo_models.keys()),
        "checkpoint_path": ckpt_converter
    }

@app.post("/extract-se")
async def extract_speaker_embedding(request: SExtractionRequest):
    """Extract speaker embedding from audio file"""
    try:
        logger.info(f"Extracting speaker embedding from: {request.audio_path}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Audio file exists: {os.path.exists(request.audio_path)}")
        logger.info(f"Audio file absolute path: {os.path.abspath(request.audio_path)}")
        
        if not os.path.exists(request.audio_path):
            raise HTTPException(status_code=400, detail=f"Audio file not found: {request.audio_path}")
        
        if tone_color_converter is None:
            raise HTTPException(status_code=500, detail="OpenVoice models not loaded")
        
        # Extract speaker embedding
        se, audio_name = se_extractor.get_se(
            request.audio_path, 
            tone_color_converter, 
            vad=request.vad
        )
        
        # Save SE to file if output path provided
        if request.output_path:
            os.makedirs(os.path.dirname(request.output_path), exist_ok=True)
            torch.save(se.cpu(), request.output_path)
            logger.info(f"Speaker embedding saved to: {request.output_path}")
        
        return {
            "status": "success",
            "message": "Speaker embedding extracted successfully",
            "audio_name": audio_name,
            "se_shape": list(se.shape),
            "output_path": request.output_path
        }
        
    except Exception as e:
        logger.error(f"Error extracting speaker embedding: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error extracting speaker embedding: {str(e)}")

@app.post("/tone-convert")
async def tone_color_conversion(request: ToneConversionRequest):
    """Convert tone color of audio using speaker embeddings"""
    try:
        logger.info(f"Converting tone color for: {request.audio_src_path}")
        
        # Check if files exist
        if not os.path.exists(request.audio_src_path):
            raise HTTPException(status_code=400, detail=f"Source audio file not found: {request.audio_src_path}")
        if not os.path.exists(request.src_se_path):
            raise HTTPException(status_code=400, detail=f"Source SE file not found: {request.src_se_path}")
        if not os.path.exists(request.tgt_se_path):
            raise HTTPException(status_code=400, detail=f"Target SE file not found: {request.tgt_se_path}")
        
        if tone_color_converter is None:
            raise HTTPException(status_code=500, detail="OpenVoice models not loaded")
        
        # Load speaker embeddings
        src_se = torch.load(request.src_se_path, map_location=device)
        tgt_se = torch.load(request.tgt_se_path, map_location=device)
        
        # Perform tone color conversion
        tone_color_converter.convert(
            audio_src_path=request.audio_src_path,
            src_se=src_se,
            tgt_se=tgt_se,
            output_path=request.output_path,
            tau=request.tau,
            message=request.message
        )
        
        logger.info(f"Tone color conversion completed: {request.output_path}")
        
        return {
            "status": "success",
            "message": "Tone color conversion completed successfully",
            "output_path": request.output_path
        }
        
    except Exception as e:
        logger.error(f"Error in tone color conversion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in tone color conversion: {str(e)}")

@app.post("/voice-clone")
async def voice_cloning(request: VoiceCloneRequest):
    """Complete voice cloning process with tone color conversion"""
    try:
        logger.info(f"Voice cloning request: {request.subtitle_text}")
        
        if not os.path.exists(request.target_speaker_path):
            raise HTTPException(status_code=400, detail=f"Target speaker file not found: {request.target_speaker_path}")
        
        if tone_color_converter is None:
            raise HTTPException(status_code=500, detail="OpenVoice models not loaded")
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_src, \
             tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as temp_src_se, \
             tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as temp_tgt_se:
            
            temp_src_path = temp_src.name
            temp_src_se_path = temp_src_se.name
            temp_tgt_se_path = temp_tgt_se.name
        
        try:
            # Step 1: Generate base TTS audio (this would come from another TTS model)
            # For now, we'll use the target speaker as the source audio
            # In practice, this would be generated by XTTS, MeloTTS, or other TTS models
            shutil.copy(request.target_speaker_path, temp_src_path)
            
            # Step 2: Extract speaker embeddings
            logger.info("Extracting source speaker embedding...")
            src_se, _ = se_extractor.get_se(temp_src_path, tone_color_converter, vad=True)
            torch.save(src_se.cpu(), temp_src_se_path)
            
            logger.info("Extracting target speaker embedding...")
            tgt_se, _ = se_extractor.get_se(request.target_speaker_path, tone_color_converter, vad=True)
            torch.save(tgt_se.cpu(), temp_tgt_se_path)
            
            # Step 3: Perform tone color conversion
            logger.info("Performing tone color conversion...")
            tone_color_converter.convert(
                audio_src_path=temp_src_path,
                src_se=src_se,
                tgt_se=tgt_se,
                output_path=request.output_path,
                tau=request.tau
            )
            
            logger.info(f"Voice cloning completed: {request.output_path}")
            
            return {
                "status": "success",
                "message": "Voice cloning completed successfully",
                "output_path": request.output_path,
                "subtitle_text": request.subtitle_text
            }
            
        finally:
            # Clean up temporary files
            for temp_file in [temp_src_path, temp_src_se_path, temp_tgt_se_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
    except Exception as e:
        logger.error(f"Error in voice cloning: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in voice cloning: {str(e)}")

@app.post("/process")
async def process_request(request: Dict[str, Any]):
    """Generic processing endpoint for compatibility with other servers"""
    try:
        request_type = request.get("type", "voice_clone")
        
        if request_type == "voice_clone":
            # Convert dict to VoiceCloneRequest
            clone_request = VoiceCloneRequest(**request)
            return await voice_cloning(clone_request)
        elif request_type == "extract_se":
            # Convert dict to SExtractionRequest
            se_request = SExtractionRequest(**request)
            return await extract_speaker_embedding(se_request)
        elif request_type == "tone_convert":
            # Convert dict to ToneConversionRequest
            tone_request = ToneConversionRequest(**request)
            return await tone_color_conversion(tone_request)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown request type: {request_type}")
            
    except Exception as e:
        logger.error(f"Error in generic processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in processing: {str(e)}")

@app.post("/shutdown")
async def shutdown():
    """Graceful shutdown endpoint"""
    logger.info("Shutdown request received")
    return {"status": "shutdown", "message": "Server shutting down"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009) 