"""ASR models wrapper using Faster-Whisper."""
import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from faster_whisper import WhisperModel
from util.logging import get_logger
from util.config import config

logger = get_logger(__name__)


class ASRModel:
    """Wrapper for Faster-Whisper model with stage-specific configuration."""
    
    def __init__(
        self,
        model_name: str,
        compute_type: str,
        is_interim: bool = False,
        device: str = "cuda"
    ):
        """
        Initialize ASR model.
        
        Args:
            model_name: Model size (small, base, large-v3, distil-large-v3)
            compute_type: Compute type (int8_float16, float16)
            is_interim: Whether this is interim (fast) or final (quality) model
            device: Device to run on (cuda or cpu)
        """
        self.model_name = model_name
        self.compute_type = compute_type
        self.is_interim = is_interim
        self.device = device
        
        logger.info(f"Loading {'interim' if is_interim else 'final'} model: {model_name}, compute={compute_type}")
        
        start_time = time.time()
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            num_workers=4 if is_interim else 8,
        )
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f}s: {model_name}")
    
    def get_transcribe_params(self, language: str = "auto") -> Dict[str, Any]:
        """
        Get stage-specific transcription parameters.
        
        Args:
            language: Language code or 'auto' for detection
            
        Returns:
            Dictionary of parameters for transcribe()
        """
        if self.is_interim:
            # Speed-first configuration
            return {
                "language": None if language == "auto" else language,
                "beam_size": 1,
                "best_of": 1,
                "patience": 0.0,
                "temperature": 0.0,
                "condition_on_previous_text": False,
                "vad_filter": False,
                "without_timestamps": True,
                "compression_ratio_threshold": 2.6,
                "no_speech_threshold": 0.4,
                "logprob_threshold": -1.2,
            }
        else:
            # Quality-first configuration
            return {
                "language": None if language == "auto" else language,
                "beam_size": 5,
                "patience": 1.0,
                "temperature": [0.0, 0.2, 0.4],
                "condition_on_previous_text": True,
                "vad_filter": True,
                "without_timestamps": False,
                "compression_ratio_threshold": 2.4,
                "no_speech_threshold": 0.6,
                "logprob_threshold": -0.9,
            }
    
    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "auto",
        initial_prompt: Optional[str] = None
    ) -> Tuple[str, Optional[str], float, List[Dict[str, Any]]]:
        """
        Transcribe audio.
        
        Args:
            audio: Audio samples as float32 numpy array [-1, 1]
            language: Language code or 'auto'
            initial_prompt: Optional prompt for domain hints
            
        Returns:
            Tuple of (text, detected_language, confidence, segments)
        """
        # Ensure audio is float32 in [-1, 1]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Get transcription parameters
        params = self.get_transcribe_params(language)
        if initial_prompt:
            params["initial_prompt"] = initial_prompt
        
        # Transcribe
        start_time = time.time()
        segments_list, info = self.model.transcribe(audio, **params)
        
        # Collect segments
        text_parts = []
        segments_data = []
        
        for segment in segments_list:
            text_parts.append(segment.text)
            segments_data.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            })
        
        text = " ".join(text_parts).strip()
        detected_language = info.language if hasattr(info, 'language') else None
        confidence = info.language_probability if hasattr(info, 'language_probability') else 0.0
        
        decode_time = time.time() - start_time
        
        logger.debug(
            f"{'Interim' if self.is_interim else 'Final'} decode: "
            f"{len(audio) / 16000:.2f}s audio in {decode_time:.3f}s, "
            f"text_len={len(text)}, lang={detected_language}"
        )
        
        return text, detected_language, confidence, segments_data
    
    def detect_language(self, audio: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Detect language from audio.
        
        Args:
            audio: Audio samples as float32 numpy array [-1, 1]
            
        Returns:
            Tuple of (language_code, confidence)
        """
        # Ensure audio is float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Use short segment for language detection (first 2 seconds)
        max_samples = 2 * 16000
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        try:
            _, info = self.model.transcribe(audio, beam_size=1, language=None)
            return info.language, info.language_probability
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return None, 0.0


class ASRModels:
    """Container for both interim and final models."""
    
    def __init__(self):
        """Initialize both ASR models."""
        logger.info("Initializing ASR models...")
        
        # Load interim model (fast)
        self.interim = ASRModel(
            model_name=config.interim_model,
            compute_type=config.interim_compute,
            is_interim=True,
            device="cuda"
        )
        
        # Load final model (quality)
        self.final = ASRModel(
            model_name=config.final_model,
            compute_type=config.final_compute,
            is_interim=False,
            device="cuda"
        )
        
        logger.info("ASR models initialized successfully")
    
    def warmup(self):
        """Warm up models with dummy audio."""
        logger.info("Warming up ASR models...")
        
        # Create 2 seconds of dummy audio
        dummy_audio = np.random.randn(32000).astype(np.float32) * 0.01
        
        # Warmup interim model
        try:
            self.interim.transcribe(dummy_audio, language="en")
            logger.info("Interim model warmup complete")
        except Exception as e:
            logger.error(f"Interim model warmup failed: {e}")
        
        # Warmup final model
        try:
            self.final.transcribe(dummy_audio, language="en")
            logger.info("Final model warmup complete")
        except Exception as e:
            logger.error(f"Final model warmup failed: {e}")
