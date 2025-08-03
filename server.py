#!/usr/bin/env python3
"""
WhisperCapRover Server - Real-time WebSocket Transcription
Robust, production-ready speech-to-text using OpenAI Whisper with VAD
"""

import asyncio
import json
import queue
import threading
import collections
import logging
import os
from typing import Deque, Optional

import numpy as np
import webrtcvad
import whisper
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# Setup logging
debug_mode = os.getenv("WHISPER_DEBUG", "false").lower() == "true"
log_level = logging.DEBUG if debug_mode else logging.INFO

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/server.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WhisperWebSocketTranscriber:
    """Real-time WebSocket transcriber using OpenAI Whisper with VAD."""
    
    def __init__(self, model_name="base", sample_rate=16000, chunk_ms=30):
        """
        Initialize the transcriber.
        
        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            sample_rate: Audio sample rate in Hz
            chunk_ms: Duration of each audio chunk in milliseconds
        """
        logger.info(f"Initializing WhisperWebSocketTranscriber with model: {model_name}")
        
        # Load Whisper model
        logger.info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)
        logger.info(f"Whisper model {model_name} loaded successfully")
        
        # Audio processing parameters
        self.sr = sample_rate
        self.chunk_size = int(sample_rate * chunk_ms / 1000)
        self.chunk_ms = chunk_ms
        
        # VAD setup
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2 (0-3)
        logger.info("VAD initialized with aggressiveness level 2")
        
        # Audio buffering
        self.audio_queue = queue.Queue()
        self.speech_flags: Deque[bool] = collections.deque(maxlen=30)
        self.min_segment_ms = 500  # Minimum speech segment duration
        self.current_segment = []
        self.last_emit = ""
        
        # Threading control
        self._running = False
        self._worker_thread = None
        
        logger.info(f"Transcriber initialized - Sample rate: {sample_rate}Hz, Chunk size: {chunk_ms}ms")

    def put_bytes(self, data: bytes):
        """Add audio bytes to the processing queue."""
        try:
            pcm = np.frombuffer(data, dtype=np.int16)
            if pcm.size > 0:
                self.audio_queue.put(pcm)
                logger.debug(f"Added {len(data)} bytes ({pcm.size} samples) to queue")
        except Exception as e:
            logger.error(f"Error processing audio bytes: {e}")

    def start(self, send_fn):
        """Start the transcription worker thread."""
        if self._running:
            logger.warning("Transcriber already running")
            return
            
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, args=(send_fn,), daemon=True)
        self._worker_thread.start()
        logger.info("Transcription worker thread started")

    def stop(self):
        """Stop the transcription worker thread."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)
        logger.info("Transcription worker thread stopped")

    def _is_speech(self, chunk: np.ndarray) -> bool:
        """Detect if audio chunk contains speech using VAD."""
        try:
            return self.vad.is_speech(chunk.tobytes(), self.sr)
        except Exception as e:
            logger.debug(f"VAD error: {e}")
            return False

    def _transcribe(self, pcm16: np.ndarray) -> str:
        """Transcribe audio using Whisper."""
        try:
            # Convert to float32 and normalize
            audio = pcm16.astype(np.float32) / 32767.0
            
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio, 
                language="en", 
                task="transcribe", 
                fp16=False
            )
            
            text = result["text"].strip()
            logger.info(f"Transcription: '{text}'")
            return text
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    def _worker(self, send_fn):
        """Main worker thread for processing audio and transcription."""
        logger.info("🎤 Starting audio processing worker")
        
        # Debug mode: bypass VAD for testing
        debug_mode = os.getenv("WHISPER_DEBUG", "false").lower() == "true"
        # Force debug mode for testing
        debug_mode = True
        logger.debug(f"DEBUG MODE: {debug_mode}")
        if debug_mode:
            logger.debug("DEBUG MODE: VAD bypassed for testing")
        
        recording_started = False
        
        while self._running:
            try:
                # Get audio chunk from queue
                logger.debug("Worker: waiting for audio chunk...")
                chunk = self.audio_queue.get(timeout=0.1)
                logger.debug(f"Worker: received chunk of {len(chunk)} samples")
                
                # Mark recording as started when we receive first chunk
                if not recording_started:
                    logger.info("🎙️ Recording started")
                    recording_started = True
                
                logger.debug("About to check debug mode...")
                try:
                    if debug_mode:
                        # Debug mode: treat all audio as speech
                        speech = True
                        logger.debug("DEBUG: Treating chunk as speech")
                    else:
                        # Run VAD on chunk
                        speech = self._is_speech(chunk)
                    
                    logger.debug(f"Speech detection result: {speech}")
                    
                    self.speech_flags.append(speech)
                except Exception as e:
                    logger.error(f"Error in speech detection: {e}")
                    import traceback
                    logger.error(f"Speech detection traceback: {traceback.format_exc()}")
                    speech = True  # Default to speech in case of error
                    self.speech_flags.append(speech)
                
                if speech:
                    # Add to current speech segment
                    self.current_segment.append(chunk)
                    logger.debug("Speech detected, adding to segment")
                    logger.debug(f"Current segment size: {len(self.current_segment)} chunks")
                    
                    # In debug mode, process speech segments immediately after a certain size
                    # Process immediately if segment is large enough (either by chunk count or total samples)
                    total_samples = sum(len(chunk) for chunk in self.current_segment)
                    logger.debug(f"DEBUG MODE: Segment size: {len(self.current_segment)} chunks, Total samples: {total_samples}")
                    if debug_mode and (len(self.current_segment) >= 10 or total_samples >= 16000):  # Process after 10 chunks or 1 second of audio
                        logger.debug("DEBUG MODE: Processing speech segment immediately")
                        try:
                            # Calculate segment duration
                            dur_ms = len(self.current_segment) * 1000 * self.chunk_size // self.sr
                            logger.debug(f"DEBUG MODE: Processing speech segment: {dur_ms}ms")
                            
                            # Concatenate all chunks in the segment
                            logger.debug(f"Concatenating {len(self.current_segment)} chunks...")
                            pcm = np.concatenate(self.current_segment)
                            logger.debug(f"Concatenated PCM size: {len(pcm)} samples")
                            
                            # Transcribe the segment
                            logger.debug("Starting transcription...")
                            text = self._transcribe(pcm)
                            logger.debug(f"Transcription result: '{text}'")
                            
                            if text and text != self.last_emit:
                                # Send transcription via WebSocket
                                try:
                                    logger.debug("Sending transcription via WebSocket...")
                                    asyncio.run(send_fn(text))
                                    self.last_emit = text
                                    logger.info(f"📝 Transcription: '{text}'")
                                except Exception as e:
                                    logger.error(f"Error sending transcription: {e}")
                                    import traceback
                                    logger.error(f"Send transcription traceback: {traceback.format_exc()}")
                            else:
                                logger.debug("No transcription or duplicate text, skipping send")
                        except Exception as e:
                            logger.error(f"Error processing speech segment: {e}")
                            import traceback
                            logger.error(f"Process segment traceback: {traceback.format_exc()}")
                        
                        # Clear the segment for next speech
                        self.current_segment.clear()
                        logger.debug("Cleared speech segment")
                    
                    continue
                
                # Check if we just finished speaking (silence detected)
                logger.debug(f"Checking speech flags: {len(self.speech_flags)} flags, speech ratio: {sum(self.speech_flags) / len(self.speech_flags) if self.speech_flags else 0}")
                if self.current_segment and sum(self.speech_flags) / len(self.speech_flags) < 0.3:
                    # Calculate segment duration
                    dur_ms = len(self.current_segment) * 1000 * self.chunk_size // self.sr
                    logger.debug(f"Silence detected, segment duration: {dur_ms}ms")
                    
                    if dur_ms >= self.min_segment_ms:
                        logger.debug(f"Processing speech segment: {dur_ms}ms")
                        
                        try:
                            # Concatenate all chunks in the segment
                            logger.debug(f"Concatenating {len(self.current_segment)} chunks...")
                            pcm = np.concatenate(self.current_segment)
                            logger.debug(f"Concatenated PCM size: {len(pcm)} samples")
                            
                            # Transcribe the segment
                            logger.debug("Starting transcription...")
                            text = self._transcribe(pcm)
                            logger.debug(f"Transcription result: '{text}'")
                            
                            if text and text != self.last_emit:
                                # Send transcription via WebSocket
                                try:
                                    logger.debug("Sending transcription via WebSocket...")
                                    asyncio.run(send_fn(text))
                                    self.last_emit = text
                                    logger.info(f"📝 Transcription: '{text}'")
                                except Exception as e:
                                    logger.error(f"Error sending transcription: {e}")
                                    import traceback
                                    logger.error(f"Send transcription traceback: {traceback.format_exc()}")
                            else:
                                logger.debug("No transcription or duplicate text, skipping send")
                        except Exception as e:
                            logger.error(f"Error processing speech segment: {e}")
                            import traceback
                            logger.error(f"Process segment traceback: {traceback.format_exc()}")
                    
                    # Clear the segment for next speech
                    self.current_segment.clear()
                    logger.debug("Cleared speech segment")
                    
            except queue.Empty:
                # No audio data, continue
                continue
            except Exception as e:
                logger.error(f"Worker thread error: {e}")
                import traceback
                logger.error(f"Worker thread traceback: {traceback.format_exc()}")
                continue
        
        logger.info("🛑 Audio processing worker stopped")

# FastAPI application
app = FastAPI(title="WhisperCapRover Server", version="2.0.0")

@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {"message": "WhisperCapRover Server", "version": "2.0.0"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "whispercaprover"}

@app.websocket("/ws/audio")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint for real-time audio transcription."""
    await ws.accept()
    session_id = f"session-{int(asyncio.get_event_loop().time() * 1000)}"
    
    logger.info(f"🔗 New WebSocket connection: {session_id}")
    
    # Initialize transcriber
    model_name = os.getenv("WHISPER_MODEL", "base")
    transcriber = WhisperWebSocketTranscriber(model_name=model_name)
    
    # Define sender function for WebSocket
    async def sender(text: str):
        """Send transcription result via WebSocket."""
        message = {
            "type": "transcription",
            "text": text,
            "is_final": True,
            "session_id": session_id,
            "timestamp": asyncio.get_event_loop().time()
        }
        await ws.send_text(json.dumps(message))
    
    # Start transcription
    transcriber.start(sender)
    
    try:
        # Send welcome message
        welcome_msg = {
            "type": "connection_established",
            "session_id": session_id,
            "message": "Connected to WhisperCapRover Server",
            "model": model_name,
            "sample_rate": 16000,
            "chunk_ms": 30
        }
        await ws.send_text(json.dumps(welcome_msg))
        logger.debug(f"Welcome message sent to {session_id}")
        
        # Process incoming audio
        while True:
            data = await ws.receive_bytes()
            logger.debug(f"Received {len(data)} bytes from WebSocket")
            transcriber.put_bytes(data)
            
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error in {session_id}: {e}")
    finally:
        transcriber.stop()
        logger.info(f"🧹 Session {session_id} cleaned up")

@app.get("/html", response_class=HTMLResponse)
async def get_html():
    """Serve the HTML client for testing."""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>WhisperCapRover - Real-time Transcription</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .connected { background-color: #d4edda; color: #155724; }
        .disconnected { background-color: #f8d7da; color: #721c24; }
        .transcription { font-size: 18px; font-family: monospace; padding: 20px; background: #f8f9fa; border-radius: 5px; min-height: 100px; }
        button { padding: 10px 20px; margin: 10px; border: none; border-radius: 5px; cursor: pointer; }
        .start { background-color: #28a745; color: white; }
        .stop { background-color: #dc3545; color: white; }
    </style>
</head>
<body>
    <h1>🎤 WhisperCapRover - Real-time Transcription</h1>
    
    <div id="status" class="status disconnected">Disconnected</div>
    
    <div>
        <button id="startBtn" class="start" onclick="startRecording()">Start Recording</button>
        <button id="stopBtn" class="stop" onclick="stopRecording()" disabled>Stop Recording</button>
    </div>
    
    <h3>Live Transcription:</h3>
    <div id="transcription" class="transcription">Click "Start Recording" and speak...</div>
    
    <script>
        let ws = null;
        let mediaRecorder = null;
        let audioContext = null;
        let processor = null;
        
        function updateStatus(message, isConnected) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = 'status ' + (isConnected ? 'connected' : 'disconnected');
        }
        
        function updateTranscription(text) {
            document.getElementById('transcription').textContent = text;
        }
        
        async function startRecording() {
            try {
                // Get microphone access
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                
                // Create WebSocket connection
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws/audio`);
                
                ws.onopen = () => {
                    updateStatus('Connected - Recording...', true);
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.type === 'transcription' && data.text) {
                        updateTranscription(data.text);
                    }
                };
                
                ws.onclose = () => {
                    updateStatus('Disconnected', false);
                    document.getElementById('startBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                };
                
                // Set up audio processing
                audioContext = new AudioContext({ sampleRate: 16000 });
                const source = audioContext.createMediaStreamSource(stream);
                
                // Create script processor for downsampling
                processor = audioContext.createScriptProcessor(4096, 1, 1);
                
                processor.onaudioprocess = (e) => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        const input = e.inputBuffer.getChannelData(0);
                        // Convert to 16-bit PCM
                        const pcm16 = new Int16Array(input.length);
                        for (let i = 0; i < input.length; i++) {
                            pcm16[i] = Math.max(-32768, Math.min(32767, input[i] * 32768));
                        }
                        ws.send(pcm16.buffer);
                    }
                };
                
                source.connect(processor);
                processor.connect(audioContext.destination);
                
            } catch (error) {
                console.error('Error starting recording:', error);
                updateStatus('Error: ' + error.message, false);
            }
        }
        
        function stopRecording() {
            if (ws) {
                ws.close();
                ws = null;
            }
            if (processor) {
                processor.disconnect();
                processor = null;
            }
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
            updateStatus('Stopped', false);
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info("Starting WhisperCapRover Server v2.0.0")
    logger.info(f"WebSocket endpoint: ws://localhost:{port}/ws/audio")
    logger.info(f"HTML client: http://localhost:{port}/html")
    
    uvicorn.run(
        "server:app", 
        host=host, 
        port=port, 
        log_level="info"
    ) 