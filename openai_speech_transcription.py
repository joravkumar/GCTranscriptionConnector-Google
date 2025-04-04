import asyncio
import audioop
import os
import json
import time
import threading
import queue
import tempfile
from datetime import timedelta
import aiohttp
import logging
from collections import deque
import math
import re

from config import (
    OPENAI_API_KEY,
    OPENAI_SPEECH_MODEL
)
from language_mapping import (
    normalize_language_code, 
    get_openai_language_code, 
    is_openai_unsupported_language,
    get_language_name_for_prompt,
    get_language_specific_prompt
)

# Known spurious transcription artifacts that should be filtered
KNOWN_ARTIFACTS = [
    "context:", 
    "ring", 
    "context",
    "begin",
    "beep",
    "[beep]",
    "[ring]"
]

# Regular expressions for more complex pattern matching
ARTIFACT_PATTERNS = [
    r"^\s*context:?\s*",  # Matches "context:" or "context" at the start
    r"^\s*ring\s*",       # Matches "ring" at the start
    r"^\s*\[?beep\]?\s*", # Matches "beep" or "[beep]" at the start
    r"^\s*\[?ring\]?\s*"  # Matches "ring" or "[ring]" at the start
]

class MockResult:
    def __init__(self):
        self.results = []

class Result:
    def __init__(self, alternatives=None, is_final=True):
        self.alternatives = alternatives or []
        self.is_final = is_final

class Alternative:
    def __init__(self, transcript="", confidence=0.9, words=None):
        self.transcript = transcript
        self.confidence = confidence
        self.words = words or []

class Word:
    def __init__(self, word="", start_offset=None, end_offset=None, confidence=0.9):
        self.word = word
        self.start_offset = start_offset or timedelta(seconds=0)
        self.end_offset = end_offset or timedelta(seconds=1)
        self.confidence = confidence
        
    def __str__(self):
        # Enhanced string representation with 5 decimal precision for debugging
        return f"Word('{self.word}', start={self.start_offset.total_seconds():.5f}s, end={self.end_offset.total_seconds():.5f}s, conf={self.confidence:.5f})"

class StreamingTranscription:
    def __init__(self, language: str, channels: int, logger):
        self.logger = logger
        self.language = normalize_language_code(language)
        self.openai_language = get_openai_language_code(self.language)
        self.is_unsupported_language = is_openai_unsupported_language(self.language)
        
        if self.is_unsupported_language:
            self.language_prompt = get_language_name_for_prompt(self.language)
            self.logger.info(f"Initialized StreamingTranscription with language={self.language}, unsupported by OpenAI API; using language name '{self.language_prompt}' in prompt")
        else:
            self.language_prompt = None
            self.logger.info(f"Initialized StreamingTranscription with language={self.language}, openai_language={self.openai_language}")
            
        self.channels = channels
        self.audio_queues = [queue.Queue() for _ in range(channels)]
        self.response_queues = [queue.Queue() for _ in range(channels)]
        self.streaming_threads = [None] * channels
        self.running = True
        
        # Audio buffering for utterance detection
        self.audio_buffers = [[] for _ in range(channels)]
        self.buffer_durations = [0.0 for _ in range(channels)]
        self.last_process_time = [time.time() for _ in range(channels)]
        
        # VAD parameters for detecting speech segments
        self.vad_threshold = 200  # RMS threshold for speech detection
        self.is_speech = [False for _ in range(channels)]
        self.silence_frames = [0 for _ in range(channels)]
        self.speech_frames = [0 for _ in range(channels)]
        
        # Silence duration to consider end of utterance (800ms)
        self.silence_threshold_frames = 8  # 8 frames at 100ms = 800ms of silence
        
        # Enhanced parameters for more accurate speech detection
        self.energy_history = [deque(maxlen=10) for _ in range(channels)]
        self.adaptive_threshold = [self.vad_threshold for _ in range(channels)]
        self.silence_energy_threshold = 150  # Lower threshold to ensure we don't clip speech
        
        # Keep track of accumulated audio for streaming
        self.accumulated_audio = [bytearray() for _ in range(channels)]
        
        # Track last utterance to prevent duplicates
        self.last_transcripts = ["" for _ in range(channels)]
        
        # Initial audio discarding to prevent artifacts from call setup sounds
        self.initial_frames_processed = [0 for _ in range(channels)]
        self.skip_initial_frames = 5  # Skip first 5 frames to avoid beeps/connection sounds

        # Minimum confidence threshold for tokens (to filter out uncertain transcriptions)
        self.token_confidence_threshold = 0.2

    def start_streaming(self):
        for channel in range(self.channels):
            self.streaming_threads[channel] = threading.Thread(
                target=self.streaming_recognize_thread, args=(channel,)
            )
            self.streaming_threads[channel].daemon = True
            self.streaming_threads[channel].start()

    def stop_streaming(self):
        self.running = False
        for channel in range(self.channels):
            self.audio_queues[channel].put(None)
        for channel in range(self.channels):
            if self.streaming_threads[channel] and self.streaming_threads[channel].is_alive():
                self.streaming_threads[channel].join(timeout=1.0)

    def streaming_recognize_thread(self, channel):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Frame size for processing (100ms of audio at 8kHz)
            frame_size = 800  # 8000 Hz / 10 frames per second
            
            while self.running:
                try:
                    audio_chunk = self.audio_queues[channel].get(timeout=0.1)
                    if audio_chunk is None:
                        # Process any remaining audio before shutting down
                        if len(self.accumulated_audio[channel]) > 0:
                            self._process_accumulated_audio(channel, loop)
                        break
                    
                    # Skip initial frames to avoid startup noises/beeps
                    if self.initial_frames_processed[channel] < self.skip_initial_frames:
                        self.initial_frames_processed[channel] += 1
                        self.logger.debug(f"Channel {channel}: Skipping initial frame {self.initial_frames_processed[channel]} to avoid startup artifacts")
                        continue
                    
                    # Approximate frame duration
                    frame_duration = len(audio_chunk) / 16000.0  # 8000 Hz * 2 bytes per sample
                    
                    # Check audio energy for VAD with enhanced precision
                    rms = audioop.rms(audio_chunk, 2)
                    
                    # Update energy history and calculate adaptive threshold
                    self.energy_history[channel].append(rms)
                    if len(self.energy_history[channel]) >= 5:
                        # Adapt threshold based on recent energy levels
                        avg_energy = sum(self.energy_history[channel]) / len(self.energy_history[channel])
                        self.adaptive_threshold[channel] = max(self.silence_energy_threshold, 
                                                             min(self.vad_threshold, avg_energy * 0.6))
                    
                    # Determine if current frame is speech using adaptive threshold
                    is_current_speech = rms > self.adaptive_threshold[channel]
                    
                    # Always add to accumulated buffer for continuous streaming
                    self.accumulated_audio[channel].extend(audio_chunk)
                    
                    # VAD state machine for utterance detection with improved timing precision
                    if is_current_speech:
                        self.silence_frames[channel] = 0
                        self.speech_frames[channel] += 1
                        
                        if not self.is_speech[channel] and self.speech_frames[channel] >= 2:
                            self.is_speech[channel] = True
                            self.logger.debug(f"Channel {channel}: Speech detected (RMS: {rms:.2f}, threshold: {self.adaptive_threshold[channel]:.2f})")
                            
                    else:
                        # Not speech
                        self.speech_frames[channel] = 0
                        
                        if self.is_speech[channel]:
                            self.silence_frames[channel] += 1
                            
                            # If we've detected enough silence after speech, process the utterance
                            if self.silence_frames[channel] >= self.silence_threshold_frames:
                                self.is_speech[channel] = False
                                self.logger.debug(f"Channel {channel}: End of speech detected after {self.silence_frames[channel] * 0.1:.2f}s of silence")
                                # Process with high precision timestamps
                                self._process_accumulated_audio(channel, loop)
                    
                    # Periodically process accumulated audio even during continuous speech
                    current_time = time.time()
                    if current_time - self.last_process_time[channel] > 3.0 and len(self.accumulated_audio[channel]) > frame_size * 30:
                        self.logger.debug(f"Channel {channel}: Processing accumulated audio due to timeout")
                        self._process_accumulated_audio(channel, loop)
                    
                    # Prevent buffer overflow
                    if len(self.accumulated_audio[channel]) > frame_size * 300:  # ~30 seconds
                        self.logger.warning(f"Channel {channel}: Buffer overflow, forcing processing")
                        self._process_accumulated_audio(channel, loop)
                
                except queue.Empty:
                    pass
                except Exception as e:
                    self.logger.error(f"Error in streaming thread for channel {channel}: {e}")
                    
            loop.close()
        except Exception as e:
            self.logger.error(f"Fatal error in streaming thread for channel {channel}: {str(e)}")
            self.response_queues[channel].put(e)

    def _process_accumulated_audio(self, channel, loop):
        """Process accumulated audio using OpenAI streaming transcription"""
        if len(self.accumulated_audio[channel]) < 1600:  # Less than 0.1s at 8kHz
            self.logger.debug(f"Accumulated audio too short ({len(self.accumulated_audio[channel])} bytes), skipping")
            self.accumulated_audio[channel] = bytearray()
            self.last_process_time[channel] = time.time()
            return
            
        try:
            # Check RMS to avoid processing silence
            audio_data = bytes(self.accumulated_audio[channel])
            rms = audioop.rms(audio_data, 2)
            
            if rms < self.vad_threshold * 0.7:
                self.logger.debug(f"Audio mostly silence (RMS: {rms}), skipping")
                self.accumulated_audio[channel] = bytearray()
                self.last_process_time[channel] = time.time()
                return
                
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                self._write_wav_file(temp_wav, audio_data)
                
                try:
                    # Use OpenAI streaming transcription
                    result = loop.run_until_complete(
                        self.stream_transcribe_audio(temp_wav.name, channel)
                    )
                    
                    if result and not isinstance(result, Exception):
                        # Check if transcript is different from last one
                        transcript_text = ""
                        if result.results and result.results[0].alternatives:
                            transcript_text = result.results[0].alternatives[0].transcript
                        
                        # Apply filtering to remove spurious artifacts
                        filtered_transcript = self._filter_spurious_artifacts(transcript_text)
                        
                        # Update the transcript in the result
                        if result.results and result.results[0].alternatives and filtered_transcript:
                            result.results[0].alternatives[0].transcript = filtered_transcript
                        
                        # Only send if non-empty and different from last transcript
                        if filtered_transcript and filtered_transcript != self.last_transcripts[channel]:
                            self.response_queues[channel].put(result)
                            self.last_transcripts[channel] = filtered_transcript
                except Exception as e:
                    self.logger.error(f"Error in OpenAI streaming transcription: {e}")
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(temp_wav.name)
                    except:
                        pass
                        
            # Reset accumulated audio and update last process time
            self.accumulated_audio[channel] = bytearray()
            self.last_process_time[channel] = time.time()
        except Exception as e:
            self.logger.error(f"Error processing accumulated audio: {e}")
            self.accumulated_audio[channel] = bytearray()
            self.last_process_time[channel] = time.time()

    def _filter_spurious_artifacts(self, transcript):
        """Filter out known spurious artifacts from transcripts"""
        if not transcript:
            return transcript
            
        # First apply regex patterns to filter out complex patterns
        for pattern in ARTIFACT_PATTERNS:
            transcript = re.sub(pattern, "", transcript)
            
        # Then check for exact matches at beginning of transcript
        for artifact in KNOWN_ARTIFACTS:
            # Full line is the artifact
            if transcript.strip() == artifact:
                return ""
            
            # Beginning of line starts with the artifact
            if transcript.strip().startswith(artifact + " "):
                transcript = transcript.replace(artifact + " ", "", 1)
            
        # Remove any remaining duplicated spaces
        transcript = re.sub(r'\s+', ' ', transcript).strip()
        
        return transcript

    def _write_wav_file(self, temp_wav, audio_data):
        """Write PCM16 data to a WAV file"""
        sample_rate = 8000
        channels = 1
        sample_width = 2
        
        # Write WAV header
        temp_wav.write(b'RIFF')
        temp_wav.write((36 + len(audio_data)).to_bytes(4, 'little'))
        temp_wav.write(b'WAVE')
        
        temp_wav.write(b'fmt ')
        temp_wav.write((16).to_bytes(4, 'little'))
        temp_wav.write((1).to_bytes(2, 'little'))
        temp_wav.write((channels).to_bytes(2, 'little'))
        temp_wav.write((sample_rate).to_bytes(4, 'little'))
        temp_wav.write((sample_rate * channels * sample_width).to_bytes(4, 'little'))
        temp_wav.write((channels * sample_width).to_bytes(2, 'little'))
        temp_wav.write((sample_width * 8).to_bytes(2, 'little'))
        
        temp_wav.write(b'data')
        temp_wav.write(len(audio_data).to_bytes(4, 'little'))
        temp_wav.write(audio_data)
        temp_wav.flush()

    async def stream_transcribe_audio(self, file_path, channel):
        """Stream transcribe audio using OpenAI's streaming API"""
        try:
            openai_lang = self.openai_language
            
            if self.is_unsupported_language:
                self.logger.info(f"Using special handling for unsupported language {self.language}: adding '{self.language_prompt}' to prompt instead of language code")
            else:
                self.logger.info(f"Simple transcribing audio from channel {channel} with OpenAI model {OPENAI_SPEECH_MODEL}")
                self.logger.info(f"Using language code for OpenAI: '{openai_lang}' (converted from '{self.language}')")
            
            url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            
            with open(file_path, 'rb') as audio_file:
                form_data = aiohttp.FormData()
                form_data.add_field('file', 
                                   audio_file, 
                                   filename=os.path.basename(file_path),
                                   content_type='audio/wav')
                form_data.add_field('model', OPENAI_SPEECH_MODEL)
                form_data.add_field('response_format', 'json')
                
                # Enable streaming for real-time results
                form_data.add_field('stream', 'true')
                
                # Include logprobs for token confidence
                form_data.add_field('include[]', 'logprobs')
                
                # Set temperature to 0 for most deterministic results
                form_data.add_field('temperature', '0')
                
                # Build prompt based on language support status
                if self.is_unsupported_language:
                    # For unsupported languages, use a language-specific prompt
                    prompt = get_language_specific_prompt(self.language)
                    # Add instruction to ignore beeps and system sounds
                    base_prompt = prompt
                    prompt = f"{base_prompt} Ignore initial beeps, rings, and system sounds."
                    self.logger.info(f"Using language-specific prompt for {self.language_prompt}: '{prompt}'")
                    form_data.add_field('prompt', prompt)
                    
                    # Don't send the language parameter for unsupported languages
                    self.logger.info(f"Omitting language parameter for unsupported language {self.language}")
                else:
                    # Only specify language if it's not empty and is a supported language
                    if openai_lang:
                        form_data.add_field('language', openai_lang)
                    
                    # Add contextual prompt for service call transcriptions
                    # Include instruction to ignore beeps/rings/system sounds
                    prompt = "This is a customer service call. The customer may be discussing problems with services or products. Ignore initial beeps, rings, and system sounds."
                    form_data.add_field('prompt', prompt)
                
                full_transcript = ""
                words = []
                avg_confidence = 0.9  # Default
                confidence_sum = 0
                confidence_count = 0
                low_confidence_tokens = []
                
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, headers=headers, data=form_data, timeout=30) as response:
                            if response.status == 200:
                                # Process streaming response
                                buffer = ""
                                async for line in response.content:
                                    line = line.decode('utf-8').strip()
                                    if line.startswith('data: '):
                                        event_data = line[6:]  # Remove 'data: ' prefix
                                        if event_data == '[DONE]':
                                            break
                                            
                                        try:
                                            event_json = json.loads(event_data)
                                            event_type = event_json.get('type')
                                            
                                            # Handle different event types
                                            if event_type == 'transcript.text.delta':
                                                delta = event_json.get('delta', '')
                                                full_transcript += delta
                                                
                                                # Extract logprobs for confidence
                                                if 'logprobs' in event_json:
                                                    for logprob in event_json['logprobs']:
                                                        token = logprob.get('token', '')
                                                        token_logprob = logprob.get('logprob', 0)
                                                        # Convert logprob to linear probability and cap at 0.99
                                                        token_confidence = min(0.99, math.exp(token_logprob))
                                                        
                                                        # Check if token has low confidence (might be an artifact)
                                                        if token_confidence < self.token_confidence_threshold:
                                                            low_confidence_tokens.append(token)
                                                            self.logger.debug(f"Low confidence token: '{token}' with confidence {token_confidence:.4f}")
                                                        
                                                        confidence_sum += token_confidence
                                                        confidence_count += 1
                                                
                                            elif event_type == 'transcript.text.done':
                                                # Final transcript with all tokens
                                                full_transcript = event_json.get('text', '')
                                                
                                                # Calculate final confidence from logprobs
                                                if 'logprobs' in event_json and event_json['logprobs']:
                                                    for logprob in event_json['logprobs']:
                                                        token = logprob.get('token', '')
                                                        token_logprob = logprob.get('logprob', 0)
                                                        token_confidence = min(0.99, math.exp(token_logprob))
                                                        
                                                        # Check if token has low confidence
                                                        if token_confidence < self.token_confidence_threshold:
                                                            low_confidence_tokens.append(token)
                                                            self.logger.debug(f"Low confidence token (final): '{token}' with confidence {token_confidence:.4f}")
                                                        
                                                        confidence_sum += token_confidence
                                                        confidence_count += 1
                                                
                                        except json.JSONDecodeError:
                                            self.logger.warning(f"Failed to parse streaming event: {event_data}")
                                            continue
                                
                                # Calculate average confidence
                                if confidence_count > 0:
                                    avg_confidence = confidence_sum / confidence_count
                                
                                # Filter out spurious artifacts from the transcript
                                filtered_transcript = self._filter_spurious_artifacts(full_transcript)
                                
                                self.logger.debug(f"Original transcript: {full_transcript}")
                                self.logger.debug(f"Filtered transcript: {filtered_transcript}")
                                self.logger.debug(f"Average confidence: {avg_confidence}")
                                if low_confidence_tokens:
                                    self.logger.debug(f"Low confidence tokens: {', '.join(low_confidence_tokens)}")
                                
                                # Log the processed transcript with timing information
                                self.logger.info(f"OpenAI transcription processed successfully: '{filtered_transcript}', " 
                                               f"confidence: {avg_confidence:.5f}, "
                                               f"audio length: {len(audio_data)/16000:.5f}s")
                                               
                                # Create a response object similar to Google API response with precise timestamps
                                return self.create_response_object(filtered_transcript, avg_confidence)
                            else:
                                error_text = await response.text()
                                self.logger.error(f"OpenAI API error: {response.status} - {error_text}")
                                # If the error is related to language, log more details for debugging
                                if "language" in error_text.lower():
                                    self.logger.error(f"Language-related error. Used language: {self.language}, OpenAI language: {openai_lang}, Is unsupported: {self.is_unsupported_language}")
                                return None
                except asyncio.TimeoutError:
                    self.logger.warning(f"OpenAI API timeout for channel {channel}")
                    return None
                except Exception as e:
                    self.logger.error(f"Error in OpenAI streaming API: {str(e)}")
                    return None
        except Exception as e:
            self.logger.error(f"Error in stream_transcribe_audio: {str(e)}")
            return None

    def create_response_object(self, transcript, confidence):
        """Create a response object compatible with Google API format"""
        if not transcript:
            return None
            
        mock_result = MockResult()
        
        # Split transcript into words for token-level information
        text_words = transcript.split()
        words = []
        
        if text_words:
            # Create synthetic timing for words
            # Calculate more realistic total duration based on word count
            # We use similar logic as we did in audio_hook_server.py for consistency
            total_duration = min(max(len(text_words) * 0.3, 1.0), 10.0)
            avg_duration = total_duration / len(text_words)
            
            for i, word_text in enumerate(text_words):
                # More precise calculation of start and end times (5 decimal places)
                start_time = i * avg_duration
                end_time = (i + 1) * avg_duration
                
                # Consider word length for more realistic duration
                # Longer words should have proportionally longer durations
                word_length_factor = len(word_text) / 5  # Normalize around 5 chars
                word_duration = avg_duration * min(max(word_length_factor, 0.5), 2.0)
                
                # Ensure we don't exceed total duration
                if i == len(text_words) - 1:
                    end_time = total_duration
                
                word = Word(
                    word=word_text,
                    start_offset=timedelta(seconds=start_time),
                    end_offset=timedelta(seconds=min(end_time, total_duration)),
                    confidence=confidence  # Use calculated confidence
                )
                words.append(word)
        
            self.logger.debug(f"Created synthetic timestamps for {len(words)} words, " 
                             f"total duration: {total_duration:.5f}s")
        
        alternative = Alternative(
            transcript=transcript,
            confidence=confidence,  # Use calculated confidence
            words=words
        )
        
        result = Result(
            alternatives=[alternative],
            is_final=True
        )
        
        mock_result.results = [result]
        return mock_result

    def feed_audio(self, audio_stream: bytes, channel: int):
        """Feed audio into the pipeline for the given channel"""
        if not audio_stream or channel >= self.channels:
            return
        
        # Convert PCMU (u-law) to PCM16 for processing
        pcm16_data = audioop.ulaw2lin(audio_stream, 2)
        self.audio_queues[channel].put(pcm16_data)

    def get_response(self, channel: int):
        """Get the next transcription response for the given channel"""
        if channel >= self.channels:
            return None
        try:
            return self.response_queues[channel].get_nowait()
        except queue.Empty:
            return None
