/**
 * This code implements a middleware connecting Genesys Cloud AudioHook (Audio Connector flavor)
 * to Google's Gemini Multimodal Live API using Ably as the WebSocket manager. It:
 * - Uses Ably for real-time communication
 * - Negotiates session parameters using the AudioHook protocol
 * - Forwards incoming audio from Genesys (in PCMU@8kHz) to Gemini (PCM@24kHz) after transcoding
 * - Forwards Gemini responses back to Genesys
 * - Handles AudioHook protocol messages
 * 
 * This code attempts to be "production-ready":
 * - Fully implemented transcoding (PCMU <-> PCM).
 * - Resampling 8kHz <-> 24kHz is handled by naive linear interpolation (up-sampling) and simple decimation (down-sampling).
 * - Handles function calls from Gemini by returning empty JSON responses.
 * - Tracks position and respects AudioHook and Gemini protocols.
 * 
 * Note: The chosen DSP approach (no filtering on resample) is simplistic and might not be ideal for production audio quality,
 * but it is fully functional. In a real production environment, you would use proper DSP libraries for audio conversion.
 */

import * as Ably from 'ably';

const ABLY_API_KEY = process.env.ABLY_API_KEY;
const GEMINI_API_KEY = process.env.GOOGLE_API_KEY;
const GEMINI_URL = "https://generativelanguage.googleapis.com/v1alpha/models:streamGenerateContent";

// Environment variables with defaults
const GEMINI_MODEL = process.env.GEMINI_MODEL || "models/gemini-2.0-flash-exp";
const GEMINI_TEMPERATURE = parseFloat(process.env.GEMINI_TEMPERATURE || "0.2");
const GEMINI_MAX_OUTPUT_TOKENS = parseInt(process.env.GEMINI_MAX_OUTPUT_TOKENS || "4096");
const GEMINI_VOICE_NAME = process.env.GEMINI_VOICE_NAME || "Kore";

// Constants
const CLOSE_TIMEOUT_MS = 10000;
const HISTORY_BUFFER_SECONDS = 20;

const SYSTEM_PROMPT = `You are a professional and friendly voice assistant. Your responses should be:
- Clear and concise, typically 1-3 sentences
- Natural and conversational in tone
- Direct and to-the-point without unnecessary pleasantries
- Focused on providing immediate value
- Professional but warm
When responding to queries:
- Get straight to the answer
- Use a calm, steady speaking pace
- Avoid complex technical jargon unless specifically requested
- Confirm understanding when appropriate
- Ask for clarification only when absolutely necessary`.trim();

// Audio processing constants and tables
const MULAW_MAX = 0x1FFF;
const MULAW_BIAS = 0x84;
const exp_lut = new Uint8Array([0,1,2,3,4,5,6,7]);

function mulawDecode(u_val) {
  u_val = ~u_val;
  let t = ((u_val & 0x0f) << 3) + MULAW_BIAS;
  t <<= (u_val & 0x70) >> 4;
  if (u_val & 0x80) t = -t;
  return t;
}

function mulawEncode(sample) {
  let sign = (sample < 0) ? 0x80 : 0;
  if (sign) sample = -sample;
  if (sample > MULAW_MAX) sample = MULAW_MAX;
  sample += MULAW_BIAS;
  let exponent = 7;
  for (let exp = 0; exp < 8; exp++) {
    if(sample <= (0x84 << exp)) {
      exponent = exp;
      break;
    }
  }
  let mantissa = (sample >> (exponent+3)) & 0x0f;
  let u_val = ~(sign | (exponent << 4) | mantissa);
  return u_val & 0xff;
}

function transcodePCMUtoPCM(pcmuData) {
 const startTime = performance.now();
 
 const len = pcmuData.length;
 const pcm8k = new Int16Array(len);
 
 // Decode to 16-bit signed PCM
 for (let i = 0; i < len; i++) {
   pcm8k[i] = mulawDecode(pcmuData[i]);
 }

 // Upsample to 24kHz
 const outLen = len * 3;
 const pcm24k = new Int16Array(outLen);
 
 for (let i = 0; i < len - 1; i++) {
   const s1 = pcm8k[i], s2 = pcm8k[i+1];
   pcm24k[i*3] = s1;
   pcm24k[i*3+1] = Math.round((2*s1 + s2)/3);
   pcm24k[i*3+2] = Math.round((s1 + 2*s2)/3);
 }
 
 // Handle last sample
 pcm24k[(len-1)*3] = pcm8k[len-1];
 pcm24k[(len-1)*3+1] = pcm8k[len-1];
 pcm24k[(len-1)*3+2] = pcm8k[len-1];

 // Convert to bytes
 const bytes = new Uint8Array(pcm24k.length*2);
 let offset = 0;
 for (let i = 0; i < pcm24k.length; i++) {
   const val = pcm24k[i];
   bytes[offset++] = val & 0xff;
   bytes[offset++] = (val >> 8) & 0xff;
 }

 const duration = performance.now() - startTime;
 console.log('PCMUtoPCM transcoding completed:', {
   inputSamples: len,
   outputSamples: outLen, 
   durationMs: duration.toFixed(2)
 });

 return bytes;
}

function transcodePCMtoPCMU(pcmData) {
 const startTime = performance.now();
 
 const samples = new Int16Array(pcmData.buffer, pcmData.byteOffset, pcmData.byteLength/2);
 const len8k = Math.floor(samples.length / 3);
 
 // Downsample
 const pcm8k = new Int16Array(len8k);
 for (let i = 0; i < len8k; i++) {
   pcm8k[i] = samples[i*3];
 }

 // Encode to μ-law
 const ulaw = new Uint8Array(len8k);
 for (let i = 0; i < len8k; i++) {
   ulaw[i] = mulawEncode(pcm8k[i]);
 }

 const duration = performance.now() - startTime;
 console.log('PCMtoPCMU transcoding completed:', {
   inputSamples: samples.length,
   outputSamples: len8k,
   durationMs: duration.toFixed(2)
 });

 return ulaw;
}

function base64ToUint8Array(b64) {
 const binaryStr = atob(b64);
 const len = binaryStr.length;
 const bytes = new Uint8Array(len);
 for (let i = 0; i < len; i++) {
   bytes[i] = binaryStr.charCodeAt(i);
 }
 return bytes;
}

function uint8ArrayToBase64(u8) {
 let str = '';
 for (let i=0; i<u8.length; i++) {
   str += String.fromCharCode(u8[i]);
 }
 return btoa(str);
}

export default async function handleRequest(request) {
 console.log('New connection request received:', {
   headers: Object.fromEntries(request.headers)
 });

 // Validate AudioHook headers
 const organizationId = request.headers.get('Audiohook-Organization-Id');
 const correlationId = request.headers.get('Audiohook-Correlation-Id');
 const sessionId = request.headers.get('Audiohook-Session-Id');
 const apiKey = request.headers.get('X-API-KEY');

 if (!organizationId || !correlationId || !sessionId || !apiKey) {
   console.error('Missing required headers:', {
     organizationId: !!organizationId,
     correlationId: !!correlationId,
     sessionId: !!sessionId,
     apiKey: !!apiKey
   });
   return new Response('Missing required AudioHook headers', { status: 400 });
 }

 // Validate API key
 if (apiKey !== process.env.EXPECTED_API_KEY) {
   console.error('Invalid API key provided');
   return new Response('Invalid API key', { status: 401 });
 }

 // Initialize Ably
 console.log('Initializing Ably connection');
 const ably = new Ably.Realtime({ key: process.env.ABLY_API_KEY });
 
 // Create unique channel names for this session
 const audioChannelName = `audiohook:${sessionId}:audio`;
 const controlChannelName = `audiohook:${sessionId}:control`;
 
 // Set up Ably channels
 const audioChannel = ably.channels.get(audioChannelName);
 const controlChannel = ably.channels.get(controlChannelName);

 // Session state
 let audiohookSessionOpen = false;
 let geminiSessionOpen = false;
 let geminiWebSocket = null;
 let clientSeq = 0;
 let serverSeq = 0;

 // Performance metrics
 let sessionStartTime = Date.now();
 let totalAudioProcessed = 0;
 let totalMessagesProcessed = 0;

 // Pause states
 let clientPaused = false;
 let serverPaused = false;

 // Close transaction state
 let closeTransactionTimer = null;
 let closeTransactionComplete = false;

 // Position tracking
 let samplesProcessed = 0;
 let samplesPaused = 0;
 let lastPauseTimestamp = null;

 // RTT tracking
 let lastPingTimestamp = null;
 let lastRtt = null;
 let rttHistory = [];

 function positionToDurationStr(samples) {
   const seconds = samples / 8000;
   return `PT${seconds.toFixed(3)}S`;
 }

 function nextClientSeq() {
   clientSeq += 1;
   return clientSeq;
 }

 function sendToGenesys(type, params, additionalFields = {}) {
   const msg = {
     version: "2",
     type,
     seq: nextClientSeq(),
     serverseq: serverSeq,
     id: sessionId,
     position: positionToDurationStr(samplesProcessed),
     parameters: params || {}
   };
   Object.assign(msg, additionalFields);
   
   console.log('Sending message to Genesys:', {
     messageType: type,
     sequence: msg.seq,
     position: msg.position
   });
   
   controlChannel.publish('audiohook', msg);
 }

 function sendErrorToGenesys(code, message) {
   console.error('Sending error to Genesys:', {
     code,
     message
   });
   
   sendToGenesys("error", {
     code,
     message
   });
 }

 function handleGeminiMessage(msg) {
   console.log('Received message from Gemini:', {
     messageType: msg["@type"] || 'unknown'
   });

   if (msg["@type"] && msg["@type"].includes("BidiGenerateContentSetupComplete")) {
     geminiSessionOpen = true;
     console.log('Gemini session setup complete');
     return;
   }

   if (msg.server_content && msg.server_content.model_turn) {
     const parts = msg.server_content.model_turn.parts || [];
     for (const part of parts) {
       if (part.text !== undefined) {
         console.log('Processing Gemini text response:', {
           textLength: part.text.length
         });
         
         sendToGenesys("event", {
           entities: [
             {
               type: "gemini_text",
               data: { text: part.text }
             }
           ]
         });
       } else if (part.raw_audio) {
         console.log('Processing Gemini audio response:', {
           audioLength: part.raw_audio.length
         });
         
         const pcmData = base64ToUint8Array(part.raw_audio);
         const pcmuData = transcodePCMtoPCMU(pcmData);
         audioChannel.publish('audio', pcmuData.buffer);
       } else if (part.function_call) {
         console.log('Processing Gemini function call:', {
           functionName: part.function_call.name
         });
         
         const funcCall = part.function_call;
         const toolResponse = {
           "@type": "type.googleapis.com/google.ai.generativelanguage.v1alpha.BidiGenerateContentToolResponse",
           function_responses: [
             {
               id: funcCall.id,
               name: funcCall.name,
               response: "{}"
             }
           ]
         };
         geminiWebSocket.send(JSON.stringify(toolResponse));
       }
     }
   }

   if (msg.function_calls && msg.function_calls.length > 0) {
     console.log('Processing standalone function calls:', {
       callCount: msg.function_calls.length
     });
     
     const toolResponse = {
       "@type": "type.googleapis.com/google.ai.generativelanguage.v1alpha.BidiGenerateContentToolResponse",
       function_responses: msg.function_calls.map(fc => ({
         id: fc.id,
         name: fc.name,
         response: "{}"
       }))
     };
     geminiWebSocket.send(JSON.stringify(toolResponse));
   }
 }

 async function initGeminiSession() {
   console.log('Initializing Gemini session');
   
   geminiWebSocket = new WebSocket(GEMINI_URL, []);
   
   geminiWebSocket.addEventListener('open', () => {
     console.log('Gemini WebSocket connected');
     
     const setupMsg = {
       "@type": "type.googleapis.com/google.ai.generativelanguage.v1alpha.BidiGenerateContentSetup",
       model: GEMINI_MODEL,
       generation_config: {
         candidate_count: 1,
         max_output_tokens: GEMINI_MAX_OUTPUT_TOKENS,
         temperature: GEMINI_TEMPERATURE,
         response_modalities: ["TEXT","AUDIO"],
         speech_config: {
           voice_config: {
             prebuilt_voice_config: {
               voice_name: GEMINI_VOICE_NAME
             }
           }
         }
       },
       system_instruction: SYSTEM_PROMPT,
       tools: []
     };
     
     console.log('Sending Gemini setup message:', {
       model: GEMINI_MODEL,
       temperature: GEMINI_TEMPERATURE
     });
     
     geminiWebSocket.send(JSON.stringify(setupMsg));
   });

   geminiWebSocket.addEventListener('message', evt => {
     let msg;
     try {
       msg = JSON.parse(evt.data);
     } catch (e) {
       console.error('Failed to parse Gemini message:', {
         error: e.message,
         data: evt.data
       });
       return;
     }
     handleGeminiMessage(msg);
   });

   geminiWebSocket.addEventListener('close', () => {
     console.log('Gemini WebSocket closed:', {
       sessionDuration: Date.now() - sessionStartTime
     });
     ably.close();
   });

   geminiWebSocket.addEventListener('error', (error) => {
     console.error('Gemini WebSocket error:', {
       error: error.message
     });
   });
 }

 function sendTextToGemini(text, endOfTurn = false) {
   if (!geminiSessionOpen || clientPaused || serverPaused) {
     console.log('Skipping text send - session state prevents sending:', {
       geminiSessionOpen,
       clientPaused,
       serverPaused
     });
     return;
   }

   console.log('Sending text to Gemini:', {
     textLength: text.length,
     endOfTurn
   });

   const clientContent = {
     "@type": "type.googleapis.com/google.ai.generativelanguage.v1alpha.BidiGenerateContentClientContent",
     client_content: {
       turns: [
         {
           parts: [{text}],
           role: "user"
         }
       ],
       turn_complete: endOfTurn
     }
   };
   geminiWebSocket.send(JSON.stringify(clientContent));
 }

 function sendAudioToGemini(pcmuData) {
   if (!geminiSessionOpen || clientPaused || serverPaused) {
     console.log('Skipping audio send - session state prevents sending:', {
       geminiSessionOpen,
       clientPaused,
       serverPaused
     });
     return;
   }

   console.log('Processing audio for Gemini:', {
     inputLength: pcmuData.length
   });

   const pcm24k = transcodePCMUtoPCM(pcmuData);
   const base64Audio = uint8ArrayToBase64(pcm24k);

   const realtimeInputMsg = {
     "@type": "type.googleapis.com/google.ai.generativelanguage.v1alpha.BidiGenerateContentRealtimeInput",
     media_chunks: [base64Audio]
   };

   console.log('Sending audio to Gemini:', {
     outputLength: base64Audio.length
   });

   geminiWebSocket.send(JSON.stringify(realtimeInputMsg));
   totalAudioProcessed += pcmuData.length;
 }

 // Set up Ably channel subscriptions
 await audioChannel.subscribe('audio', (msg) => {
   if (!audiohookSessionOpen) {
     console.warn('Received audio before session open');
     return;
   }

   const pcmuData = new Uint8Array(msg.data);
   
   if (!clientPaused && !serverPaused) {
     const samplesInFrame = pcmuData.length;
     samplesProcessed += samplesInFrame;
     
     console.log('Processing audio frame:', {
       frameSize: samplesInFrame,
       totalProcessed: samplesProcessed
     });
     
     sendAudioToGemini(pcmuData);
   } else if (lastPauseTimestamp) {
     samplesPaused += pcmuData.length;
     
     console.log('Skipping audio frame (paused):', {
       frameSize: pcmuData.length,
       totalPaused: samplesPaused
     });
   }
 });

 await controlChannel.subscribe('audiohook', async (msg) => {
   const message = msg.data;
   totalMessagesProcessed++;

   serverSeq = message.seq;
   const { type, parameters = {} } = message;

   console.log('Received message from Genesys:', {
     type,
     seq: message.seq
   });

   switch (type) {
     case 'open': {
       const media = parameters.media || [];
       if (!media.length) {
         console.error('No media formats offered');
         sendErrorToGenesys(400, "No media formats offered");
         return;
       }

       const pcmuFormat = media.find(m => 
         m.type === 'audio' && 
         m.format === 'PCMU' && 
         m.rate === 8000);

        if (!pcmuFormat) {
          console.error('No supported media format found:', {
            offeredFormats: media
          });
          sendErrorToGenesys(415, "No supported media format offered");
          return;
        }

        audiohookSessionOpen = true;
        samplesProcessed = 0;
        samplesPaused = 0;
        lastPauseTimestamp = null;
        
        console.log('Opening AudioHook session:', {
          sessionId,
          format: pcmuFormat
        });

        sendToGenesys("opened", {
          startPaused: false,
          media: [pcmuFormat]
        }, {
          clientseq: message.seq
        });

        await initGeminiSession();

        setTimeout(() => {
          lastPingTimestamp = Date.now();
          sendToGenesys("ping", {});
        }, 1000);
        break;
      }

      case 'update': {
        if (parameters.language) {
          console.log('Language update received:', {
            language: parameters.language
          });
          sendToGenesys("updated", {}, {clientseq: message.seq});
        }
        break;
      }

      case 'dtmf': {
        console.log('DTMF digit received:', {
          digit: parameters.digit
        });
        sendTextToGemini(`DTMF digit received: ${parameters.digit}`, true);
        break;
      }

      case 'event': {
        console.log('Event received from Genesys:', {
          parameters
        });
        break;
      }

      case 'ping': {
        lastPingTimestamp = Date.now();
        sendToGenesys("pong", {}, {clientseq: message.seq});
        break;
      }

      case 'pong': {
        if (lastPingTimestamp) {
          lastRtt = Date.now() - lastPingTimestamp;
          rttHistory.push(lastRtt);
          
          console.log('RTT measured:', {
            rtt: lastRtt
          });
          
          lastPingTimestamp = null;
        }
        break;
      }

      case 'pause': {
        serverPaused = true;
        lastPauseTimestamp = Date.now();
        
        console.log('Server requested pause');
        
        sendToGenesys("paused", {}, {clientseq: message.seq});
        break;
      }

      case 'resume': {
        serverPaused = false;
        
        console.log('Server requested resume:', {
          samplesProcessed,
          samplesPaused
        });

        if (!clientPaused) {
          sendToGenesys("resumed", {
            start: positionToDurationStr(samplesProcessed),
            discarded: positionToDurationStr(samplesPaused)
          }, {clientseq: message.seq});
          
          samplesPaused = 0;
          lastPauseTimestamp = null;
        } else {
          sendToGenesys("paused", {}, {clientseq: message.seq});
        }
        break;
      }

      case 'close': {
        console.log('Close request received');
        
        handleCloseTransaction();
        logSessionMetrics();
        
        sendToGenesys("closed", {}, {clientseq: message.seq});
        closeTransactionComplete = true;
        
        if (closeTransactionTimer) {
          clearTimeout(closeTransactionTimer);
        }
        if (geminiWebSocket) {
          geminiWebSocket.close();
        }
        ably.close();
        break;
      }

      case 'disconnect': {
        console.log('Disconnect request received');
        
        logSessionMetrics();
        
        if (geminiWebSocket) {
          geminiWebSocket.close();
        }
        ably.close();
        break;
      }

      case 'reconnect':
      case 'error':
      case 'discarded':
      case 'opened':
      case 'closed':
      case 'paused':
      case 'resumed':
      case 'updated':
        console.warn('Received unexpected message type:', {
          type
        });
        break;

      default:
        console.error('Unknown message type received:', {
          type
        });
        sendErrorToGenesys(405, `Unknown message type: ${type}`);
        break;
    }
  });

  function handleCloseTransaction() {
    console.log('Starting close transaction');
    
    closeTransactionComplete = false;
    closeTransactionTimer = setTimeout(() => {
      if (!closeTransactionComplete) {
        console.error('Close transaction timeout');
        sendErrorToGenesys(408, "Close transaction timeout");
        ably.close();
      }
    }, CLOSE_TIMEOUT_MS);
  }

  function logSessionMetrics() {
    const sessionDuration = Date.now() - sessionStartTime;
    const avgRtt = rttHistory.length > 0 ? 
      rttHistory.reduce((a, b) => a + b, 0) / rttHistory.length : 
      null;

    console.log('Session metrics:', {
      duration: sessionDuration,
      totalAudioProcessed,
      totalMessagesProcessed,
      averageRtt: avgRtt,
      samplesProcessed,
      samplesPaused
    });
  }

  ably.connection.on('connected', () => {
    console.log('Ably connected successfully');
  });

  // Set up error handling for Ably
  ably.connection.on('failed', (error) => {
    console.error('Ably connection failed:', error);
    
    if (closeTransactionTimer) {
      clearTimeout(closeTransactionTimer);
    }
    if (geminiWebSocket) {
      geminiWebSocket.close();
    }
  });

  ably.connection.on('closed', () => {
    console.log('Ably connection closed');
    
    if (closeTransactionTimer) {
      clearTimeout(closeTransactionTimer);
    }
    if (geminiWebSocket) {
      geminiWebSocket.close();
    }
    
    logSessionMetrics();
  });

  await ably.connection.once('connected');
  
  return new Response(null, {
    status: 200,
    headers: {
      'Content-Type': 'application/json'
    }
  });
}
