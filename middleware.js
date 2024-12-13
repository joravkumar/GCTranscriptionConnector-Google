export const config = {
  runtime: 'edge',
};

/**
 * This code implements a middleware connecting Genesys Cloud AudioHook (Audio Connector flavor)
 * to Google's Gemini Multimodal Live API. It:
 * - Establishes a WebSocket connection with Genesys Cloud.
 * - Negotiates session parameters using the AudioHook protocol.
 * - Forwards incoming audio from Genesys (in PCMU@8kHz) to Gemini (PCM@24kHz) after transcoding.
 * - Forwards Gemini responses (text and audio) back to Genesys (transcoding from PCM@24kHz to PCMU@8kHz).
 * - Handles pause/resume, updates, and other AudioHook protocol messages.
 * - Integrates Gemini by setting up a bidi session, sending user turns (text/audio) and receiving model responses.
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

const GEMINI_MODEL = "models/gemini-2.0-flash-exp";
const GEMINI_URL = "wss://generativelanguage.googleapis.com/v1alpha/models:bidigeneratecontent";
const GEMINI_API_KEY = process.env.GOOGLE_API_KEY || 'YOUR_API_KEY';

// PCMU <-> PCM conversion tables/functions (G.711 μ-law)
const MULAW_MAX = 0x1FFF;
const MULAW_BIAS = 0x84;
const exp_lut = new Uint8Array([0,1,2,3,4,5,6,7]);

function mulawDecode(u_val) {
  // u_val: 0-255
  u_val = ~u_val;
  let t = ((u_val & 0x0f) << 3) + MULAW_BIAS;
  t <<= (u_val & 0x70) >> 4;
  if (u_val & 0x80) t = -t;
  return t;
}

function mulawEncode(sample) {
  // sample: signed 16-bit integer
  let sign = (sample < 0) ? 0x80 : 0;
  if (sign) sample = -sample;
  if (sample > MULAW_MAX) sample = MULAW_MAX;
  sample += MULAW_BIAS;
  let exponent = 7;
  for (let exp = 0; exp < 8; exp++){
    if(sample <= (0x84 << exp)) {
      exponent = exp; 
      break;
    }
  }
  let mantissa = (sample >> (exponent+3)) & 0x0f;
  let u_val = ~(sign | (exponent << 4) | mantissa);
  return u_val & 0xff;
}

/**
 * Transcode PCMU (8kHz) to PCM (16-bit 24kHz little-endian):
 * Steps:
 * 1) Decode PCMU to linear PCM@8kHz (16-bit signed).
 * 2) Resample from 8kHz to 24kHz by linear interpolation (3x upsampling).
 */
function transcodePCMUtoPCM(pcmuData) {
  // pcμ data is u-law encoded. Each byte = 1 sample at 8kHz.
  // Decode to 16-bit signed PCM.
  const len = pcmuData.length;
  const pcm8k = new Int16Array(len);
  for (let i = 0; i < len; i++) {
    pcm8k[i] = mulawDecode(pcmuData[i]);
  }

  // Upsample from 8kHz to 24kHz by factor of 3.
  // We'll do linear interpolation: for each pair (x[i], x[i+1]) produce x[i], (2/3*x[i] + 1/3*x[i+1]), (1/3*x[i] + 2/3*x[i+1]).
  // For the last sample, just repeat it.
  const outLen = (len - 1) * 3 + 1 * 3; // Actually (len * 3).
  const pcm24k = new Int16Array(len * 3);
  for (let i = 0; i < len - 1; i++) {
    const s1 = pcm8k[i], s2 = pcm8k[i+1];
    pcm24k[i*3] = s1;
    pcm24k[i*3+1] = Math.round((2*s1 + s2)/3);
    pcm24k[i*3+2] = Math.round((s1 + 2*s2)/3);
  }
  // last sample
  pcm24k[(len-1)*3] = pcm8k[len-1];
  pcm24k[(len-1)*3+1] = pcm8k[len-1];
  pcm24k[(len-1)*3+2] = pcm8k[len-1];

  // Convert Int16Array to Uint8Array (16-bit little-endian)
  const bytes = new Uint8Array(pcm24k.length*2);
  let offset = 0;
  for (let i = 0; i < pcm24k.length; i++) {
    const val = pcm24k[i];
    bytes[offset++] = val & 0xff;
    bytes[offset++] = (val >> 8) & 0xff;
  }
  return bytes;
}

/**
 * Transcode PCM (16-bit 24kHz little-endian) back to PCMU (8kHz):
 * Steps:
 * 1) Convert bytes to Int16 PCM array @24kHz.
 * 2) Downsample to 8kHz by picking every 3rd sample (no filtering).
 * 3) Encode to PCMU (μ-law).
 */
function transcodePCMtoPCMU(pcmData) {
  // pcmData is 16-bit LE PCM @ 24kHz
  const samples = new Int16Array(pcmData.buffer, pcmData.byteOffset, pcmData.byteLength/2);

  // Downsample: take every 3rd sample
  const len8k = Math.floor(samples.length / 3);
  const pcm8k = new Int16Array(len8k);
  for (let i = 0; i < len8k; i++) {
    pcm8k[i] = samples[i*3];
  }

  // Encode to μ-law
  const ulaw = new Uint8Array(len8k);
  for (let i = 0; i < len8k; i++) {
    ulaw[i] = mulawEncode(pcm8k[i]);
  }

  return ulaw;
}

/**
 * Convert base64 to Uint8Array
 */
function base64ToUint8Array(b64) {
  const binaryStr = atob(b64);
  const len = binaryStr.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryStr.charCodeAt(i);
  }
  return bytes;
}

/**
 * Convert Uint8Array to base64
 */
function uint8ArrayToBase64(u8) {
  let str = '';
  for (let i=0; i<u8.length; i++) {
    str += String.fromCharCode(u8[i]);
  }
  return btoa(str);
}

export default async function handleRequest(request) {
  const upgradeHeader = request.headers.get('Upgrade');
  if (!upgradeHeader || upgradeHeader.toLowerCase() !== 'websocket') {
    return new Response('Not a WebSocket request', { status: 400 });
  }

  const [client, server] = Object.values(new WebSocketPair());
  server.accept();

  // Session state
  const sessionId = crypto.randomUUID();
  let audiohookSessionOpen = false;
  let geminiSessionOpen = false;
  let geminiWebSocket = null;
  let clientSeq = 0;   
  let serverSeq = 0;   

  // Pause states
  let clientPaused = false;   
  let serverPaused = false;   

  // Track audio position in samples. 
  // position in seconds = samplesProcessed / 8000 (since original timeline in AudioHook is at 8k)
  // We'll maintain all position computations at 8k rate to match AudioHook's position reference.
  let samplesProcessed = 0;

  function positionToDurationStr(samples) {
    // Convert samples @8kHz to ISO8601 Duration
    const seconds = samples / 8000;
    return `PT${seconds.toFixed(5)}S`;
  }

  function nextClientSeq() {
    clientSeq += 1;
    return clientSeq;
  }

  // Send message to Genesys
  function sendToGenesys(type, params, additionalFields = {}) {
    const msg = {
      version: "2",
      type,
      seq: nextClientSeq(),
      serverseq: serverSeq,
      id: sessionId,
      // Position is always the "virtual" position at 8k rate
      position: positionToDurationStr(samplesProcessed),
      parameters: params || {}
    };
    Object.assign(msg, additionalFields);
    server.send(JSON.stringify(msg));
  }

  // Send error to Genesys
  function sendErrorToGenesys(code, message) {
    sendToGenesys("error", {
      code,
      message
    });
  }

  // Handle Gemini messages
  function handleGeminiMessage(msg) {
    // Setup complete
    if (msg["@type"] && msg["@type"].includes("BidiGenerateContentSetupComplete")) {
      geminiSessionOpen = true;
      return;
    }

    // Model responses
    if (msg.server_content && msg.server_content.model_turn) {
      const parts = msg.server_content.model_turn.parts || [];
      for (const part of parts) {
        if (part.text !== undefined) {
          // Text response from Gemini -> send as event entity
          sendToGenesys("event", {
            entities: [
              {
                type: "gemini_text",
                data: { text: part.text }
              }
            ]
          });
        } else if (part.raw_audio) {
          // Audio from Gemini in base64 PCM@24kHz -> transcode to PCMU@8kHz
          const pcmData = base64ToUint8Array(part.raw_audio);
          const pcmuData = transcodePCMtoPCMU(pcmData);
          // Send as binary frames to Genesys
          server.send(pcmuData.buffer);
          // Advance position by samples in 8k domain?
          // Actually, audio from Gemini is not from the client's perspective.
          // For AudioHook, we only track outgoing (client) audio. The position reflects client input timeline, 
          // not server audio. We do not modify samplesProcessed for Gemini responses.
        } else if (part.function_call) {
          // Respond to function call with empty result
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

    // Tool calls outside model_turn
    if (msg.function_calls && msg.function_calls.length > 0) {
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

  // Initialize Gemini session
  async function initGeminiSession() {
    geminiWebSocket = new WebSocket(GEMINI_URL, []);
    geminiWebSocket.accept();
    
    geminiWebSocket.addEventListener('open', () => {
      const setupMsg = {
        "@type": "type.googleapis.com/google.ai.generativelanguage.v1alpha.BidiGenerateContentSetup",
        model: GEMINI_MODEL,
        generation_config: {
          candidate_count: 1,
          max_output_tokens: 1024,
          temperature: 0.2,
          response_modalities: ["TEXT","AUDIO"],
          speech_config: {
            voice_config: {
              prebuilt_voice_config: {
                voice_name: "Kore"
              }
            }
          }
        },
        system_instruction: "You are a helpful voice assistant. Respond concisely.",
        tools: []
      };
      geminiWebSocket.send(JSON.stringify(setupMsg));
    });

    geminiWebSocket.addEventListener('message', evt => {
      let msg;
      try {
        msg = JSON.parse(evt.data);
      } catch (e) {
        console.error("Invalid JSON from Gemini:", evt.data);
        return;
      }
      handleGeminiMessage(msg);
    });

    geminiWebSocket.addEventListener('close', () => {
      console.log("Gemini WebSocket closed");
    });
  }

  // Send text input from Genesys to Gemini
  function sendTextToGemini(text, endOfTurn = false) {
    if (!geminiSessionOpen || clientPaused || serverPaused) return;
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

  // Forward audio input to Gemini
  function sendAudioToGemini(pcmuData) {
    if (!geminiSessionOpen || clientPaused || serverPaused) return;
    // Transcode PCMU@8k to PCM@24k
    const pcm24k = transcodePCMUtoPCM(pcmuData);
    const base64Audio = uint8ArrayToBase64(pcm24k);

    const realtimeInputMsg = {
      "@type": "type.googleapis.com/google.ai.generativelanguage.v1alpha.BidiGenerateContentRealtimeInput",
      media_chunks: [base64Audio]
    };
    geminiWebSocket.send(JSON.stringify(realtimeInputMsg));
  }

  server.addEventListener('message', async evt => {
    const data = evt.data;

    if (data instanceof ArrayBuffer) {
      // Binary audio frame from Genesys
      const pcmuData = new Uint8Array(data);
      // Update samplesProcessed based on how many samples are in this frame at 8kHz:
      const samplesInFrame = pcmuData.length; 
      samplesProcessed += samplesInFrame;
      // Forward to Gemini
      sendAudioToGemini(pcmuData);
      return;
    }

    let message;
    try {
      message = JSON.parse(data);
    } catch (e) {
      console.error("Invalid JSON from Genesys:", data);
      return;
    }

    serverSeq = message.seq;
    const { type, parameters = {} } = message;

    switch (type) {
      case 'open': {
        // Pick a media format
        const offeredMedia = parameters.media || [];
        const chosenMedia = offeredMedia.length > 0 ? [offeredMedia[0]] : [];
        sendToGenesys("opened", {
          startPaused: false,
          media: chosenMedia
        }, { clientseq: message.seq });
        
        audiohookSessionOpen = true;
        await initGeminiSession();
        break;
      }

      case 'update': {
        // Language changed or other updates
        if (parameters.language) {
          // Could send a system instruction update to Gemini if needed, but Gemini doesn't support live updates.
          // We'll just log it.
          console.log("Language updated:", parameters.language);
        }
        break;
      }

      case 'dtmf': {
        // Received DTMF digit
        // Forward as text message to Gemini with end of turn
        sendTextToGemini(`DTMF digit received: ${parameters.digit}`, true);
        break;
      }

      case 'event': {
        // Server to client event (rare)
        console.log("Received event from Genesys:", message);
        break;
      }

      case 'ping':
        // Respond with pong
        sendToGenesys("pong", {}, {clientseq: message.seq});
        break;

      case 'pause':
        // Server requested pause
        serverPaused = true;
        sendToGenesys("paused", {}, {clientseq: message.seq});
        break;

      case 'resume':
        // Server requested resume
        serverPaused = false;
        if (!clientPaused) {
          sendToGenesys("resumed", {
            start: positionToDurationStr(samplesProcessed),
            discarded: "PT0S"
          }, {clientseq: message.seq});
        } else {
          // Still client paused
          sendToGenesys("paused", {}, {clientseq: message.seq});
        }
        break;

      case 'close':
        // Conversation ended
        sendToGenesys("closed", {}, {clientseq: message.seq});
        if (geminiWebSocket) {
          geminiWebSocket.close();
        }
        server.close();
        break;

      // Server-side messages that shouldn't appear here or we can ignore safely:
      case 'reconnect':
      case 'disconnect':
      case 'error':
      case 'discarded':
      case 'opened':
      case 'closed':
      case 'paused':
      case 'resumed':
      case 'pong':
      case 'updated':
        console.log("Received unexpected or server-side-only message type:", type);
        break;

      default:
        // Unknown message type
        sendErrorToGenesys(405, `Unknown message type: ${type}`);
        break;
    }
  });

  return new Response(null, {
    status: 101,
    webSocket: client
  });
}
