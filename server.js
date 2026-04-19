// Cloudflare Worker - NVIDIA NIM Proxy for Janitor AI
// Optimized for DeepSeek-V3.2 with proper streaming & CORS

// ================== CONFIGURATION ==================
const NIM_API_BASE = 'https://integrate.api.nvidia.com/v1';

// Model mapping (same as your Vercel version)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking'
};

// Toggles (same as your Vercel version)
const SHOW_REASONING = false;   // Keep false – you don't want <think> tags
const ENABLE_THINKING_MODE = false;

// ================== MAIN HANDLER ==================
export default {
  async fetch(request, env) {
    // Handle CORS preflight (required for Janitor)
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        status: 204,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization',
          'Access-Control-Max-Age': '86400',
        },
      });
    }

    const url = new URL(request.url);
    const path = url.pathname;

    // Health check
    if (path === '/health') {
      return jsonResponse({ status: 'ok', service: 'NIM Proxy (Cloudflare)', reasoning_display: SHOW_REASONING, thinking_mode: ENABLE_THINKING_MODE });
    }

    // List models (for Janitor compatibility)
    if (path === '/v1/models' && request.method === 'GET') {
      const models = Object.keys(MODEL_MAPPING).map(id => ({
        id, object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy'
      }));
      return jsonResponse({ object: 'list', data: models });
    }

    // Main chat completions endpoint
    if (path === '/v1/chat/completions' && request.method === 'POST') {
      return handleChatCompletions(request, env);
    }

    // 404 for anything else
    return new Response('Not found', { status: 404, headers: corsHeaders() });
  }
};

// ================== CHAT HANDLER ==================
async function handleChatCompletions(request, env) {
  try {
    const body = await request.json();
    const { model, messages, temperature, max_tokens, stream } = body;

    // Smart model selection (same logic as Vercel)
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      const modelLower = model.toLowerCase();
      if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
        nimModel = 'meta/llama-3.1-405b-instruct';
      } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
        nimModel = 'meta/llama-3.1-70b-instruct';
      } else {
        nimModel = 'meta/llama-3.1-8b-instruct';
      }
    }

    // Build request to NIM – ONLY essential parameters (no top_p, penalties, etc.)
    // This prevents Janitor's "0" values from killing creativity.
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature !== undefined ? temperature : 0.7,  // 0.9 from Janitor will be used
      max_tokens: max_tokens || 9024,
      stream: stream === true
    };

    // Add extra_body only if thinking mode is enabled (it's false, so skip)
    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
    }

    // Make the request to NVIDIA NIM with a normal-looking User-Agent
    const nimResponse = await fetch(`${NIM_API_BASE}/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${env.NIM_API_KEY}`,
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0 (compatible; Node.js/20.0.0; Vercel)'  // Mimic Vercel
      },
      body: JSON.stringify(nimRequest),
    });

    if (!nimResponse.ok) {
      const errorText = await nimResponse.text();
      return new Response(errorText, { status: nimResponse.status, headers: corsHeaders() });
    }

    // Handle streaming vs non-streaming
    if (stream) {
      return handleStreamingResponse(nimResponse);
    } else {
      const data = await nimResponse.json();
      return jsonResponse(data);
    }

  } catch (error) {
    console.error(error);
    return new Response(JSON.stringify({ error: error.message }), { status: 500, headers: corsHeaders() });
  }
}

// ================== STREAMING HANDLER ==================
function handleStreamingResponse(nimResponse) {
  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  const encoder = new TextEncoder();

  // Process the NIM stream and forward to client
  (async () => {
    const reader = nimResponse.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          if (trimmed === 'data: [DONE]') {
            await writer.write(encoder.encode('data: [DONE]\n\n'));
            continue;
          }

          if (trimmed.startsWith('data: ')) {
            const jsonStr = trimmed.slice(6);
            try {
              let data = JSON.parse(jsonStr);
              // Remove reasoning_content if present (since SHOW_REASONING = false)
              if (data.choices?.[0]?.delta?.reasoning_content) {
                delete data.choices[0].delta.reasoning_content;
              }
              await writer.write(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
            } catch (e) {
              // If JSON parsing fails, forward raw line (fallback)
              await writer.write(encoder.encode(`${trimmed}\n\n`));
            }
          }
        }
      }
      await writer.write(encoder.encode('data: [DONE]\n\n'));
    } catch (err) {
      console.error('Stream error:', err);
    } finally {
      await writer.close();
      reader.releaseLock();
    }
  })();

  return new Response(readable, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': '*',
      'X-Accel-Buffering': 'no',
    },
  });
}

// ================== UTILITIES ==================
function corsHeaders() {
  return {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  };
}

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...corsHeaders(),
    },
  });
}
