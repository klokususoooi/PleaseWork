const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Toggles
const SHOW_REASONING = false;           // Set to true if you want <think> tags
const ENABLE_THINKING_MODE = false;

// Model mapping (keep your original)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking' 
};

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Main chat completions endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Smart model selection with fallback (your original logic - kept as-is)
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      try {
        await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        }).then(r => {
          if (r.status >= 200 && r.status < 300) nimModel = model;
        });
      } catch (e) {}
      
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
    }
    
    // Transform to NIM format
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: !!stream
    };
    
    // Axios request with timeout (very important for Vercel)
    const axiosConfig = {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 300000   // 5 minutes - prevents hanging forever
    };
    
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, axiosConfig);
    
    if (stream) {
      // === IMPROVED STREAMING HANDLER ===
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');   // Helps with some hosts
      res.flushHeaders();   // Send headers immediately (big speed win)

      let buffer = '';

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';   // Keep incomplete line

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          if (trimmed === 'data: [DONE]') {
            res.write('data: [DONE]\n\n');
            continue;
          }

          if (trimmed.startsWith('data: ')) {
            const dataStr = trimmed.slice(6);

            try {
              let data = JSON.parse(dataStr);

              // Simple reasoning handling (keeps your toggle)
              if (data.choices?.[0]?.delta) {
                const delta = data.choices[0].delta;
                const content = delta.content || '';
                const reasoning = delta.reasoning_content || '';

                if (SHOW_REASONING && reasoning) {
                  delta.content = `<think>\n${reasoning}\n</think>\n\n${content}`;
                  delete delta.reasoning_content;
                } else {
                  delta.content = content;
                  if (delta.reasoning_content) delete delta.reasoning_content;
                }
              }

              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              // Fallback: forward raw
              res.write(`${trimmed}\n\n`);
            }
          }
        }
      });

      response.data.on('end', () => {
        res.write('data: [DONE]\n\n');
        res.end();
      });

      response.data.on('error', (err) => {
        console.error('NIM stream error:', err.message);
        res.write('data: {"error": "Stream error from NIM"}\n\n');
        res.end();
      });

      // Clean up if client disconnects
      req.on('close', () => {
        response.data.destroy();
      });

    } else {
      // Non-streaming (kept similar to yours, simplified)
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => ({
          index: choice.index,
          message: {
            role: choice.message?.role || 'assistant',
            content: SHOW_REASONING && choice.message?.reasoning_content 
              ? `<think>\n${choice.message.reasoning_content}\n</think>\n\n${choice.message.content || ''}`
              : choice.message?.content || ''
          },
          finish_reason: choice.finish_reason
        })),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy error:', error.message || error);
    const status = error.response?.status || 500;
    res.status(status).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: status
      }
    });
  }
});

// Catch-all
app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 }
  });
});

app.listen(PORT, () => {
  console.log(`🚀 OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health: http://localhost:${PORT}/health`);
});
