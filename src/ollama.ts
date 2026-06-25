// Thin client for Ollama's native /api/chat tool-calling, plus tolerant
// extraction of the first tool call. Ollama returns function.arguments as a
// parsed object on the native path, but we also accept a JSON string defensively.

export interface OllamaTool {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

export interface OllamaToolCall {
  function: { name: string; arguments: unknown };
}

export interface OllamaMessage {
  role: string;
  content: string;
  thinking?: string;
  tool_calls?: OllamaToolCall[];
}

export interface OllamaChatResponse {
  message: OllamaMessage;
  done: boolean;
}

export interface ChatOptions {
  baseUrl?: string;
  model?: string;
  think?: boolean;
  fetchImpl?: typeof fetch;
}

export async function chatWithTools(
  messages: OllamaMessage[],
  tools: OllamaTool[],
  opts: ChatOptions = {},
): Promise<OllamaChatResponse> {
  const baseUrl = opts.baseUrl ?? process.env.OLLAMA_URL ?? "http://localhost:11434";
  const model = opts.model ?? process.env.MINSTREL_MODEL ?? "qwen3";
  const doFetch = opts.fetchImpl ?? fetch;

  const res = await doFetch(`${baseUrl}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model,
      messages,
      tools,
      stream: false,
      think: opts.think ?? false,
    }),
  });
  if (!res.ok) {
    throw new Error(`ollama chat failed: ${res.status} ${await res.text()}`);
  }
  return (await res.json()) as OllamaChatResponse;
}

export interface ParsedToolCall {
  name: string;
  args: Record<string, unknown>;
}

/** First tool call from a response, or null if the model didn't call one. */
export function firstToolCall(resp: OllamaChatResponse): ParsedToolCall | null {
  const calls = resp.message.tool_calls;
  if (!calls || calls.length === 0) return null;
  const call = calls[0]!;
  let args = call.function.arguments;
  if (typeof args === "string") {
    try {
      args = JSON.parse(args);
    } catch {
      args = {};
    }
  }
  return {
    name: call.function.name,
    args: (args && typeof args === "object" ? args : {}) as Record<string, unknown>,
  };
}
