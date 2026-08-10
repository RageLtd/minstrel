import { test, expect } from "bun:test";
import {
  chatWithTools,
  firstToolCall,
  type OllamaChatResponse,
  type OllamaTool,
} from "./ollama";

function resp(toolCalls?: OllamaChatResponse["message"]["tool_calls"]): OllamaChatResponse {
  return { message: { role: "assistant", content: "", tool_calls: toolCalls }, done: true };
}

test("extracts a tool call with object arguments", () => {
  const r = resp([{ function: { name: "search_tracks", arguments: { count: 10 } } }]);
  expect(firstToolCall(r)).toEqual({ name: "search_tracks", args: { count: 10 } });
});

test("tolerates JSON-string arguments", () => {
  const r = resp([{ function: { name: "search_tracks", arguments: '{"count":5}' } }]);
  expect(firstToolCall(r)).toEqual({ name: "search_tracks", args: { count: 5 } });
});

test("returns null when the model called no tool", () => {
  expect(firstToolCall(resp(undefined))).toBeNull();
  expect(firstToolCall(resp([]))).toBeNull();
});

test("posts a well-formed chat request to the configured host", async () => {
  let captured: { url: string; body: any } | undefined;
  const fakeFetch = (async (url: string, init: RequestInit) => {
    captured = { url, body: JSON.parse(init.body as string) };
    return new Response(JSON.stringify(resp()), { status: 200 });
  }) as unknown as typeof fetch;

  const tool: OllamaTool = {
    type: "function",
    function: { name: "search_tracks", description: "d", parameters: { type: "object" } },
  };
  await chatWithTools([{ role: "user", content: "hi" }], [tool], {
    baseUrl: "http://spark:11434",
    model: "qwen3.5",
    fetchImpl: fakeFetch,
  });

  expect(captured?.url).toBe("http://spark:11434/api/chat");
  expect(captured?.body.model).toBe("qwen3.5");
  expect(captured?.body.stream).toBe(false);
  expect(captured?.body.think).toBe(true);
  expect(captured?.body.tools).toHaveLength(1);
  // no temperature given → no options key; the server default governs
  expect(captured?.body.options).toBeUndefined();
});

test("uses gemma4:26b when no model is configured", async () => {
  const saved = process.env.MINSTREL_MODEL;
  delete process.env.MINSTREL_MODEL;
  let model: unknown;
  const fakeFetch = (async (_url: string, init: RequestInit) => {
    model = JSON.parse(init.body as string).model;
    return new Response(JSON.stringify(resp()), { status: 200 });
  }) as unknown as typeof fetch;

  try {
    await chatWithTools([], [], { fetchImpl: fakeFetch });
    expect(model).toBe("gemma4:26b");
  } finally {
    if (saved === undefined) {
      delete process.env.MINSTREL_MODEL;
    } else {
      process.env.MINSTREL_MODEL = saved;
    }
  }
});

test("temperature rides in options when set", async () => {
  let captured: { body: any } | undefined;
  const fakeFetch = (async (_url: string, init: RequestInit) => {
    captured = { body: JSON.parse(init.body as string) };
    return new Response(JSON.stringify(resp()), { status: 200 });
  }) as unknown as typeof fetch;

  await chatWithTools([{ role: "user", content: "hi" }], [], {
    temperature: 0.1,
    fetchImpl: fakeFetch,
  });

  expect(captured?.body.options).toEqual({ temperature: 0.1 });
});

async function capturedThink(opts: Parameters<typeof chatWithTools>[2]) {
  let think: unknown;
  const fakeFetch = (async (_url: string, init: RequestInit) => {
    think = JSON.parse(init.body as string).think;
    return new Response(JSON.stringify(resp()), { status: 200 });
  }) as unknown as typeof fetch;
  await chatWithTools([{ role: "user", content: "hi" }], [], {
    ...opts,
    fetchImpl: fakeFetch,
  });
  return think;
}

test("thinking defaults ON, MINSTREL_THINK=false opts out, explicit option wins", async () => {
  const saved = process.env.MINSTREL_THINK;
  delete process.env.MINSTREL_THINK;
  try {
    expect(await capturedThink({})).toBe(true);

    process.env.MINSTREL_THINK = "false";
    expect(await capturedThink({})).toBe(false);
    expect(await capturedThink({ think: true })).toBe(true);

    process.env.MINSTREL_THINK = "true";
    expect(await capturedThink({})).toBe(true);
    expect(await capturedThink({ think: false })).toBe(false);
  } finally {
    if (saved === undefined) {
      delete process.env.MINSTREL_THINK;
    } else {
      process.env.MINSTREL_THINK = saved;
    }
  }
});

test("preserves an explicit thinking level", async () => {
  let body: any;
  const fakeFetch = (async (_url: string, init: RequestInit) => {
    body = JSON.parse(init.body as string);
    return new Response(JSON.stringify(resp()), { status: 200 });
  }) as unknown as typeof fetch;

  await chatWithTools([], [], { think: "high", fetchImpl: fakeFetch });

  expect(body.think).toBe("high");
});
