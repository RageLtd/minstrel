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
  expect(captured?.body.tools).toHaveLength(1);
});
