import { test, expect } from "bun:test";
import { l2normalize, meanNormalize } from "./vec";

test("l2normalize yields a unit vector", () => {
  const out = l2normalize(Float32Array.from([3, 4]));
  expect(out[0]).toBeCloseTo(0.6, 5);
  expect(out[1]).toBeCloseTo(0.8, 5);
});

test("meanNormalize averages then renormalises", () => {
  const out = meanNormalize([
    Float32Array.from([1, 0]),
    Float32Array.from([0, 1]),
  ]);
  // midpoint direction is (0.707, 0.707)
  expect(out[0]).toBeCloseTo(Math.SQRT1_2, 5);
  expect(out[1]).toBeCloseTo(Math.SQRT1_2, 5);
});
