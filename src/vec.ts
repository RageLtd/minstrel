// Small vector helpers for combining embeddings. Everything stays L2-normalised
// so cosine comparisons against the stored track vectors remain meaningful.

export function l2normalize(v: Float32Array): Float32Array {
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i]! * v[i]!;
  const norm = Math.sqrt(sum) || 1;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i]! / norm;
  return out;
}

/** Mean of several vectors, renormalised — a centroid direction. */
export function meanNormalize(vectors: Float32Array[]): Float32Array {
  if (vectors.length === 0) throw new Error("meanNormalize: no vectors");
  const dim = vectors[0]!.length;
  const acc = new Float32Array(dim);
  for (const v of vectors) {
    for (let i = 0; i < dim; i++) acc[i] = acc[i]! + v[i]!;
  }
  for (let i = 0; i < dim; i++) acc[i] = acc[i]! / vectors.length;
  return l2normalize(acc);
}
