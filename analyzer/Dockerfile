# Build context is the repo root (see compose.yaml) so schema.sql is reachable.
#
# DGX Spark / Blackwell (GB10, compute capability 12.1) needs an arm64 NGC
# PyTorch image with CUDA >= 12.9. VERIFY this tag against your DGX OS release
# before building — NGC tags are date-based (YY.MM) and must carry Blackwell +
# aarch64 support. Pulling from nvcr.io may require `docker login nvcr.io`.
FROM nvcr.io/nvidia/pytorch:25.04-py3

WORKDIR /app

# Install our deps against the image's prebuilt torch/CUDA. Do NOT reinstall
# torch — the NGC build carries the Blackwell kernels the public wheels lack.
RUN pip install --no-cache-dir \
      transformers \
      librosa \
      soundfile \
      sqlite-vec

# store.py resolves schema.sql at parents[2] of the package → repo root, which
# maps to / here (the package lives at /app/minstrel_analyzer).
COPY schema.sql /schema.sql
COPY analyzer/minstrel_analyzer /app/minstrel_analyzer

# Overridden per service in compose (analyze vs embed_server).
CMD ["python", "-m", "minstrel_analyzer.embed_server"]
