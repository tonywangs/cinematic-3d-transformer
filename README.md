# Cinematic 3D Transformer

A guided, inspectable film of a real Pythia-70M computation. Follow a prompt through embeddings, rotary attention, parallel attention/MLP branches, six residual blocks, and next-token selection. Every displayed number comes from a saved execution.

## Run

```sh
npm ci
npm run dev
```

Open `http://127.0.0.1:5174`. Six examples and eight greedy continuation steps are bundled. The production site is entirely static and makes no inference calls.

- Play, pause, scrub or jump among seven chapters.
- Explore enables orbit controls; guided mode owns the camera.
- Select tokens, layers, heads and operations with ordinary keyboard-accessible controls.
- Inspect any flat tensor index, the complete attention matrix, and exact next-token probabilities.
- Reduced motion uses fixed chapter cameras. Devices without WebGL retain the complete HTML inspector.

Motion is illustrative, not measured hardware timing. Pythia uses **parallel** attention and MLP branches: both independently normalize the incoming residual, then their outputs are added to it. Attention is weighted aggregation, not a complete explanation of a prediction.

## Reproduce data

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r scripts/requirements.txt
.venv/bin/python scripts/extract.py
npm test
npm run build
```

Model: `EleutherAI/pythia-70m` (non-deduped), final checkpoint `step143000`, immutable revision `de3e4e2d6cbb3b1a51f90fe153ea51fcd8c7c852`. Native GPT-NeoX; PyTorch 2.14.1 and Transformers 5.21.0; CPU, float32, eager attention, evaluation mode, no KV cache. Model configuration is asserted before extraction.

The six exact prompts are retained in `scripts/extract.py`; all their genuine continuations are retained, including awkward or repetitive ones. No desired completion was forced. Each forward pass recomputes the entire prefix. Read-only hooks capture embeddings, both normalizations, Q/K before and after RoPE, V, full attention, concatenated weighted values, attention output, MLP pre/post-GELU and output, residuals, final normalization and vocabulary logits.

`TraceBundle/1` manifests record tensor shapes, little-endian float32 offsets, exact token IDs, checksums, immutable model/tokenizer revision, extraction-source hash and ten source samples per tensor. Output probabilities use a float64 softmax of float32 logits: the native float32 softmax sum drifted by about 0.000021 on a fixture with a large shared logit offset. No logits were changed.

All 48 curated forward passes and 20 seeded prompts passed **exact native-logit parity**, identical greedy tokens, causal zeros, attention row sums, weighted-value reconstruction, exact parallel residual recombination and softmax normalization. Node tests independently verify every shard hash, source samples, top probabilities and selected token. See `docs/validation.json`.

Each step is fetched separately and checksum-verified; at most three decoded steps are cached. The largest complete prompt is below 50 MB uncompressed. The renderer uses data textures for scalar tiles and instanced bars. Tiles use per-tile signed color scaling; the inspector's preview shows the first 128 cells and allows access to every cell. Strongest-eight attention edges are illustrative; the full matrix and omitted probability mass remain available.

## Deterministic MP4 export

Install FFmpeg, then run both:

```sh
npm run dev
node scripts/export-server.mjs
```

Open **Notes → Export 90s film** or **Export portrait**. The browser advances the same pure `sceneAt(time, selection)` clock at exact 30 FPS, sends PNG frames to the loopback-only encoder, and FFmpeg writes `exports/film.mp4` (1920×1080, 90 seconds) or `exports/portrait.mp4` (1080×1920, 30 seconds). Captions are burned in; no soundtrack. Keep the page open until “MP4 saved.” Export requires the local server, not a public backend. Frame recording uses the actual Three.js renderer, not an independently approximated animation.

## Attribution and licenses

Model: [EleutherAI/Pythia](https://huggingface.co/EleutherAI/pythia-70m), Apache-2.0. Prior art: [Brendan Bycroft's LLM visualization](https://github.com/bbycroft/llm-viz), [Transformer Explainer](https://github.com/poloclub/transformer-explainer), [AttentionViz](https://arxiv.org/abs/2305.03210). The interface and renderer are original; no code was copied from those projects. Model weights are not included. Original project code is MIT-licensed.
