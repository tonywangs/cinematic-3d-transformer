# Cinematic 3D Transformer: Fly Through One Prediction

Status: implementation-ready future project plan. Prepared October 25, 2026. This document authorizes no compute spending or publication by itself.

## Product and scope

Build a beautiful, interactive journey through an actual small transformer predicting its next token. The camera follows a sentence into token embeddings, across attention heads and feed-forward layers, and out into a probability distribution. Every displayed activation, attention connection, and prediction comes from a saved model execution. The experience should work as both a 60–90-second film and an explorable website.

Success is a compelling, accurate artifact that someone can understand and share. A scientific discovery, training breakthrough, or performance optimization is not required. The distinctive work is the presentation: clear scale changes, excellent camera choreography, inspectable real numbers, and a readable explanation of one prediction.

Use **EleutherAI/Pythia-70M**, final training checkpoint `step143000`, resolved to an immutable revision during acquisition. Its verified configuration has six layers, eight attention heads, 512 hidden dimensions, a 2,048-wide MLP, rotary position encoding, parallel attention/MLP residual branches, and 50,304 output vocabulary entries. It is small enough for inexpensive tracing while producing genuine language-model computations. Read the configuration rather than embedding undocumented architecture assumptions. This is a small pretrained model, not a frontier assistant or a newly trained model.

V1 includes one model, six curated English prompts, and up to eight greedy continuation tokens per prompt. Prompts contain at most 48 input tokens; the full traced context stays below 64. Start with ordinary completions, repeated patterns, a short rhyme, a two-item list, simple code, and an ambiguous continuation. Select the final exact texts after inspecting their genuine outputs; keep the selection record. Do not force a desired continuation. No public arbitrary-prompt execution, model switching, training slider, causal intervention, or chatbot interface in v1. Those belong to separate projects.

## Experience and story

The default page opens on a dark, restrained canvas with a readable sentence floating above a compact overview of all six blocks. Cyan represents attention, amber represents MLP transformations, and a neutral white path represents the residual stream. Signed numerical values use a separate consistent diverging color scale. Color is reinforced with labels and shapes.

The guided film has seven beats:

1. **The question, 0–8 seconds:** Show the chosen prompt and an empty next-token slot. State that the film slows down one recorded computation. Never promise human-like thinking.
2. **Text becomes numbers, 8–18 seconds:** Split the sentence into actual tokenizer pieces; show whitespace explicitly. Follow one selected token into its 512-dimensional embedding. A tiled overview represents the vector; zooming reveals indexed numerical cells.
3. **Eight ways to combine context, 18–38 seconds:** Enter the first block. Explain normalization, project into query/key/value vectors, and show rotary encoding at its actual location. Highlight one head, one query position, and the strongest attention edges. Reveal its causal mask and weighted value sum.
4. **Parallel transformation, 38–50 seconds:** Split the camera view between attention and MLP branches. Show the actual GPT-NeoX parallel-residual structure: independently normalized versions of the incoming residual feed attention and MLP; their outputs join the residual. Do not draw a generic attention-then-MLP sequence that misrepresents this model.
5. **Through depth, 50–65 seconds:** Pull back through the other five blocks, with one selected token's measured state highlighted. Use tiled summaries for scale; avoid millions of meaningless particle edges.
6. **The choice, 65–80 seconds:** Apply final normalization and the output projection. Transition from 50,304 vocabulary possibilities to the actual top ten probabilities plus an explicitly labeled remaining probability mass. Greedy selection chooses the highest-probability token.
7. **The loop, 80–90 seconds:** Append that token and briefly repeat for the remaining recorded steps. Finish with the whole architecture and an invitation to inspect any stage.

The timeline supports play/pause, chapter jumps, scrubbing, playback speed, restart, and a reduced-motion mode. A persistent prompt strip, generation-step selector, layer/head selectors, and reset-view control remain available. Clicking an object pauses playback and opens a 2D inspector with its operation, shape, exact indices, values, and source trace. Pointer orbit works only in Explore mode; guided playback owns the camera. Keyboard controls cover every selection. Narrow screens use a fixed camera plus the same inspector, rather than cramped orbit controls.

Motion describes computation order, not measured hardware timing. Particle travel and easing are illustrative; numeric labels always show recorded values. State “illustrative animation; recorded computation” in the information panel and video captions. Attention weights describe weighted aggregation, not a complete explanation of the prediction.

## Implementation and data

Use a TypeScript/Vite application with Three.js for WebGL2 rendering and ordinary HTML/CSS for text, controls, and inspectors. Keep camera choreography and scene state independent of the rendering loop. A pure `sceneAt(time, selection)` function must produce deterministic positions, emphasis, and labels, enabling frame-perfect exports and testing. Use a fixed chapter timeline; do not add an animation framework unless the implementation cannot express a required transition with this function.

Represent matrix tiles and repeated glyph backgrounds with instanced meshes; upload scalar grids through data textures. Draw attention connections in one batched buffer, initially limited to the strongest eight incoming edges for the chosen query/head. A complete 2D attention matrix remains inspectable. Label omitted edges and their remaining mass. Never suggest that displayed top edges comprise the entire calculation. Reuse geometry, limit postprocessing to one inexpensive bloom pass, cap device pixel ratio at 1.5, and render on demand when paused. Overview tiles use labeled aggregates; zoomed cells use actual individual values.

Generate traces in Python using pinned PyTorch and Transformers versions and the model's native GPT-NeoX implementation. Run in evaluation mode, FP32, with dropout disabled, eager attention, and no KV cache. Recomputing each short prefix simplifies faithful full-context extraction. Capture embeddings, layer-normalized inputs, Q/K before and after RoPE, V, attention probabilities, head outputs, attention projection, MLP pre/post-GELU and output, residual outputs, final normalized states, and final logits. Use instrumentation that leaves reference computations unchanged. Do not substitute TransformerLens weight transformations without demonstrating numerical equivalence.

The export interface is a versioned `TraceBundle`. Its manifest records model/tokenizer revisions and hashes, architecture, exact input/token IDs, generated IDs, dtype, library versions, extraction revision, tensor names/shapes/offsets, and checksums. Store arrays as little-endian Float32 binary shards, one generation step per shard, with HTTP compression. Keep only selected vocabulary weight columns needed for the inspector; do not ship full model weights. Preserve full output logits for calculating top probabilities. Browser rendering consumes these bundles and makes no inference calls.

Load one step at a time and retain at most three decoded steps. Prefetch the next step after playback starts. Target an initial compressed download below 12 MB, a complete prompt bundle below 50 MB, and decoded browser memory below 350 MB. If exports exceed these caps, remove redundant intermediate storage that can be deterministically reconstructed and validated, then reduce continuation length. Do not silently downsample values or replace traces with synthetic data.

Publish a static preview with six bundled prompts and an accessible 2D operation/inspector fallback when WebGL2 is unavailable. Include a downloadable manifest and a concise methods page. Keep this deliverable independent of Transformer Time Machine; sharing the tokenizer, model cache, or trace primitives is optional only after each project's contracts are stable.

## Compute, delivery, and acceptance

This is predominantly graphics and product engineering. Model tracing requires forward passes, not pretraining. Begin locally; if Modal is needed, use one L4 worker, cached model files, concurrency one, and a ten-minute per-job timeout. Proposed compute caps are **$75 for the trace pilot and $250 for the entire v1**, including CPU/RAM overhead. These are ceilings, not predicted costs. Record actual billed usage, stop launching work at the cap, and request a new budget only if a concrete blocker requires it. Do not consume the remaining credits merely because they expire. Generate video frames locally or on CPU infrastructure; H100s are unnecessary for rendering this site.

Deliver in four milestones:

- **Trace proof:** One prompt, one next token, immutable provenance, checked tensors, and a plain inspector. Proceed only when instrumentation reproduces the reference output.
- **Visual slice:** Tokens → one attention head → actual next-token probabilities, with the first 20 seconds of camera choreography. Review the visual direction before building all scenes.
- **Complete experience:** All seven beats, six prompts, eight-step continuation, inspector, controls, accessibility, and responsive fallback.
- **Release assets:** Static preview, source/export commands, trace manifests, a 90-second 1920×1080 MP4, and a 30-second 1080×1920 cut. Record timestamps and viewport explicitly; prepare assets before any public release.

Validate instrumented versus uninstrumented logits on all curated prompts and 20 fixed-seed short prompts using maximum absolute error below `1e-4` and identical greedy tokens. Validate tensor dimensions, causal-mask zeros, attention row sums within `1e-5`, branch recombination, full softmax normalization, and checksums. Verify ten inspector values against source tensors per operation type. Failed parity blocks publication; investigate rather than relaxing tolerances automatically.

Browser acceptance requires complete keyboard navigation, correct timeline seeking, recoverable load errors, no missing labels at supported widths, and no fabricated numeric interpolation. Measure the full film at 1440×900 on an Apple-silicon Mac: target median 60 FPS and 95th-percentile frame time under 33 ms. The mobile fallback targets 30 FPS. Capture deterministically by advancing the scene clock at 30 FPS and encode with FFmpeg; the exported film must match the interactive scene at the same timestamps. Add burned-in captions, with no voice cloning or unlicensed soundtrack. Manually inspect the final videos for clipping, unreadable values, and misleading architecture.

## Primary references and attribution

- [Pythia model card](https://huggingface.co/EleutherAI/pythia-70m) and [configuration](https://huggingface.co/EleutherAI/pythia-70m/blob/main/config.json): model, checkpoint, architecture, and license provenance.
- [Brendan Bycroft's LLM visualization](https://github.com/bbycroft/llm-viz): prior art for cinematic 3D transformer explanation. Build an original visual treatment; preserve licenses if adapting code.
- [Transformer Explainer](https://github.com/poloclub/transformer-explainer): prior art for interactive, inspectable model operations.
- [AttentionViz](https://arxiv.org/abs/2305.03210) and [Cynthia Chen's experiments](https://github.com/chenxcynthia/transformers): prior art for attention geometry and its interpretation.
- [Modal pricing](https://modal.com/pricing): recheck rates and credit eligibility before launching paid work.

Credit these projects in the methods page. Present the result as an original cinematic explorable built on established methods, without implying a new interpretability discovery.
