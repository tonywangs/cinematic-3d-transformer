import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { World } from "./world";
import { chapters, sceneAt } from "./scene.mjs";
import { loadStep, values, type Manifest } from "./trace";
import { exportFilm } from "./video";
import "./style.css";
type Item = { id: string; title: string; prompt: string; manifest: string };
const visible = (s: string) => s.replaceAll(" ", "·").replaceAll("\n", "↵");
const descriptions: Record<string, string> = {
  embedding: "Token ID → learned 512-dimensional vector.",
  attention_norm: "Normalization of the incoming residual, feeding attention.",
  q_before_rope: "Queries before rotary position encoding.",
  k_before_rope: "Keys before rotary position encoding.",
  q: "Queries after RoPE: position rotates the first 16 of 64 dimensions per head.",
  k: "Keys after RoPE. Attention uses scaled query–key dot products.",
  v: "Values: the vectors combined by attention weights.",
  attention: "Softmax of scaled query–key scores. Future positions are masked.",
  heads: "Weighted value sums, concatenated across eight heads.",
  attention_out: "Concatenated heads after the attention output projection.",
  mlp_norm:
    "Independent normalization of the same incoming residual, feeding the parallel MLP.",
  mlp_pre: "First MLP projection: 512 → 2,048 dimensions.",
  mlp_post: "GELU applied to the first MLP projection.",
  mlp_out: "Second MLP projection: 2,048 → 512 dimensions.",
  residual_in: "Incoming stream shared by attention and MLP.",
  residual_out: "Incoming residual + attention output + MLP output.",
  final_norm: "Final normalization before the vocabulary projection.",
  logits:
    "50,304 raw output scores. Exported probabilities use float64 softmax of recorded float32 logits.",
};
function App() {
  const [items, setItems] = useState<Item[]>([]),
    [item, setItem] = useState<Item | null>(null),
    [manifest, setManifest] = useState<Manifest | null>(null),
    [loaded, setLoaded] = useState<{file:string;buffer:ArrayBuffer}|null>(null),
    [step, setStep] = useState(0),
    [layer, setLayer] = useState(0),
    [head, setHead] = useState(0),
    [token, setToken] = useState(0),
    [operation, setOperation] = useState("embedding"),
    [index, setIndex] = useState(0),
    [time, setTime] = useState(0),
    [playing, setPlaying] = useState(false),
    [speed, setSpeed] = useState(1),
    [explore, setExplore] = useState(false),
    [reduced, setReduced] = useState(
      matchMedia("(prefers-reduced-motion:reduce)").matches,
    ),
    [error, setError] = useState(""),
    [loading, setLoading] = useState("Loading traces…"),
    [notes, setNotes] = useState(false),
    [fallback, setFallback] = useState(false),
    [exporting, setExporting] = useState(""),
    [exportBusy, setExportBusy] = useState(false),
    [metrics, setMetrics] = useState<{
      frames: number;
      medianFPS: number;
      p95ms: number;
    } | null>(null);
  const container = useRef<HTMLDivElement>(null),
    world = useRef<World | null>(null),
    ticket = useRef(0),
    clock = useRef(0),
    lastFrame = useRef(0),
    frameTimes = useRef<number[]>([]);
  const current = manifest?.steps[step],
    buffer = loaded?.file === current?.file ? loaded?.buffer ?? null : null,
    scene = sceneAt(time, { layer });
  useEffect(() => {
    fetch("/data/gallery.json")
      .then((r) => r.json())
      .then((data) => {
        setItems(data);
        setItem(data[0]);
      })
      .catch(() => {
        setError("Examples could not load. Reload to retry.");
        setLoading("");
      });
  }, []);
  useEffect(() => {
    if (!item) return;
    let alive = true;
    setPlaying(false);
    setLoading("Opening trace…");
    setError("");
    fetch(item.manifest)
      .then((r) => {
        if (!r.ok) throw Error("Manifest unavailable.");
        return r.json();
      })
      .then((m) => {
        if (alive) {
          setManifest(m);
          setStep(0);
          setToken(m.steps[0].tokens.length - 1);
          setTime(0);
          clock.current = 0;
        }
      })
      .catch((e) => {
        if (alive) {
          setError(String(e));
          setLoading("");
        }
      });
    return () => {
      alive = false;
    };
  }, [item]);
  useEffect(() => {
    if (!current) return;
    const id = ++ticket.current;
    setLoading("Reading recorded tensors…");
    setLoaded(null);
    loadStep(current)
      .then((b) => {
        if (ticket.current === id) {
          setLoaded({file:current.file,buffer:b});
          setToken((t) => Math.min(t, current.tokens.length - 1));
          setLoading("");
        }
      })
      .catch((e) => {
        if (ticket.current === id) {
          setError(String(e));
          setLoading("");
          setPlaying(false);
        }
      });
  }, [current]);
  useEffect(() => {
    if (!container.current) return;
    try {
      world.current = new World(container.current, (l, op) => {
        setLayer(l);
        setOperation(op);
        setPlaying(false);
        setExplore(true);
      });
    } catch {
      setFallback(true);
    }
    return () => world.current?.dispose();
  }, []);
  useEffect(() => {
    if (buffer && current)
      world.current?.setData(buffer, current, { layer, head, token });
  }, [buffer, current, layer, head, token]);
  useEffect(() => {
    world.current?.setTime(
      reduced ? chapters[scene.chapter].at + 3 : time,
      explore,
    );
  }, [time, explore, reduced]);
  useEffect(() => {
    if (!playing || !buffer) return;
    lastFrame.current = performance.now();
    let raf = 0;
    const frame = (now: number) => {
      frameTimes.current.push(now - lastFrame.current);
      clock.current = Math.min(
        90,
        clock.current + ((now - lastFrame.current) / 1000) * speed,
      );
      lastFrame.current = now;
      setTime(clock.current);
      if (clock.current >= 90) {
        setPlaying(false);
        const sorted = [...frameTimes.current].sort((a, b) => a - b);
        setMetrics({
          frames: sorted.length,
          medianFPS: 1000 / sorted[Math.floor(sorted.length * 0.5)],
          p95ms: sorted[Math.floor(sorted.length * 0.95)],
        });
      } else raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    if (manifest && step < 7)
      loadStep(manifest.steps[step + 1]).catch(() => {});
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, buffer]);
  useEffect(() => {
    if (playing) {
      setOperation(scene.operation);
      if (scene.chapter === 4) setLayer(scene.layer);
    }
  }, [playing, scene.operation, scene.layer, scene.chapter]);
  useEffect(() => {
    if (playing && scene.chapter === 6) setStep(scene.loopStep);
  }, [scene.loopStep, scene.chapter, playing]);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).matches("input,select,textarea,button"))
        return;
      if (e.code === "Space") {
        e.preventDefault();
        setPlaying((p) => !p);
      }
      if (e.key === "Escape") {
        setNotes(false);
        setPlaying(false);
      }
    };
    document.addEventListener("keydown", key);
    return () => document.removeEventListener("keydown", key);
  }, []);
  const seek = (t: number) => {
    if (t === 0) {
      frameTimes.current = [];
      setMetrics(null);
    }
    clock.current = t;
    setTime(t);
    setExplore(false);
    setStep(t >= 80 ? sceneAt(t, { layer }).loopStep : 0);
  };
  const name = ["embedding", "final_norm", "logits"].includes(operation)
      ? operation
      : `layer.${layer}.${operation}`,
    tensor = current?.tensors[name],
    array =
      buffer && current ? values(buffer, current, name) : new Float32Array(),
    safeIndex = Math.min(index, Math.max(0, array.length - 1));
  const coordinates = (flat: number) => {
    const shape = tensor?.shape || [];
    return shape
      .map((_, i) => {
        const stride = shape.slice(i + 1).reduce((a, b) => a * b, 1);
        const v = Math.floor(flat / stride);
        flat %= stride;
        return v;
      })
      .join(", ");
  };
  const matrix =
      buffer && current
        ? values(buffer, current, `layer.${layer}.attention`)
        : new Float32Array(),
    n = current?.tokens.length || 0,
    weights = Array.from(
      matrix.slice(head * n * n + token * n, head * n * n + (token + 1) * n),
    ),
    top = weights
      .map((p, i) => ({ p, i }))
      .filter((v) => v.i <= token)
      .sort((a, b) => b.p - a.p)
      .slice(0, 8),
    omitted = Math.max(0, 1 - top.reduce((a, b) => a + b.p, 0));
  async function film(portrait: boolean) {
    if (!manifest || !world.current || exportBusy) return;
    setExportBusy(true);
    setPlaying(false);
    try {
      await exportFilm(world.current, manifest, portrait, setExporting);
    } catch (e) {
      setExporting(String(e));
    } finally {
      setExportBusy(false);
      const w = world.current,
        r = container.current!.getBoundingClientRect();
      w.renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
      w.renderer.setSize(r.width, r.height);
      w.camera.aspect = r.width / r.height;
      w.camera.updateProjectionMatrix();
      if (buffer && current) w.setData(buffer, current, { layer, head, token });
      w.setTime(time, false);
    }
  }
  return (
    <>
      <header>
        <a href="/" className="brand">
          <span>◈</span> Cinematic Transformer
        </a>
        <div>
          <span className="model">Pythia 70M</span>
          <button onClick={() => setNotes(true)}>Notes</button>
        </div>
      </header>
      <main>
        <section className="stage">
          <div className="stage-top">
            <label className="sr-only" htmlFor="preset">
              Example
            </label>
            <select
              id="preset"
              value={item?.id || ""}
              onChange={(e) =>
                setItem(items.find((i) => i.id === e.target.value)!)
              }
            >
              {items.map((i) => (
                <option key={i.id} value={i.id}>
                  {i.title}
                </option>
              ))}
            </select>
            <span>06 blocks · 08 heads · 512 dimensions</span>
            <button
              onClick={() => {
                setExplore((v) => !v);
                setPlaying(false);
              }}
            >
              {explore ? "Return to guided ↗" : "Explore ↗"}
            </button>
          </div>
          <div className="chapter-copy">
            <small>{String(scene.chapter + 1).padStart(2, "0")} / 07</small>
            <h1>{scene.title}</h1>
            <p>{scene.caption}</p>
          </div>
          <div ref={container} className="world" />
          {fallback && (
            <div className="fallback">
              3D unavailable. All recorded tensors remain available in the
              inspector.
            </div>
          )}
          <div className="legend">
            <span>— Attention</span>
            <span>— MLP</span>
            <span>— Residual</span>
            <small>Signed tiles: coral − / blue +</small>
          </div>
          <div className="prompt-strip">
            {current?.tokens.map((t, i) => (
              <button
                key={i}
                className={i === token ? "selected" : ""}
                onClick={() => {
                  setToken(i);
                  setPlaying(false);
                }}
                title={`Token ${i} · ID ${current.token_ids[i]}`}
              >
                {visible(t)}
              </button>
            ))}
            {current && (
              <span className="next">→ {visible(current.next_token)}</span>
            )}
          </div>
          <div className="load-status" role="status">
            {loading && <i />}
            {loading || error}
            {error && (
              <button onClick={() => setItem(item ? { ...item } : null)}>
                Retry
              </button>
            )}
          </div>
        </section>
        <aside className="inspector">
          <div className="inspector-heading">
            The evidence <span>FP32</span>
          </div>
          <div className="selectors">
            <label>
              Layer
              <select
                value={layer}
                onChange={(e) => {
                  setLayer(+e.target.value);
                  setPlaying(false);
                }}
              >
                {Array.from({ length: 6 }, (_, i) => (
                  <option key={i} value={i}>
                    {i + 1}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Head
              <select
                value={head}
                onChange={(e) => {
                  setHead(+e.target.value);
                  setPlaying(false);
                }}
              >
                {Array.from({ length: 8 }, (_, i) => (
                  <option key={i} value={i}>
                    {i + 1}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Step
              <select
                value={step}
                onChange={(e) => {
                  setStep(+e.target.value);
                  setPlaying(false);
                }}
              >
                {manifest?.steps.map((_, i) => (
                  <option key={i} value={i}>
                    {i + 1} / 8
                  </option>
                ))}
              </select>
            </label>
          </div>
          <label className="operation-label">
            Operation
            <select
              value={operation}
              onChange={(e) => {
                setOperation(e.target.value);
                setIndex(0);
                setPlaying(false);
              }}
            >
              {Object.keys(descriptions).map((op) => (
                <option key={op}>{op}</option>
              ))}
            </select>
          </label>
          <p className="description">{descriptions[operation]}</p>
          <div className="tensor-meta">
            <span>{tensor?.shape.join(" × ")}</span>
            <span>{tensor?.length.toLocaleString()} values</span>
          </div>
          <div className="cells" aria-label="First 128 values">
            {Array.from(array.slice(0, 128)).map((v, i) => (
              <button
                key={i}
                title={`[${coordinates(i)}] = ${v}`}
                style={{
                  background:
                    v < 0
                      ? `rgba(224,148,118,${0.1 + 0.8 * Math.min(1, Math.abs(v) / 3)})`
                      : `rgba(116,198,218,${0.1 + 0.8 * Math.min(1, Math.abs(v) / 3)})`,
                }}
                onClick={() => setIndex(i)}
                aria-label={`Inspect flat index ${i}`}
              />
            ))}
          </div>
          <div className="cell-caption">
            First 128 cells · full tensor below
          </div>
          <label className="index-field">
            Flat index
            <input
              type="number"
              min="0"
              max={Math.max(0, array.length - 1)}
              value={safeIndex}
              onChange={(e) => setIndex(Math.max(0, +e.target.value))}
            />
          </label>
          <div className="exact">
            <span>[{coordinates(safeIndex)}]</span>
            <strong>{array[safeIndex]?.toPrecision(9) || "—"}</strong>
          </div>
          <details>
            <summary>
              Attention matrix · L{layer + 1} H{head + 1}
            </summary>
            <div
              className="matrix"
              style={{ gridTemplateColumns: `repeat(${n},1fr)` }}
            >
              {Array.from(matrix.slice(head * n * n, (head + 1) * n * n)).map(
                (v, i) => (
                  <button
                    key={i}
                    style={{
                      background: `rgba(116,198,218,${0.06 + 0.94 * v})`,
                    }}
                    title={`${Math.floor(i / n)} → ${i % n}: ${v}`}
                    aria-label={`Query ${Math.floor(i / n)}, key ${i % n}, weight ${v}`}
                    onClick={() => {
                      setOperation("attention");
                      setIndex(head * n * n + i);
                    }}
                  />
                ),
              )}
            </div>
            <p className="cell-caption">
              3D edges: strongest eight · {(omitted * 100).toFixed(2)}% omitted
              for query {token}.
            </p>
          </details>
          <section className="probabilities">
            <div className="inspector-heading">
              Next token <span>final position</span>
            </div>
            {current?.predictions.map((p) => (
              <div key={p.id} className="probability">
                <span>{visible(p.text)}</span>
                <i>
                  <b
                    style={{
                      width: `${(p.p / current.predictions[0].p) * 100}%`,
                    }}
                  />
                </i>
                <small>{(p.p * 100).toFixed(2)}%</small>
              </div>
            ))}
            <p className="cell-caption">
              Other {(100 * (current?.remaining_mass || 0)).toFixed(2)}% ·
              greedy selection
            </p>
          </section>
        </aside>
      </main>
      <footer>
        <div className="chapters">
          {chapters.map((c, i) => (
            <button
              key={c.at}
              className={scene.chapter === i ? "active" : ""}
              onClick={() => {
                seek(c.at === 0 ? 0 : c.at + 2.5);
                setPlaying(false);
              }}
            >
              <small>0{i + 1}</small>
              {
                [
                  "Prompt",
                  "Embedding",
                  "Attention",
                  "Parallel",
                  "Depth",
                  "Prediction",
                  "Repeat",
                ][i]
              }
            </button>
          ))}
        </div>
        <div className="transport">
          <button
            className="play"
            disabled={!!loading || !buffer}
            onClick={() => {
              if (time >= 90) seek(0);
              setExplore(false);
              setPlaying(!playing);
            }}
          >
            {playing ? "Ⅱ Pause" : "▶ Play film"}
          </button>
          <button
            onClick={() => {
              seek(0);
              setPlaying(false);
            }}
            aria-label="Restart"
          >
            ↺
          </button>
          <span className="timestamp">
            {Math.floor(time).toString().padStart(2, "0")} / 90s
          </span>
          <input
            aria-label="Film timeline"
            type="range"
            min="0"
            max="90"
            step=".01"
            value={time}
            onChange={(e) => {
              seek(+e.target.value);
              setPlaying(false);
            }}
          />
          <select
            aria-label="Playback speed"
            value={speed}
            onChange={(e) => setSpeed(+e.target.value)}
          >
            {[0.5, 1, 1.5, 2].map((v) => (
              <option key={v} value={v}>
                {v}×
              </option>
            ))}
          </select>
          <label className="motion">
            <input
              type="checkbox"
              checked={reduced}
              onChange={(e) => setReduced(e.target.checked)}
            />{" "}
            Less motion
          </label>
        </div>
      </footer>
      {notes && (
        <div className="overlay" onClick={() => setNotes(false)}>
          <section
            role="dialog"
            aria-modal="true"
            aria-label="Methods"
            onClick={(e) => e.stopPropagation()}
          >
            <button className="close" autoFocus onClick={() => setNotes(false)}>
              Close ×
            </button>
            <h2>A recorded computation.</h2>
            <p>
              All values come from native Pythia-70M at its final checkpoint.
              Animation describes computation order, not hardware timing.
              Attention is weighted aggregation, not a complete explanation.
            </p>
            <p>
              Attention and the MLP run in parallel. Signed tiles use blue for
              positive and coral for negative values, scaled per tile. The
              inspector shows exact values.
            </p>
            <p>
              Downloaded shards are verified by SHA-256 and ten source values
              per tensor. Eight greedy steps are recorded per example. No
              inference runs on this website.
            </p>
            <p>
              Inspired by{" "}
              <a href="https://github.com/bbycroft/llm-viz">Brendan Bycroft</a>,{" "}
              <a href="https://github.com/poloclub/transformer-explainer">
                Transformer Explainer
              </a>{" "}
              and <a href="https://arxiv.org/abs/2305.03210">AttentionViz</a>.
              Model by{" "}
              <a href="https://huggingface.co/EleutherAI/pythia-70m">
                EleutherAI
              </a>
              .
            </p>
            <code>{manifest?.revision}</code>
            {metrics && (
              <p className="performance">
                Playback check: {metrics.frames} frames · median{" "}
                {metrics.medianFPS.toFixed(1)} FPS · p95{" "}
                {metrics.p95ms.toFixed(2)} ms.
              </p>
            )}
            <p>
              <a href={item?.manifest} download>
                Trace manifest ↓
              </a>{" "}
              ·{" "}
              <a href={current?.file} download>
                FP32 shard ↓
              </a>
            </p>
            <div className="exports">
              <button disabled={exportBusy} onClick={() => film(false)}>
                Export 90s film
              </button>
              <button disabled={exportBusy} onClick={() => film(true)}>
                Export portrait
              </button>
              <span role="status">{exporting}</span>
            </div>
            <p>MP4 export requires the local export server.</p>
          </section>
        </div>
      )}
    </>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
