import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { sceneAt, chapters } from "../src/scene.mjs";
test("scene is deterministic and chapter boundaries are stable", () => {
  for (let t = 0; t <= 90; t += 0.25) {
    assert.deepEqual(sceneAt(t, { layer: 2 }), sceneAt(t, { layer: 2 }));
    assert.ok(sceneAt(t, { layer: 2 }).camera.every(Number.isFinite));
  }
  assert.equal(sceneAt(89).loopStep, 7);
  assert.equal(sceneAt(18).chapter, 2);
});
test("every shard verifies hashes, samples, output probabilities and size caps", () => {
  const gallery = JSON.parse(readFileSync("public/data/gallery.json"));
  assert.equal(gallery.length, 6);
  for (const item of gallery) {
    const m = JSON.parse(readFileSync("public" + item.manifest));
    assert.equal(m.steps.length, 8);
    assert.ok(m.steps.reduce((s, v) => s + v.bytes, 0) < 50_000_000);
    for (const step of m.steps) {
      const data = readFileSync("public" + step.file);
      assert.equal(
        createHash("sha256").update(data).digest("hex"),
        step.sha256,
      );
      for (const t of Object.values(step.tensors)) {
        assert.equal(
          t.shape.reduce((a, b) => a * b, 1),
          t.length,
        );
        for (const s of t.samples)
          assert.equal(data.readFloatLE(t.offset + s.index * 4), s.value);
      }
      const logits = step.tensors.logits,
        raw = Array.from({ length: logits.length }, (_, i) =>
          data.readFloatLE(logits.offset + i * 4),
        ),
        max = Math.max(...raw),
        exp = raw.map((v) => Math.exp(v - max)),
        sum = exp.reduce((a, b) => a + b, 0);
      for (const p of step.predictions)
        assert.ok(Math.abs(exp[p.id] / sum - p.p) < 1e-10);
      assert.equal(raw.indexOf(max), step.next_id);
      assert.equal(step.parity_max_abs, 0);
    }
  }
});

test('an old step buffer cannot be read with a new step manifest', async()=>{const {values}=await import('../src/trace.ts');const gallery=JSON.parse(readFileSync('public/data/gallery.json'));const m=JSON.parse(readFileSync('public'+gallery[0].manifest));const bytes=readFileSync('public'+m.steps[0].file);const buffer=bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength);assert.equal(values(buffer,m.steps[0],'logits').length,50304);assert.equal(values(buffer,m.steps[1],'logits').length,0)});
