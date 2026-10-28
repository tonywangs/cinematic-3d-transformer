import http from "node:http";
import { spawn } from "node:child_process";
import { mkdir } from "node:fs/promises";
import path from "node:path";
let encoder = null,
  finished = null;
await mkdir("exports", { recursive: true });
const server = http.createServer(async (req, res) => {
  try {
    const origin = req.headers.origin;
    if (origin && !/^http:\/\/(127\.0\.0\.1|localhost):5174$/.test(origin)) {
      res.writeHead(403).end();
      return;
    }
    if (req.url === "/export/start" && req.method === "POST") {
      if (encoder) {
        res.writeHead(409).end();
        return;
      }
      let body = "";
      for await (const chunk of req) body += chunk;
      const { width, height, fps, name } = JSON.parse(body);
      if (
        ![
          [1920, 1080],
          [1080, 1920],
        ].some(([w, h]) => w === width && h === height) ||
        fps !== 30 ||
        !["film", "portrait"].includes(name)
      )
        throw Error("Invalid export");
      encoder = spawn(
        "ffmpeg",
        [
          "-y",
          "-f",
          "image2pipe",
          "-framerate",
          "30",
          "-i",
          "pipe:0",
          "-c:v",
          "libx264",
          "-preset",
          "fast",
          "-crf",
          "19",
          "-pix_fmt",
          "yuv420p",
          "-movflags",
          "+faststart",
          path.resolve("exports", name + ".mp4"),
        ],
        { stdio: ["pipe", "ignore", "inherit"] },
      );
      finished = new Promise((resolve, reject) => {
        encoder.on("exit", (code) =>
          code === 0 ? resolve() : reject(Error("ffmpeg failed")),
        );
        encoder.on("error", reject);
      });
      res.end("ready");
    } else if (
      req.url === "/export/frame" &&
      req.method === "POST" &&
      encoder
    ) {
      const chunks = [];
      let size = 0;
      for await (const chunk of req) {
        size += chunk.length;
        if (size > 20_000_000) throw Error("Frame too large");
        chunks.push(chunk);
      }
      await new Promise((resolve, reject) =>
        encoder.stdin.write(Buffer.concat(chunks), (e) =>
          e ? reject(e) : resolve(),
        ),
      );
      res.end("ok");
    } else if (
      req.url === "/export/finish" &&
      req.method === "POST" &&
      encoder
    ) {
      encoder.stdin.end();
      await finished;
      encoder = null;
      res.end("saved");
    } else res.writeHead(404).end();
  } catch (e) {
    console.error(e);
    res.writeHead(500).end("Export failed");
  }
});
server.listen(5175, "127.0.0.1", () =>
  console.log("Frame encoder: http://127.0.0.1:5175"),
);
