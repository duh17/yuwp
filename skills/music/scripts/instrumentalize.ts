#!/usr/bin/env bun
// Turn a YuE2 two-voice ABC score into an instrumental score: the Vocal line keeps its chord
// symbols but every note becomes a rest of the same length, and sung melody moves to the Ins
// line wherever Ins was resting. Pair the result with sections-only lyrics and --abc-file.
//
// Usage: bun instrumentalize.ts <score.abc> <out.abc>

import { existsSync, readFileSync, writeFileSync } from "node:fs";

const [input, output] = process.argv.slice(2);
if (!input || !output) {
  console.error("usage: bun instrumentalize.ts <score.abc> <out.abc>");
  process.exit(2);
}
if (existsSync(output)) {
  console.error(`refusing to overwrite ${output}`);
  process.exit(1);
}

// Quoted chord symbols and inline fields ([K:..], [M:..]) pass through untouched.
const PROTECTED = /("[^"]*"|\[[A-Za-z]:[^\]]*\])/;
const NOTE = /(?:\^\^|__|\^|_|=)?[A-Ga-g][',]*(\d*)-?/g;

function mapUnprotected(line: string, fn: (text: string) => string): string {
  return line
    .split(PROTECTED)
    .map((part, i) => (i % 2 === 1 ? part : fn(part)))
    .join("");
}

/** Same rhythm, chords kept, every note silenced. */
function silence(line: string): string {
  return mapUnprotected(line, (text) => text.replace(NOTE, (_m, dur: string) => `z${dur}`));
}

/** Notes only: chord symbols dropped, inline fields kept. */
function melodyOnly(line: string): string {
  return line.replace(/"[^"]*"/g, "");
}

function hasNotes(line: string): boolean {
  const note = new RegExp(NOTE.source);
  return line.split(PROTECTED).some((part, i) => i % 2 === 0 && note.test(part));
}

const lines = readFileSync(input, "utf8").split("\n");
let current: "Vocal" | "Ins" | null = null;
let pendingVocal: { index: number; line: string } | null = null;
let moved = 0;
let silenced = 0;
const insBlocks: { index: number; moved: boolean }[] = [];

for (let i = 0; i < lines.length; i++) {
  const line = lines[i];
  const voice = /^V:\s*(Vocal|Ins)\s*$/.exec(line.trim());
  if (voice) {
    current = voice[1] as "Vocal" | "Ins";
    continue;
  }
  if (line.startsWith("%") || /^[A-Za-z]:/.test(line) || line.trim() === "" || current === null) continue;
  if (current === "Vocal") {
    pendingVocal = { index: i, line };
    if (hasNotes(line)) silenced++;
    lines[i] = silence(line);
  } else {
    const move = pendingVocal !== null && hasNotes(pendingVocal.line) && !hasNotes(line);
    if (move && pendingVocal) {
      lines[i] = melodyOnly(pendingVocal.line);
      moved++;
    }
    insBlocks.push({ index: i, moved: move });
    pendingVocal = null;
  }
  current = null;
}

// A moved melody tied into a block whose Ins was kept would tie into a rest or a different
// note: end it at the block boundary instead.
insBlocks.forEach((block, n) => {
  if (block.moved && !insBlocks[n + 1]?.moved) {
    lines[block.index] = lines[block.index].replace(/-(\|?\s*)$/, "$1");
  }
});

writeFileSync(output, lines.join("\n"));
console.log(JSON.stringify({ output, vocal_blocks_silenced: silenced, blocks_moved_to_ins: moved }));
