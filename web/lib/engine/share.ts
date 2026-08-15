/**
 * Encoding a designed board into a URL.
 *
 * A board is 32x24 = 768 cells, which is a 768-bit bitmap: 96 bytes, or 128
 * base64url characters. That is short enough to paste into a message and means
 * a shared level needs no database row — the link *is* the level. The gallery
 * stores boards people choose to publish; sharing one privately costs nothing.
 */
import type { GridSize, Point } from './types.ts';

const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_';

function toBase64Url(bytes: Uint8Array): string {
  let out = '';
  for (let i = 0; i < bytes.length; i += 3) {
    const chunk = (bytes[i] << 16) | ((bytes[i + 1] ?? 0) << 8) | (bytes[i + 2] ?? 0);
    out += ALPHABET[(chunk >> 18) & 63] + ALPHABET[(chunk >> 12) & 63];
    if (i + 1 < bytes.length) out += ALPHABET[(chunk >> 6) & 63];
    if (i + 2 < bytes.length) out += ALPHABET[chunk & 63];
  }
  return out;
}

function fromBase64Url(text: string): Uint8Array {
  const values = [...text].map((character) => ALPHABET.indexOf(character));
  if (values.some((value) => value < 0)) throw new Error('board code contains invalid characters');

  const bytes: number[] = [];
  for (let i = 0; i < values.length; i += 4) {
    const chunk =
      (values[i] << 18) | ((values[i + 1] ?? 0) << 12) | ((values[i + 2] ?? 0) << 6) | (values[i + 3] ?? 0);
    bytes.push((chunk >> 16) & 255);
    if (i + 2 < values.length) bytes.push((chunk >> 8) & 255);
    if (i + 3 < values.length) bytes.push(chunk & 255);
  }
  return Uint8Array.from(bytes);
}

export function encodeBoard(cells: Iterable<Point>, grid: GridSize): string {
  const bits = new Uint8Array(Math.ceil((grid.cols * grid.rows) / 8));
  for (const cell of cells) {
    if (cell.x < 0 || cell.x >= grid.cols || cell.y < 0 || cell.y >= grid.rows) continue;
    const index = cell.y * grid.cols + cell.x;
    bits[index >> 3] |= 1 << (index & 7);
  }
  return toBase64Url(bits);
}

export function decodeBoard(code: string, grid: GridSize): Point[] {
  const bits = fromBase64Url(code);
  const expected = Math.ceil((grid.cols * grid.rows) / 8);
  if (bits.length < expected) {
    throw new Error(`board code is too short for a ${grid.cols}x${grid.rows} grid`);
  }

  const cells: Point[] = [];
  for (let index = 0; index < grid.cols * grid.rows; index += 1) {
    if (bits[index >> 3] & (1 << (index & 7))) {
      cells.push({ x: index % grid.cols, y: Math.floor(index / grid.cols) });
    }
  }
  return cells;
}
