import { tableFromIPC } from "apache-arrow";

/**
 * Synchronous Arrow IPC decoding — useful for small datasets or when the Web
 * Worker is not available. For large datasets prefer the Web Worker approach
 * via `useEmbedding`.
 */
export function decodeArrowBuffer(
  buffer: ArrayBuffer,
): Record<string, Float32Array | Int32Array | Uint8Array> {
  const table = tableFromIPC(new Uint8Array(buffer));
  const result: Record<string, Float32Array | Int32Array | Uint8Array> = {};

  for (const field of table.schema.fields) {
    const col = table.getChild(field.name);
    if (!col) continue;

    const arr = col.toArray();

    // Downcast Float64 to Float32 for GPU-friendliness
    if (arr instanceof Float64Array) {
      const f32 = new Float32Array(arr.length);
      for (let i = 0; i < arr.length; i++) {
        f32[i] = arr[i]!;
      }
      result[field.name] = f32;
    } else if (
      arr instanceof Float32Array ||
      arr instanceof Int32Array ||
      arr instanceof Uint8Array
    ) {
      result[field.name] = arr;
    } else {
      // Fallback: convert numeric-like arrays to Float32Array
      const f32 = new Float32Array(arr.length);
      for (let i = 0; i < arr.length; i++) {
        f32[i] = Number(arr[i]);
      }
      result[field.name] = f32;
    }
  }

  return result;
}

/**
 * Extract column names from an Arrow IPC buffer without fully decoding all
 * column data — useful for schema inspection.
 */
export function getArrowColumnNames(buffer: ArrayBuffer): string[] {
  const table = tableFromIPC(new Uint8Array(buffer));
  return table.schema.fields.map((f) => f.name);
}
