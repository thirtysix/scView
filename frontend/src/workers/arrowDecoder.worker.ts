import { tableFromIPC } from "apache-arrow";

interface DecodeRequest {
  buffer: ArrayBuffer;
  columns?: string[];
}

interface ColumnResult {
  data: ArrayBuffer;
  type: "float32" | "int32" | "uint8" | "unknown";
  length: number;
}

self.onmessage = (event: MessageEvent<DecodeRequest>) => {
  try {
    const { buffer, columns } = event.data;
    const table = tableFromIPC(new Uint8Array(buffer));

    const columnNames =
      columns ?? table.schema.fields.map((f) => f.name);

    const result: Record<string, ColumnResult> = {};
    const transfers: ArrayBuffer[] = [];

    for (const col of columnNames) {
      const column = table.getChild(col);
      if (!column) continue;

      const arr = column.toArray();

      // Determine the typed array type
      let type: ColumnResult["type"] = "unknown";
      if (arr instanceof Float32Array) type = "float32";
      else if (arr instanceof Float64Array) type = "float32"; // we'll convert
      else if (arr instanceof Int32Array) type = "int32";
      else if (arr instanceof Uint8Array) type = "uint8";
      else if (arr instanceof Int8Array) type = "int32";
      else if (arr instanceof Int16Array) type = "int32";

      // Copy data into a transferable buffer.
      // For Float64Array, downcast to Float32Array to save memory.
      let outputBuf: ArrayBuffer;
      let length: number;

      if (arr instanceof Float64Array) {
        const f32 = new Float32Array(arr.length);
        for (let i = 0; i < arr.length; i++) {
          f32[i] = arr[i]!;
        }
        outputBuf = f32.buffer;
        length = f32.length;
        type = "float32";
      } else {
        outputBuf = arr.buffer.slice(
          arr.byteOffset,
          arr.byteOffset + arr.byteLength,
        );
        length = arr.length;
      }

      result[col] = { data: outputBuf, type, length };
      transfers.push(outputBuf);
    }

    (self as unknown as { postMessage(msg: unknown, transfer: Transferable[]): void }).postMessage(
      { type: "success", result, numRows: table.numRows },
      transfers as Transferable[],
    );
  } catch (error) {
    self.postMessage({
      type: "error",
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
