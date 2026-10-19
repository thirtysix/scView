import { apiFetch } from "./client";

/** The backend's memory budget (GET /system/memory). */
export interface SystemMemory {
  /** What scView may use; null when unknown */
  limit_bytes: number | null;
  used_bytes: number | null;
  available_bytes: number | null;
  /** A container limit (SCVIEW_MEMORY_GB), not the whole machine's memory */
  limited: boolean;
  source: "container limit" | "machine" | "unknown";
}

export function getSystemMemory(): Promise<SystemMemory> {
  return apiFetch<SystemMemory>("/system/memory");
}
