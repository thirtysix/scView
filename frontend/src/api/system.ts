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
  /** Set when the previous backend was killed (most often out of memory) and restarted */
  unclean_restart?: { restarted_at: string; previous_start: string | null } | null;
}

export function getSystemMemory(): Promise<SystemMemory> {
  return apiFetch<SystemMemory>("/system/memory");
}
