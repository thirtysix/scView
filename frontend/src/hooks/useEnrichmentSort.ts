import { useCallback, useMemo, useState } from "react";
import type { EnrichmentResult } from "@/api/enrichment";
import {
  nextSort,
  sortEnrichmentResults,
  type EnrichmentSortField,
  type SortDir,
} from "@/lib/enrichment";

/** Sort state for an enrichment terms table, most significant first by default. */
export function useEnrichmentSort(results: EnrichmentResult[]) {
  const [sort, setSort] = useState<{ field: EnrichmentSortField; dir: SortDir }>({
    field: "adj_pval",
    dir: "asc",
  });
  const sorted = useMemo(
    () => sortEnrichmentResults(results, sort.field, sort.dir),
    [results, sort],
  );
  const onSort = useCallback((field: EnrichmentSortField) => setSort((s) => nextSort(s, field)), []);
  return { sorted, sortField: sort.field, sortDir: sort.dir, onSort };
}
