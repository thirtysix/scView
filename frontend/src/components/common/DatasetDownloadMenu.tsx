import { useCallback, useEffect, useState } from "react";

import {
  downloadRecipeFile,
  downloadTable,
  downloadUrl,
  getH5adInfo,
  h5adDownloadUrl,
  type H5adInfo,
} from "@/api/export";
import { ExportMenu, type ExportMenuItem } from "@/components/common/ExportMenu";
import { formatBytes } from "@/lib/formatting";

interface DatasetDownloadMenuProps {
  datasetId: string;
  /** Icon-only button, for the library rows */
  compact?: boolean;
  className?: string;
}

/**
 * "Download" menu for one dataset: the .h5ad scView works on (size shown before
 * the download starts), the cell metadata, the marker tables, and the replay
 * script + run manifest of its recorded steps.
 */
export function DatasetDownloadMenu({ datasetId, compact = false, className }: DatasetDownloadMenuProps) {
  const [info, setInfo] = useState<H5adInfo | null>(null);
  const [infoError, setInfoError] = useState(false);

  useEffect(() => {
    setInfo(null);
    setInfoError(false);
  }, [datasetId]);

  // Fetched on every open: processing replaces the file, so the size changes.
  const loadInfo = useCallback(() => {
    getH5adInfo(datasetId)
      .then((i) => {
        setInfo(i);
        setInfoError(false);
      })
      .catch(() => setInfoError(true));
  }, [datasetId]);

  const h5adHint = info
    ? `${info.source_label.charAt(0).toUpperCase()}${info.source_label.slice(1)} · ${info.large ? "about " : ""}${formatBytes(info.size_bytes)}`
    : infoError
      ? "No .h5ad file available yet"
      : "Checking size…";

  const items: ExportMenuItem[] = [
    {
      key: "h5ad",
      icon: "data",
      label: "Dataset (.h5ad)",
      hint: h5adHint,
      disabled: !info,
      onSelect: () => {
        if (info) downloadUrl(h5adDownloadUrl(datasetId), info.filename);
      },
    },
    {
      key: "metadata-csv",
      icon: "csv",
      label: "Cell metadata (.csv)",
      hint: "Every obs column, one row per cell",
      dividerBefore: true,
      onSelect: () => downloadTable(datasetId, "metadata", "csv"),
    },
    {
      key: "markers-csv",
      icon: "csv",
      label: "Marker genes (.csv)",
      hint: "Every grouping with markers, top 100 per group",
      onSelect: () => downloadTable(datasetId, "markers", "csv"),
    },
    {
      key: "markers-xlsx",
      icon: "xlsx",
      label: "Marker genes (.xlsx)",
      hint: "One sheet per grouping",
      onSelect: () => downloadTable(datasetId, "markers", "xlsx"),
    },
    {
      key: "replay-script",
      icon: "csv",
      label: "Replay script (.py)",
      hint: "Python (scanpy) that repeats the recorded steps",
      dividerBefore: true,
      onSelect: () => downloadRecipeFile(datasetId, "script"),
    },
    {
      key: "run-manifest",
      icon: "csv",
      label: "Run manifest (.json)",
      hint: "Versions, input checksum and every step",
      onSelect: () => downloadRecipeFile(datasetId, "manifest"),
    },
  ];

  return (
    <ExportMenu
      items={items}
      label="Download"
      onOpen={loadInfo}
      compact={compact}
      className={className}
    />
  );
}
