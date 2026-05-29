import { useState, useRef, useEffect, useCallback } from "react";
import { Download, FileSpreadsheet, FileText, Image, ChevronDown } from "lucide-react";

interface ExportMenuProps {
  onExportCsv?: () => void;
  onExportXlsx?: () => void;
  onExportPng?: () => void;
  disabled?: boolean;
  className?: string;
}

/**
 * Dropdown menu that exposes CSV, Excel, and PNG export actions via callback
 * props.  Each callback is optional — buttons whose callbacks are not provided
 * will be hidden.
 */
export function ExportMenu({
  onExportCsv,
  onExportXlsx,
  onExportPng,
  disabled = false,
  className = "",
}: ExportMenuProps) {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const handleAction = useCallback(
    (action: (() => void) | undefined) => {
      if (action) action();
      setIsOpen(false);
    },
    [],
  );

  const hasAnyAction = !!(onExportCsv || onExportXlsx || onExportPng);

  if (!hasAnyAction) return null;

  return (
    <div ref={menuRef} className={`relative inline-block ${className}`}>
      <button
        onClick={() => setIsOpen((prev) => !prev)}
        disabled={disabled}
        className="flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm transition-colors hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
      >
        <Download className="h-4 w-4" />
        Export
        <ChevronDown className="h-3.5 w-3.5" />
      </button>

      {isOpen && (
        <div className="absolute right-0 z-50 mt-1 min-w-[180px] rounded-lg border border-slate-200 bg-white py-1 shadow-lg">
          {onExportCsv && (
            <button
              onClick={() => handleAction(onExportCsv)}
              className="flex w-full items-center gap-2.5 px-4 py-2 text-left text-sm text-slate-700 transition-colors hover:bg-slate-50"
            >
              <FileText className="h-4 w-4 text-slate-400" />
              Export as CSV
            </button>
          )}
          {onExportXlsx && (
            <button
              onClick={() => handleAction(onExportXlsx)}
              className="flex w-full items-center gap-2.5 px-4 py-2 text-left text-sm text-slate-700 transition-colors hover:bg-slate-50"
            >
              <FileSpreadsheet className="h-4 w-4 text-emerald-500" />
              Export as Excel
            </button>
          )}
          {onExportPng && (
            <>
              <div className="mx-3 my-1 border-t border-slate-100" />
              <button
                onClick={() => handleAction(onExportPng)}
                className="flex w-full items-center gap-2.5 px-4 py-2 text-left text-sm text-slate-700 transition-colors hover:bg-slate-50"
              >
                <Image className="h-4 w-4 text-blue-500" />
                Export Plot as PNG
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}
