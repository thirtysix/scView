import { useState, useRef, useEffect, useCallback, type KeyboardEvent } from "react";
import {
  ChevronDown,
  Database,
  Download,
  FileSpreadsheet,
  FileText,
  Image,
  Loader2,
} from "lucide-react";

const ICONS = {
  csv: <FileText className="h-4 w-4 text-slate-400" />,
  xlsx: <FileSpreadsheet className="h-4 w-4 text-emerald-500" />,
  image: <Image className="h-4 w-4 text-blue-500" />,
  data: <Database className="h-4 w-4 text-violet-500" />,
} as const;

function enabledItems(menu: HTMLElement | null): HTMLButtonElement[] {
  return Array.from(menu?.querySelectorAll<HTMLButtonElement>('[role="menuitem"]:not(:disabled)') ?? []);
}

export interface ExportMenuItem {
  key: string;
  label: string;
  /** Secondary text under the label, e.g. a file size */
  hint?: string;
  icon?: keyof typeof ICONS;
  /** Draw a separator above this item */
  dividerBefore?: boolean;
  disabled?: boolean;
  /** A returned promise keeps the menu open with a spinner; a rejection shows its message. */
  onSelect: () => void | Promise<void>;
}

interface ExportMenuProps {
  items: ExportMenuItem[];
  /** Button text (default "Export") */
  label?: string;
  /** Called each time the menu opens, e.g. to fetch a file size lazily */
  onOpen?: () => void;
  /** Called when the menu opens or closes (e.g. to lift its container above an overlay) */
  onOpenChange?: (open: boolean) => void;
  disabled?: boolean;
  /** Small icon-only button for dense rows */
  compact?: boolean;
  className?: string;
}

/**
 * Dropdown of export/download actions. Items with async actions stay open
 * while they run and show an error in place if they fail, so the caller
 * doesn't need its own error banner.
 */
export function ExportMenu({
  items,
  label = "Export",
  onOpen,
  onOpenChange,
  disabled = false,
  compact = false,
  className = "",
}: ExportMenuProps) {
  const [isOpen, setIsOpen] = useState(false);
  const onOpenChangeRef = useRef(onOpenChange);
  onOpenChangeRef.current = onOpenChange;
  useEffect(() => onOpenChangeRef.current?.(isOpen), [isOpen]);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
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

  const toggle = useCallback(() => {
    if (!isOpen) {
      setError(null);
      onOpen?.();
    }
    setIsOpen(!isOpen);
  }, [isOpen, onOpen]);

  // Keyboard (menu pattern): opening focuses the first item, Up/Down/Home/End
  // move between items, Escape closes and returns to the button, Tab closes.
  const buttonRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const menuItems = () => enabledItems(listRef.current);
  useEffect(() => {
    if (isOpen) enabledItems(listRef.current)[0]?.focus();
  }, [isOpen]);
  const onMenuKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      setIsOpen(false);
      buttonRef.current?.focus();
      return;
    }
    if (e.key === "Tab") {
      setIsOpen(false);
      return;
    }
    const all = menuItems();
    if (all.length === 0) return;
    const at = all.indexOf(document.activeElement as HTMLButtonElement);
    const next =
      e.key === "ArrowDown" ? (at + 1) % all.length
      : e.key === "ArrowUp" ? (at <= 0 ? all.length - 1 : at - 1)
      : e.key === "Home" ? 0
      : e.key === "End" ? all.length - 1
      : null;
    if (next === null) return;
    e.preventDefault();
    all[next]!.focus();
  };

  const handleAction = useCallback(async (item: ExportMenuItem) => {
    setError(null);
    const result = item.onSelect();
    if (!(result instanceof Promise)) {
      setIsOpen(false);
      return;
    }
    setBusyKey(item.key);
    try {
      await result;
      setIsOpen(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Export failed.");
    } finally {
      setBusyKey(null);
    }
  }, []);

  if (items.length === 0) return null;

  return (
    <div ref={menuRef} className={`relative inline-block ${className}`}>
      <button
        ref={buttonRef}
        onClick={toggle}
        disabled={disabled}
        title={compact ? label : undefined}
        aria-label={compact ? label : undefined}
        aria-haspopup="menu"
        aria-expanded={isOpen}
        className={
          compact
            ? "rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
            : "flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 shadow-sm transition-colors hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
        }
      >
        <Download className="h-4 w-4" />
        {!compact && (
          <>
            {label}
            <ChevronDown className="h-3.5 w-3.5" />
          </>
        )}
      </button>

      {isOpen && (
        <div
          ref={listRef}
          role="menu"
          aria-label={label}
          onKeyDown={onMenuKeyDown}
          className="absolute right-0 z-50 mt-1 w-64 rounded-lg border border-slate-200 bg-white py-1 shadow-lg"
        >
          {items.map((item) => (
            <div key={item.key}>
              {item.dividerBefore && <div className="mx-3 my-1 border-t border-slate-100" />}
              <button
                role="menuitem"
                onClick={() => void handleAction(item)}
                disabled={item.disabled || busyKey !== null}
                className="flex w-full items-start gap-2.5 px-4 py-2 text-left text-sm text-slate-700 transition-colors hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <span className="mt-0.5 flex-shrink-0">
                  {busyKey === item.key ? (
                    <Loader2 className="h-4 w-4 animate-spin text-slate-400" />
                  ) : (
                    ICONS[item.icon ?? "csv"]
                  )}
                </span>
                <span className="min-w-0">
                  <span className="block">{item.label}</span>
                  {item.hint && <span className="block text-xs text-slate-400">{item.hint}</span>}
                </span>
              </button>
            </div>
          ))}
          {error && (
            <p role="alert" className="mx-3 mt-1 rounded bg-red-50 px-2 py-1.5 text-xs text-red-700">
              {error}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
