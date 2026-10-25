import { useRef, type KeyboardEvent, type ReactNode } from "react";

export interface TabItem<T extends string> {
  id: T;
  label: ReactNode;
}

interface TabListProps<T extends string> {
  tabs: readonly TabItem<T>[];
  active: T;
  onChange: (id: T) => void;
  /** What the tabs switch between, read out by screen readers */
  label: string;
  /** Prefix of the tab and panel ids (`<prefix>-tab-<id>`, `<prefix>-panel-<id>`) */
  idPrefix: string;
  className?: string;
  /** Classes of one tab, by whether it is the open one */
  tabClassName: (selected: boolean) => string;
}

/** The id of a tab's panel, for `aria-controls` / `TabPanel`. */
export const tabPanelId = (idPrefix: string, id: string) => `${idPrefix}-panel-${id}`;
const tabId = (idPrefix: string, id: string) => `${idPrefix}-tab-${id}`;

/**
 * A row of tabs with the ARIA tabs pattern: `tablist` / `tab` roles, the open
 * tab `aria-selected` and the only one in the Tab order; Left/Right arrows,
 * Home and End move to (and open) the previous, next, first or last tab.
 */
export function TabList<T extends string>({
  tabs,
  active,
  onChange,
  label,
  idPrefix,
  className = "",
  tabClassName,
}: TabListProps<T>) {
  const refs = useRef(new Map<string, HTMLButtonElement>());
  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>, index: number) => {
    const last = tabs.length - 1;
    const next =
      e.key === "ArrowRight" ? (index === last ? 0 : index + 1)
      : e.key === "ArrowLeft" ? (index === 0 ? last : index - 1)
      : e.key === "Home" ? 0
      : e.key === "End" ? last
      : null;
    if (next === null) return;
    e.preventDefault();
    const tab = tabs[next]!;
    onChange(tab.id);
    refs.current.get(tab.id)?.focus();
  };
  return (
    <div role="tablist" aria-label={label} className={className}>
      {tabs.map((tab, i) => {
        const selected = tab.id === active;
        return (
          <button
            key={tab.id}
            ref={(el) => {
              if (el) refs.current.set(tab.id, el);
              else refs.current.delete(tab.id);
            }}
            type="button"
            role="tab"
            id={tabId(idPrefix, tab.id)}
            aria-selected={selected}
            aria-controls={tabPanelId(idPrefix, tab.id)}
            tabIndex={selected ? 0 : -1}
            onClick={() => onChange(tab.id)}
            onKeyDown={(e) => onKeyDown(e, i)}
            className={tabClassName(selected)}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}

/** The panel of the open tab, labelled by its tab. */
export function TabPanel({
  idPrefix,
  id,
  className,
  children,
}: {
  idPrefix: string;
  id: string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <div role="tabpanel" id={tabPanelId(idPrefix, id)} aria-labelledby={tabId(idPrefix, id)} className={className}>
      {children}
    </div>
  );
}
