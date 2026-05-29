import { useViewStore } from "@/stores/viewStore";
import { useDatasetStore } from "@/stores/datasetStore";
import { PANEL_IDS, PANEL_LABELS, type PanelId } from "@/lib/constants";
import {
  Upload,
  Import,
  ClipboardCheck,
  ChartScatter,
  LayoutDashboard,
  Table2,
  Dna,
  Library,
  ListTree,
  GitBranch,
  ClipboardList,
  PanelLeftClose,
  PanelLeftOpen,
} from "lucide-react";

const PANEL_ICONS: Record<PanelId, React.ElementType> = {
  load: Upload,
  ingest: Import,
  assessment: ClipboardCheck,
  overview: ChartScatter,
  unified: LayoutDashboard,
  observations: Table2,
  expression: Dna,
  genesets: Library,
  markers: ListTree,
  trajectory: GitBranch,
  provenance: ClipboardList,
};

export function Sidebar() {
  const { activePanel, setPanel, sidebarCollapsed, toggleSidebar } =
    useViewStore();
  const currentDataset = useDatasetStore((s) => s.currentDataset);
  const hasDataset = currentDataset !== null;

  return (
    <aside
      className={`fixed left-0 top-0 z-40 flex h-screen flex-col bg-sidebar text-sidebar-foreground transition-all duration-200 ${
        sidebarCollapsed ? "w-16" : "w-60"
      }`}
    >
      {/* Logo */}
      <div className="flex h-14 items-center gap-3 border-b border-white/10 px-4">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary text-sm font-bold text-primary-foreground">
          sc
        </div>
        {!sidebarCollapsed && (
          <span className="text-lg font-semibold tracking-tight">scView</span>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto p-2">
        {PANEL_IDS.map((id) => {
          const Icon = PANEL_ICONS[id];
          const isActive = activePanel === id;
          const isDisabled = id !== "load" && id !== "ingest" && !hasDataset;

          return (
            <button
              key={id}
              onClick={() => !isDisabled && setPanel(id)}
              disabled={isDisabled}
              className={`mb-0.5 flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition-colors ${
                isActive
                  ? "bg-primary/20 text-primary"
                  : isDisabled
                    ? "cursor-not-allowed text-sidebar-foreground/30"
                    : "text-sidebar-foreground/70 hover:bg-sidebar-accent hover:text-sidebar-foreground"
              }`}
              title={sidebarCollapsed ? PANEL_LABELS[id] : undefined}
            >
              <Icon className="h-4 w-4 shrink-0" />
              {!sidebarCollapsed && <span>{PANEL_LABELS[id]}</span>}
            </button>
          );
        })}
      </nav>

      {/* Collapse toggle */}
      <button
        onClick={toggleSidebar}
        className="flex h-10 items-center justify-center border-t border-white/10 text-sidebar-foreground/50 hover:text-sidebar-foreground"
      >
        {sidebarCollapsed ? (
          <PanelLeftOpen className="h-4 w-4" />
        ) : (
          <PanelLeftClose className="h-4 w-4" />
        )}
      </button>
    </aside>
  );
}
