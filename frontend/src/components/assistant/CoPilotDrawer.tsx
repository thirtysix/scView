import { useCallback, useEffect } from "react";
import { Sparkles, PanelRightClose } from "lucide-react";
import { useViewStore } from "@/stores/viewStore";
import { useDatasetStore } from "@/stores/datasetStore";
import { AssistantChat } from "@/components/assistant/AssistantChat";

const MIN_WIDTH = 320;

/** A persistent ✦ launcher (bottom-right) + a right-side overlay drawer that
 *  renders the co-pilot chat over any panel. Mounted once in AppLayout. The
 *  chat is kept mounted (drawer slides via transform) so the conversation
 *  survives hide/show. The width is draggable (left edge). Available whenever a
 *  dataset is loaded. */
export function CoPilotDrawer() {
  const open = useViewStore((s) => s.copilotOpen);
  const toggle = useViewStore((s) => s.toggleCopilot);
  const setOpen = useViewStore((s) => s.setCopilotOpen);
  const width = useViewStore((s) => s.copilotWidth);
  const setWidth = useViewStore((s) => s.setCopilotWidth);
  const onAssistantPanel = useViewStore((s) => s.activePanel === "assistant");
  const hasDataset = useDatasetStore((s) => s.currentDatasetId !== null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, setOpen]);

  // Drag the left edge to resize. The drawer is right-anchored, so width grows
  // as the cursor moves left: width = viewport - clientX.
  const startDrag = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      const onMove = (ev: MouseEvent) => {
        const max = Math.max(MIN_WIDTH, window.innerWidth * 0.8);
        setWidth(Math.min(max, Math.max(MIN_WIDTH, window.innerWidth - ev.clientX)));
      };
      const onUp = () => {
        document.body.style.userSelect = "";
        document.body.style.cursor = "";
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };
      document.body.style.userSelect = "none";
      document.body.style.cursor = "col-resize";
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    },
    [setWidth]
  );

  if (!hasDataset) return null;

  return (
    <>
      {/* Floating launcher (hidden when the full AI Co-pilot panel is open) */}
      {!open && !onAssistantPanel && (
        <button
          onClick={toggle}
          title="Ask the AI co-pilot"
          className="fixed bottom-6 right-6 z-50 flex items-center gap-2 rounded-full bg-primary px-4 py-3 text-sm font-medium text-primary-foreground shadow-lg transition-transform hover:scale-105"
        >
          <Sparkles className="h-5 w-5" />
          Ask
        </button>
      )}

      {/* Drawer */}
      <aside
        style={{ width, maxWidth: "90vw" }}
        className={`fixed bottom-0 right-0 top-0 z-[60] flex flex-col border-l bg-white shadow-2xl transition-transform duration-200 ${
          open ? "translate-x-0" : "pointer-events-none translate-x-full"
        }`}
        aria-hidden={!open}
      >
        {/* Left-edge resize handle */}
        <div
          onMouseDown={startDrag}
          title="Drag to resize"
          className="group absolute left-0 top-0 z-10 h-full w-1.5 -translate-x-1/2 cursor-col-resize"
        >
          <div className="mx-auto h-full w-px bg-transparent transition-colors group-hover:bg-primary/40" />
        </div>

        <div className="flex items-center justify-between border-b px-4 py-3">
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            <h2 className="text-sm font-semibold">AI Co-pilot</h2>
          </div>
          <button
            onClick={() => setOpen(false)}
            title="Hide (Esc) — your conversation is kept"
            className="flex items-center gap-1 rounded px-1.5 py-1 text-xs text-muted-foreground hover:bg-accent hover:text-foreground"
          >
            <PanelRightClose className="h-4 w-4" />
            Hide
          </button>
        </div>
        <div className="min-h-0 flex-1">
          <AssistantChat />
        </div>
      </aside>
    </>
  );
}
