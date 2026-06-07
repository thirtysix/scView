import { useEffect } from "react";
import { Sparkles, X } from "lucide-react";
import { useViewStore } from "@/stores/viewStore";
import { useDatasetStore } from "@/stores/datasetStore";
import { AssistantChat } from "@/components/assistant/AssistantChat";

/** A persistent ✦ launcher (bottom-right) + a right-side overlay drawer that
 *  renders the co-pilot chat over any panel. Mounted once in AppLayout. The
 *  chat is kept mounted (drawer slides via transform) so the conversation
 *  survives open/close. Available whenever a dataset is loaded. */
export function CoPilotDrawer() {
  const open = useViewStore((s) => s.copilotOpen);
  const toggle = useViewStore((s) => s.toggleCopilot);
  const setOpen = useViewStore((s) => s.setCopilotOpen);
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
        className={`fixed bottom-0 right-0 top-0 z-[60] flex w-[400px] max-w-[90vw] flex-col border-l bg-white shadow-2xl transition-transform duration-200 ${
          open ? "translate-x-0" : "pointer-events-none translate-x-full"
        }`}
        aria-hidden={!open}
      >
        <div className="flex items-center justify-between border-b px-4 py-3">
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            <h2 className="text-sm font-semibold">AI Co-pilot</h2>
          </div>
          <button
            onClick={() => setOpen(false)}
            title="Close (Esc)"
            className="rounded p-1 text-muted-foreground hover:bg-accent hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="min-h-0 flex-1">
          <AssistantChat />
        </div>
      </aside>
    </>
  );
}
