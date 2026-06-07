import { Sparkles } from "lucide-react";
import { useDatasetStore } from "@/stores/datasetStore";
import { AssistantChat } from "@/components/assistant/AssistantChat";

export function AssistantPanel() {
  const datasetName = useDatasetStore((s) => s.currentDataset?.name);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b px-4 py-3">
        <Sparkles className="h-5 w-5 text-primary" />
        <div>
          <h2 className="text-sm font-semibold">AI Co-pilot</h2>
          <p className="text-xs text-muted-foreground">
            Grounded in {datasetName ?? "your dataset"}&apos;s analysis &amp; results, plus the
            methods/literature corpora — every answer cites its sources.
          </p>
        </div>
      </div>
      <div className="min-h-0 flex-1">
        <AssistantChat />
      </div>
    </div>
  );
}
