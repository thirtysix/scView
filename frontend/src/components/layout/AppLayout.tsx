import { Sidebar } from "./Sidebar";
import { Header } from "./Header";
import { PanelContainer } from "./PanelContainer";
import { CoPilotDrawer } from "@/components/assistant/CoPilotDrawer";
import { InsightBanner } from "@/components/assistant/InsightBanner";
import { ShareLinkBanner } from "@/components/common/ShareLinkBanner";
import { useViewStore } from "@/stores/viewStore";
import { useSessionRestore } from "@/hooks/useSessionRestore";

export function AppLayout() {
  const sidebarCollapsed = useViewStore((s) => s.sidebarCollapsed);
  useSessionRestore();

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div
        className={`flex min-w-0 flex-1 flex-col transition-all duration-200 ${
          sidebarCollapsed ? "ml-16" : "ml-60"
        }`}
      >
        <Header />
        <main className="flex-1 overflow-auto bg-slate-50 p-6">
          <ShareLinkBanner />
          <InsightBanner />
          <PanelContainer />
        </main>
      </div>
      <CoPilotDrawer />
    </div>
  );
}
