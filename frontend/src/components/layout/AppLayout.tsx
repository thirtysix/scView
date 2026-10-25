import { Sidebar } from "./Sidebar";
import { Header } from "./Header";
import { PanelContainer } from "./PanelContainer";
import { CoPilotDrawer } from "@/components/assistant/CoPilotDrawer";
import { InsightBanner } from "@/components/assistant/InsightBanner";
import { ShareLinkBanner } from "@/components/common/ShareLinkBanner";
import { SelectionAnnouncer } from "./SelectionAnnouncer";
import { useViewStore } from "@/stores/viewStore";
import { useSessionRestore } from "@/hooks/useSessionRestore";

export function AppLayout() {
  const sidebarCollapsed = useViewStore((s) => s.sidebarCollapsed);
  useSessionRestore();

  return (
    <div className="flex h-screen overflow-hidden">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:left-2 focus:top-2 focus:z-[100] focus:rounded-md focus:bg-white focus:px-3 focus:py-2 focus:text-sm focus:font-medium focus:text-primary focus:shadow-lg"
      >
        Skip to main content
      </a>
      <Sidebar />
      <div
        className={`flex min-w-0 flex-1 flex-col transition-all duration-200 ${
          sidebarCollapsed ? "ml-16" : "ml-60"
        }`}
      >
        <Header />
        <main id="main-content" tabIndex={-1} className="flex-1 overflow-auto bg-slate-50 p-6 focus:outline-none">
          <ShareLinkBanner />
          <InsightBanner />
          <PanelContainer />
        </main>
      </div>
      <CoPilotDrawer />
      <SelectionAnnouncer />
    </div>
  );
}
