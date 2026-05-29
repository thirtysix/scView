import { Sidebar } from "./Sidebar";
import { Header } from "./Header";
import { PanelContainer } from "./PanelContainer";
import { useViewStore } from "@/stores/viewStore";

export function AppLayout() {
  const sidebarCollapsed = useViewStore((s) => s.sidebarCollapsed);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div
        className={`flex flex-1 flex-col transition-all duration-200 ${
          sidebarCollapsed ? "ml-16" : "ml-60"
        }`}
      >
        <Header />
        <main className="flex-1 overflow-auto bg-slate-50 p-6">
          <PanelContainer />
        </main>
      </div>
    </div>
  );
}
