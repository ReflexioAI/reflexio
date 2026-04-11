"use client";

import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";

export function SplitPanel({
  left,
  right,
}: {
  left: React.ReactNode;
  right: React.ReactNode;
}) {
  return (
    <ResizablePanelGroup
      orientation="horizontal"
      className="min-h-0 flex-1 rounded-lg border border-border"
    >
      <ResizablePanel defaultSize={50} minSize={30}>
        <div className="h-full overflow-auto">{left}</div>
      </ResizablePanel>
      <ResizableHandle withHandle />
      <ResizablePanel defaultSize={50} minSize={30}>
        <div className="h-full overflow-auto">{right}</div>
      </ResizablePanel>
    </ResizablePanelGroup>
  );
}
