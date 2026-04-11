"use client";

import { ThemeProvider } from "next-themes";
import { SettingsProvider } from "@/hooks/use-settings";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
      <SettingsProvider>{children}</SettingsProvider>
    </ThemeProvider>
  );
}
