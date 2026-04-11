"use client";

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  ReactNode,
} from "react";

interface Settings {
  apiEndpoint: string;
}

interface SettingsContextValue extends Settings {
  setApiEndpoint: (endpoint: string) => void;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);

const STORAGE_KEY = "reflexio-docs-settings";

function loadSettings(): Settings {
  if (typeof window === "undefined") {
    return { apiEndpoint: "http://localhost:8081" };
  }
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) return JSON.parse(stored);
  } catch {
    // ignore
  }
  return { apiEndpoint: "http://localhost:8081" };
}

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>({
    apiEndpoint: "http://localhost:8081",
  });
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setSettings(loadSettings());
    setMounted(true);
  }, []);

  useEffect(() => {
    if (mounted) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    }
  }, [settings, mounted]);

  const setApiEndpoint = useCallback((endpoint: string) => {
    setSettings((prev) => ({ ...prev, apiEndpoint: endpoint }));
  }, []);

  return (
    <SettingsContext.Provider
      value={{ ...settings, setApiEndpoint }}
    >
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings(): SettingsContextValue {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error("useSettings must be used within SettingsProvider");
  return ctx;
}
