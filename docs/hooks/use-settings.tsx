"use client";

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useRef,
  type ReactNode,
} from "react";

interface Settings {
  apiEndpoint: string;
}

interface SettingsContextValue extends Settings {
  apiKey: string;
  setApiKey: (key: string) => void;
  setApiEndpoint: (endpoint: string) => void;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);

const STORAGE_KEY = "reflexio-docs-settings";
const DEFAULT_SETTINGS: Settings = { apiEndpoint: "http://localhost:8061" };

function loadSettings(): Settings {
  if (typeof window === "undefined") {
    return DEFAULT_SETTINGS;
  }
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      if (typeof parsed.apiEndpoint === "string") return { apiEndpoint: parsed.apiEndpoint };
    }
  } catch {
    // ignore
  }
  return DEFAULT_SETTINGS;
}

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  // Credentials stay in memory, separate from persisted endpoint settings.
  const [apiKey, setApiKey] = useState("");
  const hasLoadedClientSettings = useRef(false);
  const [clientSettingsLoaded, setClientSettingsLoaded] = useState(false);

  useEffect(() => {
    let isMounted = true;
    queueMicrotask(() => {
      if (!isMounted) return;
      setSettings(loadSettings());
      hasLoadedClientSettings.current = true;
      setClientSettingsLoaded(true);
    });
    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    if (
      typeof window !== "undefined" &&
      hasLoadedClientSettings.current &&
      clientSettingsLoaded
    ) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    }
  }, [settings, clientSettingsLoaded]);

  const setApiEndpoint = useCallback((endpoint: string) => {
    setApiKey("");
    setSettings({ apiEndpoint: endpoint });
  }, []);

  return (
    <SettingsContext.Provider
      value={{ ...settings, apiKey, setApiKey, setApiEndpoint }}
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
