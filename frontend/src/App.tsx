import { useEffect, useState } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import Navbar from "./components/Navbar";
import HomePage from "./pages/HomePage";
import ForecastPage from "./pages/ForecastPage";
import ConfigPage from "./pages/ConfigPage";
import { cn } from "./lib/utils";

type ThemeMode = "light" | "dark";

const prefersDark = () =>
  typeof window !== "undefined" &&
  window.matchMedia &&
  window.matchMedia("(prefers-color-scheme: dark)").matches;

const getInitialTheme = (): ThemeMode => {
  if (typeof window === "undefined") {
    return "light";
  }
  const stored = window.localStorage.getItem("theme-mode");
  if (stored === "dark" || stored === "light") {
    return stored;
  }
  return prefersDark() ? "dark" : "light";
};

export function App() {
  const [theme, setTheme] = useState<ThemeMode>(getInitialTheme);

  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle("dark", theme === "dark");
    window.localStorage.setItem("theme-mode", theme);
  }, [theme]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = (event: MediaQueryListEvent) => {
      const stored = window.localStorage.getItem("theme-mode");
      if (stored === "dark" || stored === "light") {
        setTheme(stored);
        return;
      }
      setTheme(event.matches ? "dark" : "light");
    };
    media.addEventListener("change", handler);
    return () => media.removeEventListener("change", handler);
  }, []);

  const toggleTheme = () =>
    setTheme((prev) => (prev === "light" ? "dark" : "light"));

  return (
    <BrowserRouter>
      <div className={cn("min-h-screen bg-background pb-16 text-foreground transition-colors")}>
        
        <Navbar
          theme={theme}
          onToggleTheme={toggleTheme}
          items={[
            { label: "Inicio", to: "/" },
            { label: "Predicciones", to: "/predicciones" },
            { label: "Configuraciones", to: "/configuraciones" }
          ]}
        />

        <main className="container max-w-5xl py-10">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/predicciones" element={<ForecastPage />} />
            <Route path="/configuraciones" element={<ConfigPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;