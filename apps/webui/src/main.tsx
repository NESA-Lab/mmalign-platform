import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { applyDefaultAntdModeClass } from "./utils/defaultAntd";
import "./index.css";

applyDefaultAntdModeClass();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
