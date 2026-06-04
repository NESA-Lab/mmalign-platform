import { createBrowserRouter, Navigate } from "react-router-dom";
import AppLayout from "./layouts/AppLayout";
import OverviewPage from "./pages/overview";
import BenchmarkPage from "./pages/benchmark";
import ComparePage from "./pages/compare";
import ConsistencyPage from "./pages/consistency";
import DemoPage from "./pages/demo";
import ReviewPage from "./pages/review";
import DriftPage from "./pages/drift";
import ScenariosPage from "./pages/scenarios";
import SettingsPage from "./pages/settings";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppLayout />,
    children: [
      { index: true, element: <Navigate to="/overview" replace /> },
      { path: "overview", element: <OverviewPage /> },
      { path: "benchmark", element: <BenchmarkPage /> },
      { path: "compare", element: <ComparePage /> },
      { path: "consistency", element: <ConsistencyPage /> },
      { path: "demo", element: <DemoPage /> },
      { path: "review", element: <ReviewPage /> },
      { path: "drift", element: <DriftPage /> },
      { path: "scenarios", element: <ScenariosPage /> },
      { path: "settings", element: <SettingsPage /> },
    ],
  },
]);
