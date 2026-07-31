import { RouterProvider } from "react-router-dom";
import { ConfigProvider, App as AntApp, type ThemeConfig } from "antd";
import zhCN from "antd/locale/zh_CN";
import { router } from "./router";
import { isDefaultAntdMode } from "./utils/defaultAntd";

const customTheme: ThemeConfig = {
  token: {
    colorPrimary: "#3159f5",
    colorInfo: "#3159f5",
    colorSuccess: "#16a34a",
    colorWarning: "#d97706",
    colorError: "#dc2626",
    colorText: "#172033",
    colorTextSecondary: "#667085",
    colorBgLayout: "#f3f6fb",
    colorBgContainer: "#ffffff",
    colorBorder: "rgba(22, 32, 51, 0.08)",
    colorBorderSecondary: "rgba(22, 32, 51, 0.05)",
    borderRadius: 10,
    borderRadiusLG: 16,
    boxShadow: "0 18px 45px rgba(15, 23, 42, 0.06)",
    fontFamily:
      "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    controlHeight: 38,
    controlHeightSM: 32,
    wireframe: false,
  },
  components: {
    Button: {
      borderRadius: 10,
      controlHeight: 38,
      fontWeight: 600,
    },
    Card: {
      borderRadiusLG: 18,
      paddingLG: 24,
      headerFontSize: 16,
      headerHeight: 58,
    },
    Layout: {
      bodyBg: "#f3f6fb",
      headerBg: "#ffffff",
      siderBg: "#ffffff",
    },
    Menu: {
      itemBorderRadius: 10,
      itemHeight: 42,
      itemMarginInline: 12,
      itemSelectedBg: "rgba(49, 89, 245, 0.1)",
      itemSelectedColor: "#3159f5",
    },
    Table: {
      borderColor: "rgba(22, 32, 51, 0.05)",
      headerBg: "#f8faff",
      headerColor: "#667085",
      rowHoverBg: "#f8faff",
    },
  },
};

const defaultAntdTheme: ThemeConfig = {
  token: {
    colorPrimary: "#1677ff",
    colorInfo: "#1677ff",
  },
  components: {
    Menu: {
      itemSelectedBg: "#e6f4ff",
      itemSelectedColor: "#1677ff",
    },
  },
};

export default function App() {
  const theme = isDefaultAntdMode() ? defaultAntdTheme : customTheme;

  return (
    <ConfigProvider locale={zhCN} theme={theme}>
      <AntApp>
        <RouterProvider router={router} />
      </AntApp>
    </ConfigProvider>
  );
}
