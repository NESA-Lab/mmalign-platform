import { useState } from "react";
import { Outlet, useNavigate, useLocation } from "react-router-dom";
import { Layout, Menu, Typography, theme } from "antd";
import {
  DashboardOutlined,
  ExperimentOutlined,
  SwapOutlined,
  BranchesOutlined,
  PlayCircleOutlined,
  AuditOutlined,
  AlertOutlined,
  AppstoreOutlined,
  SettingOutlined,
} from "@ant-design/icons";

const { Header, Sider, Content } = Layout;

const menuItems = [
  { key: "/overview", icon: <DashboardOutlined />, label: "总览" },
  { key: "/benchmark", icon: <ExperimentOutlined />, label: "评测" },
  { key: "/compare", icon: <SwapOutlined />, label: "对比" },
  { key: "/consistency", icon: <BranchesOutlined />, label: "一致性" },
  { key: "/demo", icon: <PlayCircleOutlined />, label: "演示" },
  { key: "/review", icon: <AuditOutlined />, label: "审核" },
  { key: "/drift", icon: <AlertOutlined />, label: "漂移" },
  { key: "/scenarios", icon: <AppstoreOutlined />, label: "场景" },
  { key: "/settings", icon: <SettingOutlined />, label: "配置" },
];

export default function AppLayout() {
  const [collapsed, setCollapsed] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { token } = theme.useToken();

  return (
    <Layout style={{ minHeight: "100vh", background: token.colorBgLayout }}>
      <Sider
        collapsible
        collapsed={collapsed}
        onCollapse={setCollapsed}
        theme="light"
        width={232}
        style={{
          background: token.colorBgContainer,
          boxShadow: "12px 0 30px rgba(15, 23, 42, 0.04)",
          zIndex: 10,
        }}
      >
        <div
          style={{
            height: 72,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: collapsed ? "0 12px" : "0 20px",
          }}
        >
          <Typography.Title
            level={4}
            style={{
              margin: 0,
              color: token.colorPrimary,
              fontWeight: 800,
              letterSpacing: "-0.04em",
              whiteSpace: "nowrap",
            }}
          >
            {collapsed ? "MM" : "MMAlign"}
          </Typography.Title>
        </div>
        <Menu
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
          style={{ borderRight: 0, background: "transparent" }}
        />
      </Sider>
      <Layout style={{ background: "transparent" }}>
        <Header
          style={{
            background: "rgba(255, 255, 255, 0.82)",
            padding: "0 32px",
            display: "flex",
            alignItems: "center",
            backdropFilter: "blur(18px)",
            boxShadow: "0 1px 20px rgba(15, 23, 42, 0.04)",
          }}
        >
          <Typography.Text type="secondary" style={{ fontSize: 14, fontWeight: 500 }}>
            多模态大模型价值对齐与纠偏平台
          </Typography.Text>
        </Header>
        <Content style={{ margin: 32, minHeight: 280 }}>
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  );
}
