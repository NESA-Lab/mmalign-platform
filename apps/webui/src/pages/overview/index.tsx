import { useState } from "react";
import { Segmented, Typography } from "antd";
import ProjectBoard from "./ProjectBoard";
import UserDashboard from "./UserDashboard";
import "./overview.css";

type OverviewTab = "user" | "project";

export default function OverviewPage() {
  const [activeView, setActiveView] = useState<OverviewTab>("user");

  return (
    <div className="overview-page">
      <div className="overview-header">
        <Typography.Title level={3} className="overview-title">
          Overview
        </Typography.Title>
        <Segmented
          size="large"
          value={activeView}
          onChange={(value) => setActiveView(value as OverviewTab)}
          options={[
            { label: "用户工作台", value: "user" },
            { label: "项目看板", value: "project" },
          ]}
        />
      </div>
      {activeView === "user" ? <UserDashboard /> : <ProjectBoard />}
    </div>
  );
}
