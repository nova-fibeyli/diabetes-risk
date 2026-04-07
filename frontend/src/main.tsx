import React from "react";
import ReactDOM from "react-dom/client";
import { ConfigProvider, theme } from "antd";
import App from "./App";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ConfigProvider
      theme={{
        algorithm: theme.defaultAlgorithm,
        token: {
          colorPrimary: "#14b8a6",
          colorInfo: "#14b8a6",
          colorBgBase: "#f8fbfb",
          colorTextBase: "#111827",
          colorBorderSecondary: "rgba(17,24,39,0.08)",
          borderRadius: 18,
          fontFamily:
            '"Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif',
        },
        components: {
          Card: {
            headerHeight: 56,
          },
          Tabs: {
            itemSelectedColor: "#0f766e",
            inkBarColor: "#14b8a6",
          },
        },
      }}
    >
      <App />
    </ConfigProvider>
  </React.StrictMode>,
);
