"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export default function PlotlyChart({ data, layout }: { data: any[]; layout?: any }) {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const checkTheme = () => {
      const isDarkTheme =
        document.documentElement.classList.contains("dark") ||
        document.documentElement.getAttribute("data-theme") === "dark";
      setIsDark(isDarkTheme);
    };

    checkTheme();

    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.attributeName === "data-theme" || mutation.attributeName === "class") {
          checkTheme();
        }
      });
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme", "class"],
    });

    return () => observer.disconnect();
  }, []);

  const fontColor = isDark ? "#F0F6FC" : "#12151B";
  const gridColor = isDark ? "#30363D" : "#EEDBD1";
  const zeroLineColor = isDark ? "#485461" : "#DEDBD1";

  // Map light-theme lines dynamically to readable dark mode equivalent
  const adjustedData = data.map((trace) => {
    if (trace.line?.color === "#12151B" || trace.line?.color === "rgb(var(--color-ink))") {
      return {
        ...trace,
        line: {
          ...trace.line,
          color: fontColor,
        },
      };
    }
    return trace;
  });

  const mergedLayout = {
    ...layout,
    font: {
      family: "Inter, sans-serif",
      size: 12,
      color: fontColor,
      ...(layout?.font || {}),
    },
    title: typeof layout?.title === "string" 
      ? { text: layout.title, font: { color: fontColor } } 
      : { ...layout?.title, font: { color: fontColor, ...(layout?.title?.font || {}) } },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    xaxis: {
      title: typeof layout?.xaxis?.title === "string"
        ? { text: layout.xaxis.title, font: { color: fontColor } }
        : { ...layout?.xaxis?.title, font: { color: fontColor, ...(layout?.xaxis?.title?.font || {}) } },
      ...layout?.xaxis,
      gridcolor: layout?.xaxis?.gridcolor ?? gridColor,
      zerolinecolor: layout?.xaxis?.zerolinecolor ?? zeroLineColor,
      tickfont: {
        color: fontColor,
        ...layout?.xaxis?.tickfont,
      },
    },
    yaxis: {
      title: typeof layout?.yaxis?.title === "string"
        ? { text: layout.yaxis.title, font: { color: fontColor } }
        : { ...layout?.yaxis?.title, font: { color: fontColor, ...(layout?.yaxis?.title?.font || {}) } },
      ...layout?.yaxis,
      gridcolor: layout?.yaxis?.gridcolor ?? gridColor,
      zerolinecolor: layout?.yaxis?.zerolinecolor ?? zeroLineColor,
      tickfont: {
        color: fontColor,
        ...layout?.yaxis?.tickfont,
      },
    },
  };

  return (
    <Plot
      data={adjustedData}
      layout={mergedLayout}
      style={{ width: "100%", height: "360px" }}
      config={{ displayModeBar: false, responsive: true }}
    />
  );
}
