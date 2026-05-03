"use client";

import { useEffect, useRef } from "react";
import {
  Chart,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  LineController,
  BarController,
  Filler,
  Tooltip,
  Legend,
} from "chart.js";

Chart.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  LineController,
  BarController,
  Filler,
  Tooltip,
  Legend
);

export default function PriceChart() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<Chart | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    // Destroy previous chart if it exists
    if (chartRef.current) {
      chartRef.current.destroy();
    }

    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;

    // --- Data ---
    const labels = [
      "Jan",
      "Feb",
      "Mar",
      "Apr",
      "Mei",
      "Jun",
      "Jul",
      "Agt",
      "Sep",
      "Okt",
      "Nov",
      "Des",
      "Jan",
      "Feb",
      "Mar",
      "Apr",
      "24/4",
      "25/4",
      "26/4",
      "27/4",
      "28/4",
      "29/4",
      "30/4",
      "1/5",
      "2/5",
      "3/5",
      "4/5",
    ];

    // Actual prices for CMK (historical)
    const actualCMK = [
      32000, 34500, 38000, 36000, 33000, 31000, 30000, 32500, 35000, 37000,
      39500, 41000, 44000, 40000, 37500, 38500, 39000, 36100, 38500, 42500,
      null, null, null, null, null, null, null,
    ];

    // Predicted CMK (last portion overlaps + future)
    const predictedCMK = [
      null, null, null, null, null, null, null, null, null, null, null, null,
      null, null, null, null, null, null, null, 42500, 43800, 44500, 45900,
      47200, 48600, 49300, 50100,
    ];

    // Actual prices for CRM (historical)
    const actualCRM = [
      28000, 30000, 33000, 31500, 29000, 27500, 26000, 28500, 31000, 33500,
      36000, 37500, 39500, 36500, 34000, 35000, 35500, 33000, 35500, 38000,
      null, null, null, null, null, null, null,
    ];

    // Rainfall data (mm)
    const rainfall = [
      120, 140, 180, 200, 150, 80, 60, 70, 90, 130, 160, 220, 250, 200, 170,
      190, 280, 300, 250, 320, 280, 260, 240, 220, 200, 180, 170,
    ];

    // --- Gradient for actual line ---
    const greenGradient = ctx.createLinearGradient(0, 0, 0, 350);
    greenGradient.addColorStop(0, "rgba(15, 110, 86, 0.15)");
    greenGradient.addColorStop(1, "rgba(15, 110, 86, 0.01)");

    chartRef.current = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            type: "bar",
            label: "Curah hujan",
            data: rainfall,
            backgroundColor: "rgba(56, 189, 248, 0.15)",
            borderColor: "rgba(56, 189, 248, 0.3)",
            borderWidth: 1,
            borderRadius: 3,
            yAxisID: "y1",
            order: 3,
          },
          {
            type: "line",
            label: "Aktual CMK",
            data: actualCMK,
            borderColor: "#0F6E56",
            backgroundColor: greenGradient,
            borderWidth: 2.5,
            pointRadius: 0,
            pointHoverRadius: 5,
            pointHoverBackgroundColor: "#0F6E56",
            tension: 0.4,
            fill: true,
            yAxisID: "y",
            order: 1,
          },
          {
            type: "line",
            label: "Prediksi CMK",
            data: predictedCMK,
            borderColor: "#0F6E56",
            borderWidth: 2,
            borderDash: [6, 4],
            pointRadius: 3,
            pointBackgroundColor: "#0F6E56",
            pointBorderColor: "#fff",
            pointBorderWidth: 2,
            pointHoverRadius: 6,
            tension: 0.4,
            fill: false,
            yAxisID: "y",
            order: 0,
          },
          {
            type: "line",
            label: "Aktual CRM",
            data: actualCRM,
            borderColor: "#E67E22",
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 5,
            pointHoverBackgroundColor: "#E67E22",
            tension: 0.4,
            fill: false,
            yAxisID: "y",
            order: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: "index",
          intersect: false,
        },
        plugins: {
          legend: {
            display: false,
          },
          tooltip: {
            backgroundColor: "#1F2937",
            titleColor: "#F9FAFB",
            bodyColor: "#D1D5DB",
            padding: 12,
            cornerRadius: 8,
            titleFont: { size: 12, weight: "bold" as const },
            bodyFont: { size: 11 },
            displayColors: true,
            boxWidth: 8,
            boxHeight: 8,
            boxPadding: 4,
            callbacks: {
              label: function (context) {
                const label = context.dataset.label || "";
                const value = context.parsed.y;
                if (label === "Curah hujan") {
                  return ` ${label}: ${value} mm`;
                }
                return ` ${label}: Rp ${value?.toLocaleString("id-ID")}`;
              },
            },
          },
        },
        scales: {
          x: {
            grid: {
              display: false,
            },
            ticks: {
              color: "#9CA3AF",
              font: { size: 10 },
              maxRotation: 0,
            },
            border: {
              display: false,
            },
          },
          y: {
            position: "left" as const,
            grid: {
              color: "rgba(0,0,0,0.04)",
            },
            ticks: {
              color: "#9CA3AF",
              font: { size: 10 },
              callback: function (tickValue: string | number) {
                const val =
                  typeof tickValue === "string"
                    ? parseFloat(tickValue)
                    : tickValue;
                return `Rp ${(val / 1000).toFixed(0)}k`;
              },
            },
            border: {
              display: false,
            },
            min: 24000,
            max: 55000,
          },
          y1: {
            position: "right" as const,
            grid: {
              display: false,
            },
            ticks: {
              color: "#9CA3AF",
              font: { size: 10 },
              callback: function (tickValue: string | number) {
                return `${tickValue}mm`;
              },
            },
            border: {
              display: false,
            },
            min: 0,
            max: 400,
          },
        },
      },
    });

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
  }, []);

  return <canvas ref={canvasRef} />;
}
