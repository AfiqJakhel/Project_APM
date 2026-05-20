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

interface PriceChartProps {
  labels?: string[];
  data?: number[];
  prediksiH1?: number | null;
  prediksiH3?: number | null;
  prediksiH7?: number | null;
}

export default function PriceChart({
  labels,
  data,
  prediksiH1,
  prediksiH3,
  prediksiH7,
}: PriceChartProps) {
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

    // Use real data if provided, otherwise fallback to demo data
    let chartLabels: string[];
    let actualPrices: (number | null)[];
    let predictedPrices: (number | null)[];

    if (labels && data && data.length > 0) {
      // Real data from API
      chartLabels = [...labels];
      actualPrices = [...data];
      predictedPrices = new Array(data.length).fill(null);

      // Add prediction points
      const lastPrice = data[data.length - 1];
      // Connect prediction line from last actual point
      predictedPrices[predictedPrices.length - 1] = lastPrice;

      if (prediksiH1 !== null && prediksiH1 !== undefined) {
        chartLabels.push("H+1");
        actualPrices.push(null);
        predictedPrices.push(prediksiH1);
      }
      if (prediksiH3 !== null && prediksiH3 !== undefined) {
        chartLabels.push("H+3");
        actualPrices.push(null);
        predictedPrices.push(prediksiH3);
      }
      if (prediksiH7 !== null && prediksiH7 !== undefined) {
        chartLabels.push("H+7");
        actualPrices.push(null);
        predictedPrices.push(prediksiH7);
      }
    } else {
      // Fallback demo data
      chartLabels = [
        "Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agt",
        "Sep", "Okt", "Nov", "Des", "Jan", "Feb", "Mar", "Apr",
        "24/4", "25/4", "26/4", "27/4", "28/4", "29/4", "30/4",
        "1/5", "2/5", "3/5", "4/5",
      ];
      actualPrices = [
        32000, 34500, 38000, 36000, 33000, 31000, 30000, 32500, 35000, 37000,
        39500, 41000, 44000, 40000, 37500, 38500, 39000, 36100, 38500, 42500,
        null, null, null, null, null, null, null,
      ];
      predictedPrices = [
        null, null, null, null, null, null, null, null, null, null, null, null,
        null, null, null, null, null, null, null, 42500, 43800, 44500, 45900,
        47200, 48600, 49300, 50100,
      ];
    }

    // --- Gradient for actual line ---
    const greenGradient = ctx.createLinearGradient(0, 0, 0, 350);
    greenGradient.addColorStop(0, "rgba(15, 110, 86, 0.15)");
    greenGradient.addColorStop(1, "rgba(15, 110, 86, 0.01)");

    // Compute reasonable Y axis bounds
    const allValues = [...actualPrices, ...predictedPrices].filter(
      (v): v is number => v !== null
    );
    const minVal = allValues.length > 0 ? Math.min(...allValues) : 24000;
    const maxVal = allValues.length > 0 ? Math.max(...allValues) : 55000;
    const padding = (maxVal - minVal) * 0.15 || 5000;

    chartRef.current = new Chart(ctx, {
      type: "line",
      data: {
        labels: chartLabels,
        datasets: [
          {
            type: "line",
            label: "Harga Aktual",
            data: actualPrices,
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
            label: "Prediksi",
            data: predictedPrices,
            borderColor: "#0F6E56",
            borderWidth: 2,
            borderDash: [6, 4],
            pointRadius: 4,
            pointBackgroundColor: "#0F6E56",
            pointBorderColor: "#fff",
            pointBorderWidth: 2,
            pointHoverRadius: 6,
            tension: 0.4,
            fill: false,
            yAxisID: "y",
            order: 0,
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
                return ` ${label}: Rp ${value?.toLocaleString("id-ID")}`;
              },
            },
          },
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: {
              color: "#9CA3AF",
              font: { size: 10 },
              maxRotation: 45,
              autoSkip: true,
              maxTicksLimit: 20,
            },
            border: { display: false },
          },
          y: {
            position: "left" as const,
            grid: { color: "rgba(0,0,0,0.04)" },
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
            border: { display: false },
            min: Math.floor((minVal - padding) / 1000) * 1000,
            max: Math.ceil((maxVal + padding) / 1000) * 1000,
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
  }, [labels, data, prediksiH1, prediksiH3, prediksiH7]);

  return <canvas ref={canvasRef} />;
}
