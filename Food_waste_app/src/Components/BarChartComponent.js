import React from "react";
import { BarChart } from "@mui/x-charts/BarChart";
import { Paper } from "@mui/material";
 
const BarChartComponent = () => {
  return (
    <Paper
      sx={{
        backgroundColor: "black", // Black background
        padding: 2,
        borderRadius: 2,
      }}
      elevation={2}
    >
      <BarChart
        xAxis={[
            {
                scaleType: "band",
                data: ["group A", "group B", "group C", "group D"],
                tickLabelStyle: { fill: "white" },
                lineStyle: { stroke: "white", strokeWidth: 2 },
            },
        ]}
        yAxis={[
            {
                tickLabelStyle: { fill: "white" },
                lineStyle: { stroke: "white", strokeWidth: 2 },
            },
        ]}
        series={[
            { data: [3, 5, 4, 7], color: "blue" },
            { data: [6, 3, 8, 9], color: "purple" },
        ]}
        width={500}
        height={300}
        sx={{
            backgroundColor: "black",
            '& .MuiChartsAxis-line': {
            stroke: 'white',
            },
        }}
      />
    </Paper>
  );
};
 
export default BarChartComponent;
 