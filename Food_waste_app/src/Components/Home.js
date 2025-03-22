//Home.jsx
import React from 'react';
import { Grid2, Typography, Card, CardContent, Button } from '@mui/material';
import { CloudUpload as CloudUploadIcon, BarChart as BarChartIcon } from '@mui/icons-material';

const Home = () => {
  return (
    <Grid2 container spacing={4} justifyContent="flex-start" sx={{ padding: 4 }}> {/* Changed justifyContent to flex-start */}
      {/* Introduction Section */}
      <Grid2 item xs={12} textAlign="left"> {/* Changed textAlign to left */}
        <Typography variant="h3" fontWeight="bold" gutterBottom>
          Smart Food Planning
        </Typography>
        <Typography variant="h6" sx={{ color: 'white' }} gutterBottom>
          Reduce food wastage and optimize tomorrow’s food requirements using AI-driven predictions.
        </Typography>
      </Grid2>

      {/* Feature Cards */}
      <Grid2 item xs={12} md={5}>
        <Card
          sx={{
            height: '100%',
            background: 'rgba(255, 255, 255, 0.1)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
          }}
        >
          <CardContent>
            <Typography variant="h5" fontWeight="bold" gutterBottom sx={{ color: 'white' }}>
              📊 Train Model
            </Typography>
            <Typography variant="body1" color="white" paragraph>
              Upload an Excel sheet containing past food consumption data. The AI learns patterns to predict future needs.
            </Typography>
            <Button variant="contained" startIcon={<CloudUploadIcon />} fullWidth>
              Train Model
            </Button>
          </CardContent>
        </Card>
      </Grid2>

      <Grid2 item xs={12} md={5}>
        <Card
          sx={{
            height: '100%',
            background: 'rgba(255, 255, 255, 0.1)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
          }}
        >
          <CardContent>
            <Typography variant="h5" fontWeight="bold" gutterBottom sx={{ color: 'white' }}>
              🔮 Prediction
            </Typography>
            <Typography variant="body1" color="white" paragraph>
              Enter food ordered and extra food required to generate AI-driven food wastage predictions and requirement heatmaps.
            </Typography>
            <Button variant="contained" startIcon={<BarChartIcon />} fullWidth>
              Get Predictions
            </Button>
          </CardContent>
        </Card>
      </Grid2>
    </Grid2>
  );
};

export default Home;