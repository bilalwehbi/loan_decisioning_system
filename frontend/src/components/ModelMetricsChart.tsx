import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { ModelMetrics } from '../services/api';

interface ModelMetricsChartProps {
  metrics: ModelMetrics[];
  modelType?: string;
  onModelTypeChange?: (type: string) => void;
}

const ModelMetricsChart: React.FC<ModelMetricsChartProps> = ({
  metrics,
  modelType,
  onModelTypeChange,
}) => {
  const chartData = metrics.map((metric) => ({
    timestamp: new Date(metric.timestamp).toLocaleDateString(),
    accuracy: metric.accuracy * 100,
    precision: metric.precision * 100,
    recall: metric.recall * 100,
    f1Score: metric.f1_score * 100,
    aucRoc: metric.auc_roc * 100,
  }));

  const modelTypes = Array.from(new Set(metrics.map((m) => m.model_name)));

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto', mt: 4, mb: 4 }}>
      <Paper sx={{ p: 3 }}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h4">
                Model Performance Metrics
              </Typography>
              {onModelTypeChange && (
                <FormControl sx={{ minWidth: 200 }}>
                  <InputLabel>Model Type</InputLabel>
                  <Select
                    value={modelType || ''}
                    label="Model Type"
                    onChange={(e) => onModelTypeChange(e.target.value)}
                  >
                    {modelTypes.map((type) => (
                      <MenuItem key={type} value={type}>
                        {type}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              )}
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ height: 400 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={chartData}
                  margin={{
                    top: 5,
                    right: 30,
                    left: 20,
                    bottom: 5,
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis
                    domain={[0, 100]}
                    tickFormatter={(value) => `${value}%`}
                  />
                  <Tooltip
                    formatter={(value: number) => [`${value.toFixed(2)}%`]}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="accuracy"
                    stroke="#8884d8"
                    name="Accuracy"
                  />
                  <Line
                    type="monotone"
                    dataKey="precision"
                    stroke="#82ca9d"
                    name="Precision"
                  />
                  <Line
                    type="monotone"
                    dataKey="recall"
                    stroke="#ffc658"
                    name="Recall"
                  />
                  <Line
                    type="monotone"
                    dataKey="f1Score"
                    stroke="#ff8042"
                    name="F1 Score"
                  />
                  <Line
                    type="monotone"
                    dataKey="aucRoc"
                    stroke="#0088fe"
                    name="AUC-ROC"
                  />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Latest Metrics
            </Typography>
            <Grid container spacing={2}>
              {chartData.length > 0 && (
                <>
                  <Grid item xs={12} sm={6} md={2.4}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="subtitle2" color="text.secondary">
                        Accuracy
                      </Typography>
                      <Typography variant="h6">
                        {chartData[chartData.length - 1].accuracy.toFixed(2)}%
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} sm={6} md={2.4}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="subtitle2" color="text.secondary">
                        Precision
                      </Typography>
                      <Typography variant="h6">
                        {chartData[chartData.length - 1].precision.toFixed(2)}%
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} sm={6} md={2.4}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="subtitle2" color="text.secondary">
                        Recall
                      </Typography>
                      <Typography variant="h6">
                        {chartData[chartData.length - 1].recall.toFixed(2)}%
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} sm={6} md={2.4}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="subtitle2" color="text.secondary">
                        F1 Score
                      </Typography>
                      <Typography variant="h6">
                        {chartData[chartData.length - 1].f1Score.toFixed(2)}%
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} sm={6} md={2.4}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="subtitle2" color="text.secondary">
                        AUC-ROC
                      </Typography>
                      <Typography variant="h6">
                        {chartData[chartData.length - 1].aucRoc.toFixed(2)}%
                      </Typography>
                    </Paper>
                  </Grid>
                </>
              )}
            </Grid>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
};

export default ModelMetricsChart; 