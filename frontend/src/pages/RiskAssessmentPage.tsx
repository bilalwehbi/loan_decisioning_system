import React, { useState, useEffect } from 'react';
import { Container, Typography, Box, Grid, CircularProgress, Alert } from '@mui/material';
import RiskScoreDisplay from '../components/RiskScoreDisplay';
import ModelMetricsChart from '../components/ModelMetricsChart';
import { loanApi, RiskScore, ModelMetrics } from '../services/api';

const RiskAssessmentPage: React.FC = () => {
  const [riskScore, setRiskScore] = useState<RiskScore | null>(null);
  const [metrics, setMetrics] = useState<ModelMetrics[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch latest risk assessment
        const latestAssessment = await loanApi.getApplicationHistory(
          new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
          new Date().toISOString(),
          'completed'
        ) as { risk_assessment: RiskScore }[];
        
        if (latestAssessment.length > 0) {
          setRiskScore(latestAssessment[0].risk_assessment);
        }

        // Fetch model metrics
        const modelMetrics = await loanApi.getModelMetrics(7);
        setMetrics(modelMetrics);
      } catch (err: any) {
        setError(err.message || 'Failed to fetch risk assessment data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <Container maxWidth="lg">
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '60vh' }}>
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg">
        <Box sx={{ py: 4 }}>
          <Alert severity="error">{error}</Alert>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom align="center">
          Risk Assessment Dashboard
        </Typography>
        <Typography variant="subtitle1" gutterBottom align="center" color="text.secondary">
          Monitor risk scores and model performance
        </Typography>

        <Grid container spacing={4}>
          {riskScore && (
            <Grid item xs={12}>
              <RiskScoreDisplay riskScore={riskScore} />
            </Grid>
          )}

          <Grid item xs={12}>
            <ModelMetricsChart metrics={metrics} />
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
};

export default RiskAssessmentPage; 