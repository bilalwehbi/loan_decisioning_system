import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  Chip,
  Divider,
} from '@mui/material';
import { RiskScore } from '../services/api';

interface RiskScoreDisplayProps {
  riskScore: RiskScore;
}

const RiskScoreDisplay: React.FC<RiskScoreDisplayProps> = ({ riskScore }) => {
  const getRiskColor = (score: number) => {
    if (score >= 0.8) return 'error';
    if (score >= 0.5) return 'warning';
    return 'success';
  };

  const getRiskLabel = (score: number) => {
    if (score >= 0.8) return 'High Risk';
    if (score >= 0.5) return 'Medium Risk';
    return 'Low Risk';
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', mt: 4, mb: 4 }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Risk Assessment Results
        </Typography>

        <Grid container spacing={3}>
          {/* Risk Score */}
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Overall Risk Score
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Box sx={{ flexGrow: 1, mr: 2 }}>
                <LinearProgress
                  variant="determinate"
                  value={riskScore.score * 100}
                  color={getRiskColor(riskScore.score)}
                  sx={{ height: 10, borderRadius: 5 }}
                />
              </Box>
              <Typography variant="body1">
                {Math.round(riskScore.score * 100)}%
              </Typography>
            </Box>
            <Chip
              label={getRiskLabel(riskScore.score)}
              color={getRiskColor(riskScore.score)}
              sx={{ mt: 1 }}
            />
          </Grid>

          <Grid item xs={12}>
            <Divider />
          </Grid>

          {/* Probability of Default */}
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              Probability of Default
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Box sx={{ flexGrow: 1, mr: 2 }}>
                <LinearProgress
                  variant="determinate"
                  value={riskScore.probability_default * 100}
                  color={getRiskColor(riskScore.probability_default)}
                  sx={{ height: 10, borderRadius: 5 }}
                />
              </Box>
              <Typography variant="body1">
                {Math.round(riskScore.probability_default * 100)}%
              </Typography>
            </Box>
          </Grid>

          {/* Fraud Risk Score */}
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              Fraud Risk Score
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Box sx={{ flexGrow: 1, mr: 2 }}>
                <LinearProgress
                  variant="determinate"
                  value={riskScore.fraud_risk_score * 100}
                  color={getRiskColor(riskScore.fraud_risk_score)}
                  sx={{ height: 10, borderRadius: 5 }}
                />
              </Box>
              <Typography variant="body1">
                {Math.round(riskScore.fraud_risk_score * 100)}%
              </Typography>
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Divider />
          </Grid>

          {/* Risk Factors */}
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Key Risk Factors
            </Typography>
            <List>
              {riskScore.top_factors.map((factor, index) => (
                <ListItem key={index}>
                  <ListItemText
                    primary={Object.keys(factor)[0]}
                    secondary={`Impact: ${Math.round(Object.values(factor)[0] * 100)}%`}
                  />
                </ListItem>
              ))}
            </List>
          </Grid>

          {/* Fraud Flags */}
          {riskScore.fraud_flags.length > 0 && (
            <>
              <Grid item xs={12}>
                <Divider />
              </Grid>
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Fraud Flags
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {riskScore.fraud_flags.map((flag, index) => (
                    <Chip
                      key={index}
                      label={flag}
                      color="error"
                      variant="outlined"
                    />
                  ))}
                </Box>
              </Grid>
            </>
          )}

          {/* Explanations */}
          <Grid item xs={12}>
            <Divider />
          </Grid>
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Decision Explanation
            </Typography>
            <List>
              {Object.entries(riskScore.explanation).map(([key, value], index) => (
                <ListItem key={index}>
                  <ListItemText
                    primary={key}
                    secondary={value}
                  />
                </ListItem>
              ))}
            </List>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
};

export default RiskScoreDisplay; 