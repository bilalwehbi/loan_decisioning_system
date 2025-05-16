import { useEffect, useState } from 'react';
import {
  Box,
  Container,
  Grid,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  CardHeader,
  Divider,
} from '@mui/material';
import { loanApi } from '../services/api';

interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc_roc: number;
}

interface LoanApplication {
  id: string;
  loan_amount: number;
  loan_term: number;
  loan_purpose: string;
  status: string;
  decision: string;
  created_at: string;
}

interface MetricsResponse {
  credit_risk: ModelMetrics;
  fraud_detection: ModelMetrics;
}

const Dashboard: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<ModelMetrics[]>([]);
  const [applications, setApplications] = useState<LoanApplication[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch model metrics
        const metricsResponse = await loanApi.getModelMetrics();
        setMetrics(metricsResponse);

        // Fetch recent applications (last 30 days)
        const endDate = new Date().toISOString();
        const startDate = new Date();
        startDate.setDate(startDate.getDate() - 30);
        const applicationsResponse = await loanApi.getApplicationHistory(
          startDate.toISOString(),
          endDate
        );
        setApplications(applicationsResponse as LoanApplication[]);
      } catch (err) {
        setError('Failed to load dashboard data. Please try again.');
        console.error('Dashboard data fetch error:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="80vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Alert severity="error">{error}</Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>

      {/* Model Performance Metrics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12}>
          <Typography variant="h5" gutterBottom>
            Model Performance
          </Typography>
        </Grid>
        {metrics && (
          <>
            <Grid item xs={12} md={6}>
              <Card>
                <CardHeader title="Credit Risk Model" />
                <Divider />
                <CardContent>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="subtitle2">Accuracy</Typography>
                      <Typography variant="h6">{(metrics[0].accuracy * 100).toFixed(2)}%</Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="subtitle2">Precision</Typography>
                      <Typography variant="h6">{(metrics[0].precision * 100).toFixed(2)}%</Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="subtitle2">Recall</Typography>
                      <Typography variant="h6">{(metrics[0].recall * 100).toFixed(2)}%</Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="subtitle2">F1 Score</Typography>
                      <Typography variant="h6">{(metrics[0].f1_score * 100).toFixed(2)}%</Typography>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card>
                <CardHeader title="Fraud Detection Model" />
                <Divider />
                <CardContent>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="subtitle2">Accuracy</Typography>
                      <Typography variant="h6">{(metrics[1].accuracy * 100).toFixed(2)}%</Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="subtitle2">Precision</Typography>
                      <Typography variant="h6">{(metrics[1].precision * 100).toFixed(2)}%</Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="subtitle2">Recall</Typography>
                      <Typography variant="h6">{(metrics[1].recall * 100).toFixed(2)}%</Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="subtitle2">F1 Score</Typography>
                      <Typography variant="h6">{(metrics[1].f1_score * 100).toFixed(2)}%</Typography>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          </>
        )}
      </Grid>

      {/* Recent Applications */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Typography variant="h5" gutterBottom>
            Recent Applications
          </Typography>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>ID</TableCell>
                  <TableCell>Amount</TableCell>
                  <TableCell>Term</TableCell>
                  <TableCell>Purpose</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Decision</TableCell>
                  <TableCell>Date</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {applications.map((app) => (
                  <TableRow key={app.id}>
                    <TableCell>{app.id}</TableCell>
                    <TableCell>${app.loan_amount.toLocaleString()}</TableCell>
                    <TableCell>{app.loan_term} months</TableCell>
                    <TableCell>{app.loan_purpose}</TableCell>
                    <TableCell>{app.status}</TableCell>
                    <TableCell>{app.decision}</TableCell>
                    <TableCell>{new Date(app.created_at).toLocaleDateString()}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Dashboard;
