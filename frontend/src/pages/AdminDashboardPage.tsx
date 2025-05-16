import React, { useState, useEffect } from 'react';
import { Container, Typography, Box, Paper, Tabs, Tab, CircularProgress, Alert } from '@mui/material';
import ExperimentManager from '../components/ExperimentManager';
import ComplianceReportViewer from '../components/ComplianceReportViewer';
import { loanApi, ComplianceReport } from '../services/api';

const AdminDashboardPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [complianceReport, setComplianceReport] = useState<ComplianceReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchComplianceReport = async () => {
      try {
        const report = await loanApi.getComplianceReport(
          new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
          new Date().toISOString(),
          'full'
        );
        setComplianceReport(report);
      } catch (err: any) {
        setError(err.message || 'Failed to fetch compliance report');
      } finally {
        setLoading(false);
      }
    };

    fetchComplianceReport();
  }, []);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

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
          Admin Dashboard
        </Typography>
        <Typography variant="subtitle1" gutterBottom align="center" color="text.secondary">
          Manage experiments and monitor compliance
        </Typography>

        <Paper sx={{ mt: 4 }}>
          <Tabs
            value={activeTab}
            onChange={handleTabChange}
            indicatorColor="primary"
            textColor="primary"
            centered
          >
            <Tab label="A/B Testing" />
            <Tab label="Compliance Reports" />
          </Tabs>

          <Box sx={{ p: 3 }}>
            {activeTab === 0 && <ExperimentManager />}
            {activeTab === 1 && complianceReport && (
              <ComplianceReportViewer report={complianceReport} />
            )}
          </Box>
        </Paper>
      </Box>
    </Container>
  );
};

export default AdminDashboardPage; 