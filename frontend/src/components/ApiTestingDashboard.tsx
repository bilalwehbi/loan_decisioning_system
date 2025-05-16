import React, { useState } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Tabs,
  Tab,
  Button,
  TextField,
  Grid,
  Alert,
  CircularProgress,
} from '@mui/material';
import { loanApi, fraudApi, validationApi, experimentApi } from '../services/api';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`api-tabpanel-${index}`}
      aria-labelledby={`api-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

const ApiTestingDashboard: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [response, setResponse] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [requestData, setRequestData] = useState<any>({});
  const [experimentId, setExperimentId] = useState('');
  const [description, setDescription] = useState('');
  const [durationDays, setDurationDays] = useState(14);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
    setResponse(null);
    setError(null);
  };

  const handleRequestDataChange = (field: string, value: any) => {
    setRequestData((prev: any) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleApiCall = async (apiFunction: () => Promise<any>) => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiFunction();
      setResponse(result);
    } catch (err: any) {
      const errorMessage = err.message || err.msg || 'An error occurred';
      setError(typeof errorMessage === 'string' ? errorMessage : JSON.stringify(errorMessage));
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        API Testing Dashboard
      </Typography>

      <Paper sx={{ width: '100%', mb: 2 }}>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          indicatorColor="primary"
          textColor="primary"
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab label="Loan API" />
          <Tab label="Fraud API" />
          <Tab label="Validation API" />
          <Tab label="A/B Testing" />
        </Tabs>

        {/* Loan API Tab */}
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Loan Application Assessment
              </Typography>
              <TextField
                fullWidth
                multiline
                rows={4}
                label="Application Data (JSON)"
                value={JSON.stringify(requestData, null, 2)}
                onChange={(e) => {
                  try {
                    const parsed = JSON.parse(e.target.value);
                    setRequestData(parsed);
                  } catch (err) {
                    // Handle invalid JSON
                  }
                }}
                sx={{ mb: 2 }}
              />
              <Button
                variant="contained"
                onClick={() => handleApiCall(() => loanApi.assessLoanApplication(requestData))}
                disabled={loading}
              >
                Assess Application
              </Button>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Fraud API Tab */}
        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Fraud Check
              </Typography>
              <TextField
                fullWidth
                multiline
                rows={4}
                label="Application Data (JSON)"
                value={JSON.stringify(requestData, null, 2)}
                onChange={(e) => {
                  try {
                    const parsed = JSON.parse(e.target.value);
                    setRequestData(parsed);
                  } catch (err) {
                    // Handle invalid JSON
                  }
                }}
                sx={{ mb: 2 }}
              />
              <Button
                variant="contained"
                onClick={() => handleApiCall(() => fraudApi.checkFraud(requestData.applicationId))}
                disabled={loading}
              >
                Check Fraud
              </Button>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Validation API Tab */}
        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Income Validation
              </Typography>
              <TextField
                fullWidth
                multiline
                rows={4}
                label="Validation Data (JSON)"
                value={JSON.stringify(requestData, null, 2)}
                onChange={(e) => {
                  try {
                    const parsed = JSON.parse(e.target.value);
                    setRequestData(parsed);
                  } catch (err) {
                    // Handle invalid JSON
                  }
                }}
                sx={{ mb: 2 }}
              />
              <Button
                variant="contained"
                onClick={() => handleApiCall(() => validationApi.validatePaystub(requestData.applicationId, requestData.paystubImage))}
                disabled={loading}
                sx={{ mr: 2 }}
              >
                Validate Paystub
              </Button>
              <Button
                variant="contained"
                onClick={() => handleApiCall(() => validationApi.validateBankTransactions(requestData.applicationId, requestData.transactions))}
                disabled={loading}
                sx={{ mr: 2 }}
              >
                Validate Bank Transactions
              </Button>
              <Button
                variant="contained"
                onClick={() => handleApiCall(() => validationApi.validateEmployment(requestData.employmentData))}
                disabled={loading}
              >
                Validate Employment
              </Button>
            </Grid>
          </Grid>
        </TabPanel>

        {/* A/B Testing Tab */}
        <TabPanel value={tabValue} index={3}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                A/B Testing
              </Typography>
              <TextField
                fullWidth
                label="Model Type"
                value={requestData.modelType || ''}
                onChange={(e) => setRequestData({ ...requestData, modelType: e.target.value })}
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                label="Days"
                type="number"
                value={requestData.days || ''}
                onChange={(e) => setRequestData({ ...requestData, days: Number(e.target.value) })}
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                label="Experiment ID"
                value={requestData.experimentId || ''}
                onChange={(e) => setRequestData({ ...requestData, experimentId: e.target.value })}
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                label="Model ID"
                value={requestData.modelId || ''}
                onChange={(e) => setRequestData({ ...requestData, modelId: e.target.value })}
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                multiline
                rows={4}
                label="Experiment Data (JSON)"
                value={JSON.stringify(requestData, null, 2)}
                onChange={(e) => {
                  try {
                    const parsed = JSON.parse(e.target.value);
                    setRequestData(parsed);
                  } catch (err) {
                    // Handle invalid JSON
                  }
                }}
                sx={{ mb: 2 }}
              />
              <Box sx={{ mt: 2 }}>
                <Button
                  variant="contained"
                  onClick={() => handleApiCall(() => loanApi.getModelMetrics(requestData.days))}
                  disabled={loading}
                  sx={{ mr: 2 }}
                >
                  Get Model Metrics
                </Button>
                <Button
                  variant="contained"
                  onClick={() => handleApiCall(() => experimentApi.getExperimentResults(requestData.experimentId))}
                  disabled={loading}
                  sx={{ mr: 2 }}
                >
                  Get Results
                </Button>
                <Button
                  variant="contained"
                  onClick={() => handleApiCall(() => loanApi.getPerformanceMetrics(
                    requestData.startDate,
                    requestData.endDate,
                    requestData.modelId
                  ))}
                  disabled={loading}
                >
                  Get Model Performance
                </Button>
              </Box>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Response Display */}
        {(response || error) && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Response
            </Typography>
            {loading ? (
              <CircularProgress />
            ) : error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {typeof error === 'string' ? error : JSON.stringify(error)}
              </Alert>
            )}
            {response && (
              <Paper sx={{ p: 2, bgcolor: '#f5f5f5' }}>
                <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                  {JSON.stringify(response, null, 2)}
                </pre>
              </Paper>
            )}
          </Box>
        )}
      </Paper>
    </Container>
  );
};

export default ApiTestingDashboard; 