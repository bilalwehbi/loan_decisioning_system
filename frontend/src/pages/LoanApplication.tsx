import React, { useState } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  Stepper,
  Step,
  StepLabel,
  Button,
  TextField,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  CircularProgress,
} from '@mui/material';
import { loanApi } from '../services/api';
import { LoanApplication as LoanApplicationType } from '../services/api';

const steps = [
  'Basic Information',
  'Credit Information',
  'Banking Information',
  'Employment Information',
  'Review & Submit',
];

const LoanApplication: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [formData, setFormData] = useState<Partial<LoanApplicationType>>({
    loan_amount: 0,
    loan_term: 12,
    loan_purpose: '',
    credit_data: {
      credit_score: 0,
      delinquencies: 0,
      inquiries_last_6m: 0,
      tradelines: 0,
      utilization: 0,
      payment_history_score: 0,
      credit_age_months: 0,
      credit_mix_score: 0,
    },
    banking_data: {
      avg_monthly_income: 0,
      income_stability_score: 0,
      spending_pattern_score: 0,
      transaction_count: 0,
      avg_account_balance: 0,
      overdraft_frequency: 0,
      savings_rate: 0,
      recurring_payments: [],
    },
    employment_data: {
      employer: '',
      job_title: '',
      start_date: '',
      end_date: '',
      income: 0,
      employment_status: 'employed'
    },
    application_data: {
      device_id: 'web-' + Math.random().toString(36).substr(2, 9),
      ip_address: '',
      email_domain: '',
      phone_number: '',
      application_timestamp: new Date().toISOString(),
      device_fingerprint: {},
      browser_fingerprint: {},
      network_info: {},
      location_data: {},
    },
  });

  const handleNext = () => {
    if (validateStep()) {
      setActiveStep((prevStep) => prevStep + 1);
    }
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const validateStep = (): boolean => {
    switch (activeStep) {
      case 0: // Basic Information
        if (!formData.loan_amount || !formData.loan_term || !formData.loan_purpose) {
          setError('Please fill in all required fields');
          return false;
        }
        break;
      case 1: // Credit Information
        if (!formData.credit_data?.credit_score || !formData.credit_data?.utilization) {
          setError('Please fill in all required credit information');
          return false;
        }
        break;
      case 2: // Banking Information
        if (!formData.banking_data?.avg_monthly_income || !formData.banking_data?.avg_account_balance) {
          setError('Please fill in all required banking information');
          return false;
        }
        break;
      case 3: // Employment Information
        if (!formData.employment_data?.employer || !formData.employment_data?.job_title) {
          setError('Please fill in all required employment information');
          return false;
        }
        break;
    }
    setError(null);
    return true;
  };

  const handleSubmit = async () => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);

      const response = await loanApi.assessLoanApplication(formData as LoanApplicationType);
      setSuccess('Application submitted successfully!');
      console.log('Application response:', response);
    } catch (err: any) {
      const errorMessage = err.message || err.msg || 'Failed to submit application. Please try again.';
      setError(typeof errorMessage === 'string' ? errorMessage : JSON.stringify(errorMessage));
      console.error('Application submission error:', err);
    } finally {
      setLoading(false);
    }
  };

  const renderStepContent = (step: number) => {
    switch (step) {
      case 0:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Loan Amount"
                type="number"
                value={formData.loan_amount || ''}
                onChange={(e) => setFormData({ ...formData, loan_amount: Number(e.target.value) })}
                required
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth required>
                <InputLabel>Loan Term</InputLabel>
                <Select
                  value={formData.loan_term || ''}
                  onChange={(e) => setFormData({ ...formData, loan_term: Number(e.target.value) })}
                  label="Loan Term"
                >
                  <MenuItem value={12}>12 months</MenuItem>
                  <MenuItem value={24}>24 months</MenuItem>
                  <MenuItem value={36}>36 months</MenuItem>
                  <MenuItem value={48}>48 months</MenuItem>
                  <MenuItem value={60}>60 months</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Loan Purpose"
                value={formData.loan_purpose || ''}
                onChange={(e) => setFormData({ ...formData, loan_purpose: e.target.value })}
                required
              />
            </Grid>
          </Grid>
        );

      case 1:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Credit Score"
                type="number"
                value={formData.credit_data?.credit_score || ''}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    credit_data: { ...formData.credit_data!, credit_score: Number(e.target.value) },
                  })
                }
                required
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Credit Utilization"
                type="number"
                value={formData.credit_data?.utilization || ''}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    credit_data: { ...formData.credit_data!, utilization: Number(e.target.value) },
                  })
                }
                required
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Number of Delinquencies"
                type="number"
                value={formData.credit_data?.delinquencies || ''}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    credit_data: { ...formData.credit_data!, delinquencies: Number(e.target.value) },
                  })
                }
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Credit Inquiries (Last 6 Months)"
                type="number"
                value={formData.credit_data?.inquiries_last_6m || ''}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    credit_data: { ...formData.credit_data!, inquiries_last_6m: Number(e.target.value) },
                  })
                }
              />
            </Grid>
          </Grid>
        );

      case 2:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Average Monthly Income"
                type="number"
                value={formData.banking_data?.avg_monthly_income || ''}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    banking_data: { ...formData.banking_data!, avg_monthly_income: Number(e.target.value) },
                  })
                }
                required
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Average Account Balance"
                type="number"
                value={formData.banking_data?.avg_account_balance || ''}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    banking_data: { ...formData.banking_data!, avg_account_balance: Number(e.target.value) },
                  })
                }
                required
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Number of Transactions"
                type="number"
                value={formData.banking_data?.transaction_count || ''}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    banking_data: { ...formData.banking_data!, transaction_count: Number(e.target.value) },
                  })
                }
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Savings Rate"
                type="number"
                value={formData.banking_data?.savings_rate || ''}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    banking_data: { ...formData.banking_data!, savings_rate: Number(e.target.value) },
                  })
                }
              />
            </Grid>
          </Grid>
        );

      case 3:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Employer Name"
                value={formData.employment_data?.employer || ''}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    employment_data: { ...formData.employment_data!, employer: e.target.value },
                  })
                }
                required
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Job Title"
                value={formData.employment_data?.job_title || ''}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    employment_data: { ...formData.employment_data!, job_title: e.target.value },
                  })
                }
                required
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Employment Status"
                value={formData.employment_data?.employment_status || ''}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    employment_data: { ...formData.employment_data!, employment_status: e.target.value },
                  })
                }
              />
            </Grid>
          </Grid>
        );

      case 4:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Review Your Application
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Typography variant="subtitle1">Basic Information</Typography>
                <Typography>Loan Amount: ${formData.loan_amount?.toLocaleString()}</Typography>
                <Typography>Loan Term: {formData.loan_term} months</Typography>
                <Typography>Purpose: {formData.loan_purpose}</Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="subtitle1">Credit Information</Typography>
                <Typography>Credit Score: {formData.credit_data?.credit_score}</Typography>
                <Typography>Utilization: {formData.credit_data?.utilization}%</Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="subtitle1">Banking Information</Typography>
                <Typography>Monthly Income: ${formData.banking_data?.avg_monthly_income?.toLocaleString()}</Typography>
                <Typography>Account Balance: ${formData.banking_data?.avg_account_balance?.toLocaleString()}</Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="subtitle1">Employment Information</Typography>
                <Typography>Employer: {formData.employment_data?.employer}</Typography>
                <Typography>Job Title: {formData.employment_data?.job_title}</Typography>
                <Typography>Employment Status: {formData.employment_data?.employment_status}</Typography>
              </Grid>
            </Grid>
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
      <Paper sx={{ p: 4 }}>
        <Typography variant="h4" gutterBottom align="center">
          Loan Application
        </Typography>

        <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {typeof error === 'string' ? error : JSON.stringify(error)}
          </Alert>
        )}

        {success && (
          <Alert severity="success" sx={{ mb: 2 }}>
            {success}
          </Alert>
        )}

        <Box sx={{ mt: 2, mb: 2 }}>
          {renderStepContent(activeStep)}
        </Box>

        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
          <Button
            disabled={activeStep === 0}
            onClick={handleBack}
          >
            Back
          </Button>
          <Box>
            {activeStep === steps.length - 1 ? (
              <Button
                variant="contained"
                color="primary"
                onClick={handleSubmit}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'Submit Application'}
              </Button>
            ) : (
              <Button
                variant="contained"
                color="primary"
                onClick={handleNext}
              >
                Next
              </Button>
            )}
          </Box>
        </Box>
      </Paper>
    </Container>
  );
};

export default LoanApplication;
