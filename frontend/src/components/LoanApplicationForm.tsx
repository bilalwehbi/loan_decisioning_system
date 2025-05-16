import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Grid,
  Stepper,
  Step,
  StepLabel,
  CircularProgress,
  Alert,
} from '@mui/material';
import { useForm, Controller } from 'react-hook-form';
import { LoanApplication, loanApi } from '../services/api';

const steps = [
  'Basic Information',
  'Credit Information',
  'Banking Information',
  'Employment Information',
  'Additional Information',
];

const LoanApplicationForm: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const { control, handleSubmit, trigger, formState: { errors } } = useForm<LoanApplication>();

  const handleNext = async () => {
    const fields = getFieldsForStep(activeStep);
    const isValid = await trigger(fields as any);
    if (isValid) {
      setActiveStep((prevStep) => prevStep + 1);
    }
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const onSubmit = async (data: LoanApplication) => {
    setLoading(true);
    setError(null);
    try {
      await loanApi.assessLoanApplication(data);
      setSuccess(true);
      // Handle successful submission (e.g., show results, redirect)
    } catch (err: any) {
      setError(err.message || 'An error occurred while submitting the application');
    } finally {
      setLoading(false);
    }
  };

  const getFieldsForStep = (step: number): string[] => {
    switch (step) {
      case 0:
        return ['loan_amount', 'loan_term', 'loan_purpose'];
      case 1:
        return ['credit_data.credit_score', 'credit_data.delinquencies'];
      case 2:
        return ['banking_data.avg_monthly_income', 'banking_data.avg_account_balance'];
      case 3:
        return ['employment_data.employer', 'employment_data.job_title'];
      case 4:
        return ['behavioral_data.device_trust_score', 'behavioral_data.location_risk_score'];
      default:
        return [];
    }
  };

  const renderStepContent = (step: number) => {
    switch (step) {
      case 0:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Controller
                name="loan_amount"
                control={control}
                rules={{ required: 'Loan amount is required' }}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Loan Amount"
                    type="number"
                    fullWidth
                    error={!!errors.loan_amount}
                    helperText={errors.loan_amount?.message}
                  />
                )}
              />
            </Grid>
            <Grid item xs={12}>
              <Controller
                name="loan_term"
                control={control}
                rules={{ required: 'Loan term is required' }}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Loan Term (months)"
                    type="number"
                    fullWidth
                    error={!!errors.loan_term}
                    helperText={errors.loan_term?.message}
                  />
                )}
              />
            </Grid>
            <Grid item xs={12}>
              <Controller
                name="loan_purpose"
                control={control}
                rules={{ required: 'Loan purpose is required' }}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Loan Purpose"
                    fullWidth
                    error={!!errors.loan_purpose}
                    helperText={errors.loan_purpose?.message}
                  />
                )}
              />
            </Grid>
          </Grid>
        );

      case 1:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Controller
                name="credit_data.credit_score"
                control={control}
                rules={{ required: 'Credit score is required' }}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Credit Score"
                    type="number"
                    fullWidth
                    error={!!errors.credit_data?.credit_score}
                    helperText={errors.credit_data?.credit_score?.message}
                  />
                )}
              />
            </Grid>
            <Grid item xs={12}>
              <Controller
                name="credit_data.delinquencies"
                control={control}
                rules={{ required: 'Number of delinquencies is required' }}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Number of Delinquencies"
                    type="number"
                    fullWidth
                    error={!!errors.credit_data?.delinquencies}
                    helperText={errors.credit_data?.delinquencies?.message}
                  />
                )}
              />
            </Grid>
          </Grid>
        );

      case 2:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Controller
                name="banking_data.avg_monthly_income"
                control={control}
                rules={{ required: 'Average monthly income is required' }}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Average Monthly Income"
                    type="number"
                    fullWidth
                    error={!!errors.banking_data?.avg_monthly_income}
                    helperText={errors.banking_data?.avg_monthly_income?.message}
                  />
                )}
              />
            </Grid>
            <Grid item xs={12}>
              <Controller
                name="banking_data.avg_account_balance"
                control={control}
                rules={{ required: 'Average account balance is required' }}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Average Account Balance"
                    type="number"
                    fullWidth
                    error={!!errors.banking_data?.avg_account_balance}
                    helperText={errors.banking_data?.avg_account_balance?.message}
                  />
                )}
              />
            </Grid>
          </Grid>
        );

      case 3:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Controller
                name="employment_data.employer"
                control={control}
                rules={{ required: 'Employer name is required' }}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Employer Name"
                    fullWidth
                    error={!!errors.employment_data?.employer}
                    helperText={errors.employment_data?.employer?.message}
                  />
                )}
              />
            </Grid>
            <Grid item xs={12}>
              <Controller
                name="employment_data.job_title"
                control={control}
                rules={{ required: 'Job title is required' }}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Job Title"
                    fullWidth
                    error={!!errors.employment_data?.job_title}
                    helperText={errors.employment_data?.job_title?.message}
                  />
                )}
              />
            </Grid>
          </Grid>
        );

      case 4:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Controller
                name="behavioral_data.device_trust_score"
                control={control}
                rules={{ required: 'Device trust score is required' }}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Device Trust Score"
                    type="number"
                    fullWidth
                    error={!!errors.behavioral_data?.device_trust_score}
                    helperText={errors.behavioral_data?.device_trust_score?.message}
                  />
                )}
              />
            </Grid>
            <Grid item xs={12}>
              <Controller
                name="behavioral_data.location_risk_score"
                control={control}
                rules={{ required: 'Location risk score is required' }}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Location Risk Score"
                    type="number"
                    fullWidth
                    error={!!errors.behavioral_data?.location_risk_score}
                    helperText={errors.behavioral_data?.location_risk_score?.message}
                  />
                )}
              />
            </Grid>
          </Grid>
        );

      default:
        return null;
    }
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', mt: 4, mb: 4 }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Loan Application
        </Typography>

        <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        <form onSubmit={handleSubmit(onSubmit)}>
          {renderStepContent(activeStep)}

          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}

          {success && (
            <Alert severity="success" sx={{ mt: 2 }}>
              Application submitted successfully!
            </Alert>
          )}

          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
            <Button
              disabled={activeStep === 0}
              onClick={handleBack}
            >
              Back
            </Button>

            {activeStep === steps.length - 1 ? (
              <Button
                type="submit"
                variant="contained"
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'Submit Application'}
              </Button>
            ) : (
              <Button
                variant="contained"
                onClick={handleNext}
              >
                Next
              </Button>
            )}
          </Box>
        </form>
      </Paper>
    </Box>
  );
};

export default LoanApplicationForm; 