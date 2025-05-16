import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Button,
  CircularProgress,
  Alert,
  TextField,
  MenuItem,
  FormControl,
  InputLabel,
  Select,
} from '@mui/material';
import { validationApi, EmploymentData } from '../services/api';

interface ValidationUploaderProps {
  applicationId: string;
  onValidationComplete?: (result: any) => void;
}

const ValidationUploader: React.FC<ValidationUploaderProps> = ({
  applicationId,
  onValidationComplete,
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [validationType, setValidationType] = useState<string>('');
  const [file, setFile] = useState<File | null>(null);
  const [employmentData, setEmploymentData] = useState<EmploymentData>({
    employer: '',
    start_date: '',
    end_date: '',
    job_title: '',
    income: 0,
    employment_status: 'employed'
  });

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setFile(event.target.files[0]);
    }
  };

  const handleEmploymentDataChange = (field: string) => (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setEmploymentData((prev) => ({
      ...prev,
      [field]: event.target.value,
    }));
  };

  const handleValidation = async () => {
    if (!validationType) {
      setError('Please select a validation type');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(false);

    try {
      let result;

      switch (validationType) {
        case 'paystub':
          if (!file) {
            throw new Error('Please select a paystub file');
          }
          result = await validationApi.validatePaystub(applicationId, file);
          break;

        case 'bank_transactions':
          if (!file) {
            throw new Error('Please select a bank statement file');
          }
          // Assuming the file contains JSON data
          const transactions = await file.text();
          result = await validationApi.validateBankTransactions(
            applicationId,
            JSON.parse(transactions)
          );
          break;

        case 'employment':
          if (!employmentData.employer || !employmentData.job_title) {
            throw new Error('Please fill in all employment details');
          }
          result = await validationApi.validateEmployment(employmentData);
          break;

        default:
          throw new Error('Invalid validation type');
      }

      setSuccess(true);
      if (onValidationComplete) {
        onValidationComplete(result);
      }
    } catch (err: any) {
      setError(err.message || 'An error occurred during validation');
    } finally {
      setLoading(false);
    }
  };

  const renderValidationForm = () => {
    switch (validationType) {
      case 'paystub':
        return (
          <Grid item xs={12}>
            <Button
              variant="outlined"
              component="label"
              fullWidth
              sx={{ mb: 2 }}
            >
              Upload Paystub
              <input
                type="file"
                hidden
                accept=".pdf,.jpg,.jpeg,.png"
                onChange={handleFileChange}
              />
            </Button>
            {file && (
              <Typography variant="body2" color="text.secondary">
                Selected file: {file.name}
              </Typography>
            )}
          </Grid>
        );

      case 'bank_transactions':
        return (
          <Grid item xs={12}>
            <Button
              variant="outlined"
              component="label"
              fullWidth
              sx={{ mb: 2 }}
            >
              Upload Bank Statement
              <input
                type="file"
                hidden
                accept=".json,.csv"
                onChange={handleFileChange}
              />
            </Button>
            {file && (
              <Typography variant="body2" color="text.secondary">
                Selected file: {file.name}
              </Typography>
            )}
          </Grid>
        );

      case 'employment':
        return (
          <>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Employer"
                value={employmentData.employer}
                onChange={handleEmploymentDataChange('employer')}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Job Title"
                value={employmentData.job_title}
                onChange={handleEmploymentDataChange('job_title')}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Start Date"
                type="date"
                value={employmentData.start_date}
                onChange={handleEmploymentDataChange('start_date')}
                InputLabelProps={{ shrink: true }}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="End Date"
                type="date"
                value={employmentData.end_date}
                onChange={handleEmploymentDataChange('end_date')}
                InputLabelProps={{ shrink: true }}
              />
            </Grid>
          </>
        );

      default:
        return null;
    }
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', mt: 4, mb: 4 }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Document Validation
        </Typography>

        <Grid container spacing={3}>
          <Grid item xs={12}>
            <FormControl fullWidth>
              <InputLabel>Validation Type</InputLabel>
              <Select
                value={validationType}
                label="Validation Type"
                onChange={(e) => setValidationType(e.target.value)}
              >
                <MenuItem value="paystub">Paystub Verification</MenuItem>
                <MenuItem value="bank_transactions">Bank Statement Verification</MenuItem>
                <MenuItem value="employment">Employment Verification</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          {renderValidationForm()}

          {error && (
            <Grid item xs={12}>
              <Alert severity="error">{error}</Alert>
            </Grid>
          )}

          {success && (
            <Grid item xs={12}>
              <Alert severity="success">
                Validation completed successfully!
              </Alert>
            </Grid>
          )}

          <Grid item xs={12}>
            <Button
              variant="contained"
              onClick={handleValidation}
              disabled={loading}
              fullWidth
            >
              {loading ? <CircularProgress size={24} /> : 'Validate'}
            </Button>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
};

export default ValidationUploader; 