import React from 'react';
import { Box, Grid, TextField, Typography, MenuItem, FormControl, InputLabel, Select, SelectChangeEvent } from '@mui/material';
import { FinancialInfo } from '../../types/loan';

interface FinancialInfoFormProps {
  data: FinancialInfo;
  onChange: (data: FinancialInfo) => void;
  errors?: Partial<Record<keyof FinancialInfo, string>>;
}

const employmentStatuses = ['Full-time', 'Part-time', 'Self-employed', 'Retired', 'Unemployed'];

const FinancialInfoForm: React.FC<FinancialInfoFormProps> = ({ data, onChange, errors = {} }) => {
  const handleTextChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    onChange({
      ...data,
      [name]: ['annualIncome', 'otherIncome', 'employmentDuration', 'creditScore', 'delinquencies', 
               'inquiriesLast6m', 'tradelines', 'utilization', 'paymentHistoryScore', 
               'creditAgeMonths', 'creditMixScore'].includes(name) ? Number(value) : value,
    });
  };

  const handleSelectChange = (e: SelectChangeEvent<string>) => {
    const { name, value } = e.target;
    onChange({
      ...data,
      [name]: value,
    });
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Financial Information
      </Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6}>
          <TextField
            required
            fullWidth
            label="Annual Income"
            name="annualIncome"
            type="number"
            value={data.annualIncome}
            onChange={handleTextChange}
            error={!!errors.annualIncome}
            helperText={errors.annualIncome}
            InputProps={{
              startAdornment: '$',
              inputProps: { min: 0, step: 1000 },
            }}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth required error={!!errors.employmentStatus}>
            <InputLabel>Employment Status</InputLabel>
            <Select
              name="employmentStatus"
              value={data.employmentStatus}
              onChange={handleSelectChange}
              label="Employment Status"
            >
              {employmentStatuses.map(status => (
                <MenuItem key={status} value={status}>
                  {status}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            required
            fullWidth
            label="Employer Name"
            name="employerName"
            value={data.employerName}
            onChange={handleTextChange}
            error={!!errors.employerName}
            helperText={errors.employerName}
            inputProps={{ maxLength: 100 }}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            required
            fullWidth
            label="Employment Duration (months)"
            name="employmentDuration"
            type="number"
            value={data.employmentDuration}
            onChange={handleTextChange}
            error={!!errors.employmentDuration}
            helperText={errors.employmentDuration}
            inputProps={{ min: 0, step: 1 }}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Other Income"
            name="otherIncome"
            type="number"
            value={data.otherIncome}
            onChange={handleTextChange}
            error={!!errors.otherIncome}
            helperText={errors.otherIncome || "Include any additional income sources"}
            InputProps={{
              startAdornment: '$',
              inputProps: { min: 0, step: 1000 },
            }}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            required
            fullWidth
            label="Credit Score"
            name="creditScore"
            type="number"
            value={data.creditScore}
            onChange={handleTextChange}
            error={!!errors.creditScore}
            helperText={errors.creditScore}
            inputProps={{ min: 300, max: 850, step: 1 }}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            required
            fullWidth
            label="Number of Delinquencies"
            name="delinquencies"
            type="number"
            value={data.delinquencies}
            onChange={handleTextChange}
            error={!!errors.delinquencies}
            helperText={errors.delinquencies}
            inputProps={{ min: 0, step: 1 }}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            required
            fullWidth
            label="Credit Inquiries (Last 6 Months)"
            name="inquiriesLast6m"
            type="number"
            value={data.inquiriesLast6m}
            onChange={handleTextChange}
            error={!!errors.inquiriesLast6m}
            helperText={errors.inquiriesLast6m}
            inputProps={{ min: 0, step: 1 }}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            required
            fullWidth
            label="Number of Tradelines"
            name="tradelines"
            type="number"
            value={data.tradelines}
            onChange={handleTextChange}
            error={!!errors.tradelines}
            helperText={errors.tradelines}
            inputProps={{ min: 0, step: 1 }}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            required
            fullWidth
            label="Credit Utilization (%)"
            name="utilization"
            type="number"
            value={data.utilization * 100}
            onChange={(e) => {
              const value = Number(e.target.value);
              onChange({
                ...data,
                utilization: value / 100,
              });
            }}
            error={!!errors.utilization}
            helperText={errors.utilization}
            inputProps={{ min: 0, max: 100, step: 1 }}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            required
            fullWidth
            label="Payment History Score (%)"
            name="paymentHistoryScore"
            type="number"
            value={data.paymentHistoryScore * 100}
            onChange={(e) => {
              const value = Number(e.target.value);
              onChange({
                ...data,
                paymentHistoryScore: value / 100,
              });
            }}
            error={!!errors.paymentHistoryScore}
            helperText={errors.paymentHistoryScore}
            inputProps={{ min: 0, max: 100, step: 1 }}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            required
            fullWidth
            label="Credit Age (months)"
            name="creditAgeMonths"
            type="number"
            value={data.creditAgeMonths}
            onChange={handleTextChange}
            error={!!errors.creditAgeMonths}
            helperText={errors.creditAgeMonths}
            inputProps={{ min: 0, step: 1 }}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            required
            fullWidth
            label="Credit Mix Score (%)"
            name="creditMixScore"
            type="number"
            value={data.creditMixScore * 100}
            onChange={(e) => {
              const value = Number(e.target.value);
              onChange({
                ...data,
                creditMixScore: value / 100,
              });
            }}
            error={!!errors.creditMixScore}
            helperText={errors.creditMixScore}
            inputProps={{ min: 0, max: 100, step: 1 }}
          />
        </Grid>
      </Grid>
    </Box>
  );
};

export default FinancialInfoForm;
