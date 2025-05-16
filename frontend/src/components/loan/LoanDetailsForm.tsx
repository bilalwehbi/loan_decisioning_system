import React from 'react';
import {
  Box,
  Grid,
  TextField,
  Typography,
  MenuItem,
  FormControl,
  InputLabel,
  Select,
  SelectChangeEvent,
} from '@mui/material';
import { LoanDetails } from '../../types/loan';

interface LoanDetailsFormProps {
  data: LoanDetails;
  onChange: (data: LoanDetails) => void;
  errors?: Partial<Record<keyof LoanDetails, string>>;
}

const loanPurposes = [
  'Home Purchase',
  'Home Improvement',
  'Debt Consolidation',
  'Business',
  'Education',
  'Personal',
  'Other',
];

const loanTerms = [12, 24, 36, 48, 60, 72];

const collateralTypes = ['Real Estate', 'Vehicle', 'Savings Account', 'Investment Account', 'None'];

const LoanDetailsForm: React.FC<LoanDetailsFormProps> = ({ data, onChange, errors = {} }) => {
  const handleTextChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    onChange({
      ...data,
      [name]: name === 'loanAmount' || name === 'collateralValue' ? Number(value) : value,
    });
  };

  const handleSelectChange = (e: SelectChangeEvent<string | number>) => {
    const { name, value } = e.target;
    onChange({
      ...data,
      [name]: name === 'loanTerm' ? Number(value) : value,
    });
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Loan Details
      </Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6}>
          <TextField
            required
            fullWidth
            label="Loan Amount"
            name="loanAmount"
            type="number"
            value={data.loanAmount}
            onChange={handleTextChange}
            error={!!errors.loanAmount}
            helperText={errors.loanAmount}
            InputProps={{
              startAdornment: '$',
              inputProps: { min: 0, step: 1000 },
            }}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth required error={!!errors.loanPurpose}>
            <InputLabel>Loan Purpose</InputLabel>
            <Select<string>
              name="loanPurpose"
              value={data.loanPurpose}
              onChange={handleSelectChange}
              label="Loan Purpose"
            >
              {loanPurposes.map(purpose => (
                <MenuItem key={purpose} value={purpose}>
                  {purpose}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth required error={!!errors.loanTerm}>
            <InputLabel>Loan Term (months)</InputLabel>
            <Select<number>
              name="loanTerm"
              value={data.loanTerm}
              onChange={handleSelectChange}
              label="Loan Term (months)"
            >
              {loanTerms.map(term => (
                <MenuItem key={term} value={term}>
                  {term} months
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth required error={!!errors.collateralType}>
            <InputLabel>Collateral Type</InputLabel>
            <Select<string>
              name="collateralType"
              value={data.collateralType}
              onChange={handleSelectChange}
              label="Collateral Type"
            >
              {collateralTypes.map(type => (
                <MenuItem key={type} value={type}>
                  {type}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        {data.collateralType !== 'None' && (
          <Grid item xs={12}>
            <TextField
              required
              fullWidth
              label="Collateral Value"
              name="collateralValue"
              type="number"
              value={data.collateralValue}
              onChange={handleTextChange}
              error={!!errors.collateralValue}
              helperText={errors.collateralValue}
              InputProps={{
                startAdornment: '$',
                inputProps: { min: 0, step: 1000 },
              }}
            />
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default LoanDetailsForm;
