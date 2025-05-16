import React from 'react';
import { Box, Grid, TextField, Typography } from '@mui/material';
import { PersonalInfo } from '../../types/loan';

interface PersonalInfoFormProps {
  data: PersonalInfo;
  onChange: (data: PersonalInfo) => void;
  errors?: Partial<Record<keyof PersonalInfo, string>>;
}

const PersonalInfoForm: React.FC<PersonalInfoFormProps> = ({ data, onChange, errors = {} }) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    onChange({
      ...data,
      [name]: value,
    });
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Personal Information
      </Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6}>
          <TextField
            required
            fullWidth
            label="First Name"
            name="firstName"
            value={data.firstName}
            onChange={handleChange}
            error={!!errors.firstName}
            helperText={errors.firstName}
            inputProps={{ maxLength: 50 }}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            required
            fullWidth
            label="Last Name"
            name="lastName"
            value={data.lastName}
            onChange={handleChange}
            error={!!errors.lastName}
            helperText={errors.lastName}
            inputProps={{ maxLength: 50 }}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            required
            fullWidth
            label="Email Address"
            name="email"
            type="email"
            value={data.email}
            onChange={handleChange}
            error={!!errors.email}
            helperText={errors.email}
            inputProps={{ maxLength: 100 }}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            required
            fullWidth
            label="Phone Number"
            name="phone"
            type="tel"
            value={data.phone}
            onChange={handleChange}
            error={!!errors.phone}
            helperText={errors.phone}
            inputProps={{ 
              pattern: '[0-9]{10}',
              maxLength: 10,
              placeholder: '1234567890'
            }}
          />
        </Grid>
        <Grid item xs={12}>
          <TextField
            required
            fullWidth
            label="Street Address"
            name="address"
            value={data.address}
            onChange={handleChange}
            error={!!errors.address}
            helperText={errors.address}
            inputProps={{ maxLength: 100 }}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            required
            fullWidth
            label="City"
            name="city"
            value={data.city}
            onChange={handleChange}
            error={!!errors.city}
            helperText={errors.city}
            inputProps={{ maxLength: 50 }}
          />
        </Grid>
        <Grid item xs={12} sm={3}>
          <TextField
            required
            fullWidth
            label="State"
            name="state"
            value={data.state}
            onChange={handleChange}
            error={!!errors.state}
            helperText={errors.state}
            inputProps={{ 
              maxLength: 2,
              style: { textTransform: 'uppercase' }
            }}
          />
        </Grid>
        <Grid item xs={12} sm={3}>
          <TextField
            required
            fullWidth
            label="ZIP Code"
            name="zipCode"
            value={data.zipCode}
            onChange={handleChange}
            error={!!errors.zipCode}
            helperText={errors.zipCode}
            inputProps={{ 
              pattern: '[0-9]{5}',
              maxLength: 5,
              placeholder: '12345'
            }}
          />
        </Grid>
      </Grid>
    </Box>
  );
};

export default PersonalInfoForm;
