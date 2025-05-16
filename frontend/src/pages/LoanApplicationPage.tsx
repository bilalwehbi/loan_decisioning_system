import React from 'react';
import { Container, Typography, Box } from '@mui/material';
import LoanApplicationForm from '../components/LoanApplicationForm';

const LoanApplicationPage: React.FC = () => {
  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom align="center">
          Loan Application
        </Typography>
        <Typography variant="subtitle1" gutterBottom align="center" color="text.secondary">
          Complete the form below to apply for a loan
        </Typography>
        <LoanApplicationForm />
      </Box>
    </Container>
  );
};

export default LoanApplicationPage; 