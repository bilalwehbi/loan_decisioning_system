import React from 'react';
import { Container, Typography, Box } from '@mui/material';
import ValidationUploader from '../components/ValidationUploader';

const ValidationPage: React.FC = () => {
  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom align="center">
          Document Validation
        </Typography>
        <Typography variant="subtitle1" gutterBottom align="center" color="text.secondary">
          Upload and validate documents for loan applications
        </Typography>
        <ValidationUploader applicationId="current" />
      </Box>
    </Container>
  );
};

export default ValidationPage; 