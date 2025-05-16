import React from 'react';
import {
  Box,
  Grid,
  Typography,
  Paper,
  List,
  ListItem,
  ListItemText,
  Divider,
  Button,
  Alert,
} from '@mui/material';
import { PersonalInfo, FinancialInfo, LoanDetails, Document } from '../../types/loan';

interface ReviewFormProps {
  personalInfo: PersonalInfo;
  financialInfo: FinancialInfo;
  loanDetails: LoanDetails;
  documents: Document[];
  onSubmit: () => void;
  onBack: () => void;
  isSubmitting?: boolean;
  error?: string | null;
}

const ReviewForm: React.FC<ReviewFormProps> = ({
  personalInfo,
  financialInfo,
  loanDetails,
  documents,
  onSubmit,
  onBack,
  isSubmitting = false,
  error = null,
}) => {
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Review Your Application
      </Typography>
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              Personal Information
            </Typography>
            <List dense>
              <ListItem>
                <ListItemText
                  primary="Name"
                  secondary={`${personalInfo.firstName} ${personalInfo.lastName}`}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Contact"
                  secondary={`${personalInfo.email} | ${personalInfo.phone}`}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Address"
                  secondary={`${personalInfo.address}, ${personalInfo.city}, ${personalInfo.state} ${personalInfo.zipCode}`}
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              Financial Information
            </Typography>
            <List dense>
              <ListItem>
                <ListItemText
                  primary="Annual Income"
                  secondary={formatCurrency(financialInfo.annualIncome)}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Employment"
                  secondary={`${financialInfo.employmentStatus} at ${financialInfo.employerName} (${financialInfo.employmentDuration} months)`}
                />
              </ListItem>
              {financialInfo.otherIncome > 0 && (
                <ListItem>
                  <ListItemText
                    primary="Other Income"
                    secondary={formatCurrency(financialInfo.otherIncome)}
                  />
                </ListItem>
              )}
              <Divider />
              <ListItem>
                <ListItemText
                  primary="Credit Score"
                  secondary={financialInfo.creditScore}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Credit History"
                  secondary={`${financialInfo.creditAgeMonths} months, ${financialInfo.tradelines} tradelines`}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Credit Utilization"
                  secondary={formatPercentage(financialInfo.utilization)}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Payment History Score"
                  secondary={formatPercentage(financialInfo.paymentHistoryScore)}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Credit Mix Score"
                  secondary={formatPercentage(financialInfo.creditMixScore)}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Recent Activity"
                  secondary={`${financialInfo.delinquencies} delinquencies, ${financialInfo.inquiriesLast6m} inquiries in last 6 months`}
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              Loan Details
            </Typography>
            <List dense>
              <ListItem>
                <ListItemText
                  primary="Loan Amount"
                  secondary={formatCurrency(loanDetails.loanAmount)}
                />
              </ListItem>
              <ListItem>
                <ListItemText primary="Purpose" secondary={loanDetails.loanPurpose} />
              </ListItem>
              <ListItem>
                <ListItemText primary="Term" secondary={`${loanDetails.loanTerm} months`} />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Collateral"
                  secondary={`${loanDetails.collateralType}${
                    loanDetails.collateralType !== 'None'
                      ? ` - ${formatCurrency(loanDetails.collateralValue)}`
                      : ''
                  }`}
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              Uploaded Documents
            </Typography>
            <List dense>
              {documents.map(doc => (
                <ListItem key={doc.id}>
                  <ListItemText 
                    primary={doc.name} 
                    secondary={`Type: ${doc.type} (${(doc.file.size / 1024 / 1024).toFixed(2)}MB)`} 
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
            <Button variant="outlined" onClick={onBack} disabled={isSubmitting}>
              Back
            </Button>
            <Button 
              variant="contained" 
              color="primary" 
              onClick={onSubmit}
              disabled={isSubmitting}
            >
              {isSubmitting ? 'Submitting...' : 'Submit Application'}
            </Button>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ReviewForm;
