import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  List,
  ListItem,
  ListItemText,
  Divider,
} from '@mui/material';
import { ComplianceReport } from '../services/api';

interface ComplianceReportViewerProps {
  report: ComplianceReport;
}

const ComplianceReportViewer: React.FC<ComplianceReportViewerProps> = ({ report }) => {
  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto', mt: 4, mb: 4 }}>
      <Paper sx={{ p: 3 }}>
        <Grid container spacing={3}>
          {/* Report Header */}
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h4">
                Compliance Report
              </Typography>
              <Chip
                label={report.report_type}
                color="primary"
                variant="outlined"
              />
            </Box>
            <Typography variant="subtitle1" color="text.secondary">
              Period: {new Date(report.start_date).toLocaleDateString()} - {new Date(report.end_date).toLocaleDateString()}
            </Typography>
          </Grid>

          <Grid item xs={12}>
            <Divider />
          </Grid>

          {/* Summary Statistics */}
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Summary Statistics
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6} md={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Total Decisions
                  </Typography>
                  <Typography variant="h6">
                    {report.total_decisions}
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Fraud Incidents
                  </Typography>
                  <Typography variant="h6" color="error">
                    {report.fraud_incidents}
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Compliance Issues
                  </Typography>
                  <Typography variant="h6" color="warning.main">
                    {report.compliance_issues.length}
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Report ID
                  </Typography>
                  <Typography variant="body2" sx={{ wordBreak: 'break-all' }}>
                    {report.report_id}
                  </Typography>
                </Paper>
              </Grid>
            </Grid>
          </Grid>

          {/* Approval Rates */}
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Approval Rates by Segment
            </Typography>
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Segment</TableCell>
                    <TableCell align="right">Approval Rate</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(report.approval_rates).map(([segment, rate]) => (
                    <TableRow key={segment}>
                      <TableCell>{segment}</TableCell>
                      <TableCell align="right">
                        {(rate * 100).toFixed(2)}%
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Grid>

          {/* Risk Distribution */}
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Risk Distribution
            </Typography>
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Risk Level</TableCell>
                    <TableCell align="right">Percentage</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(report.risk_distribution).map(([risk, percentage]) => (
                    <TableRow key={risk}>
                      <TableCell>{risk}</TableCell>
                      <TableCell align="right">
                        {(percentage * 100).toFixed(2)}%
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Grid>

          {/* Compliance Issues */}
          {report.compliance_issues.length > 0 && (
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Compliance Issues
              </Typography>
              <List>
                {report.compliance_issues.map((issue, index) => (
                  <ListItem key={index}>
                    <ListItemText primary={issue} />
                  </ListItem>
                ))}
              </List>
            </Grid>
          )}

          {/* Recommendations */}
          {report.recommendations.length > 0 && (
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Recommendations
              </Typography>
              <List>
                {report.recommendations.map((recommendation, index) => (
                  <ListItem key={index}>
                    <ListItemText primary={recommendation} />
                  </ListItem>
                ))}
              </List>
            </Grid>
          )}
        </Grid>
      </Paper>
    </Box>
  );
};

export default ComplianceReportViewer; 