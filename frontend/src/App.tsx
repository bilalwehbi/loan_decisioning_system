import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  CssBaseline,
  ThemeProvider,
  createTheme,
} from '@mui/material';

// Import pages
import LoanApplicationPage from './pages/LoanApplicationPage';
import RiskAssessmentPage from './pages/RiskAssessmentPage';
import AdminDashboardPage from './pages/AdminDashboardPage';
import ValidationPage from './pages/ValidationPage';

// Create theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ flexGrow: 1 }}>
          <AppBar position="static">
            <Toolbar>
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                Credex Systems
              </Typography>
              <Button color="inherit" component={Link} to="/">
                Loan Application
              </Button>
              <Button color="inherit" component={Link} to="/risk">
                Risk Assessment
              </Button>
              <Button color="inherit" component={Link} to="/validation">
                Validation
              </Button>
              <Button color="inherit" component={Link} to="/admin">
                Admin
              </Button>
            </Toolbar>
          </AppBar>

          <Routes>
            <Route path="/" element={<LoanApplicationPage />} />
            <Route path="/risk" element={<RiskAssessmentPage />} />
            <Route path="/validation" element={<ValidationPage />} />
            <Route path="/admin" element={<AdminDashboardPage />} />
          </Routes>
        </Box>
      </Router>
    </ThemeProvider>
  );
};

export default App;
