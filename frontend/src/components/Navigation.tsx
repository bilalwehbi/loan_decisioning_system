import React from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { AppBar, Toolbar, Button, Box, Typography, IconButton } from '@mui/material';
import LogoutIcon from '@mui/icons-material/Logout';
import { useDispatch } from 'react-redux';
import { logout } from '../store/slices/authSlice';

const navLinks = [
  { label: 'Dashboard', path: '/dashboard' },
  { label: 'Apply for Loan', path: '/apply' },
  { label: 'Risk Assessment', path: '/risk-assessment' },
  { label: 'Admin Panel', path: '/admin' },
  { label: 'API Testing', path: '/api-testing' },
];

const Navigation: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const dispatch = useDispatch();

  const handleLogout = () => {
    dispatch(logout());
    navigate('/');
  };

  return (
    <AppBar
      position="static"
      color="primary"
      elevation={2}
      sx={{ m: 0, px: 0, py: 0 }}
    >
      <Toolbar
        sx={{
          minHeight: 64,
          display: 'flex',
          justifyContent: 'space-between',
          px: 3,
        }}
      >
        <Typography
          variant="h6"
          sx={{ fontWeight: 700, color: 'white', letterSpacing: 1 }}
        >
          Loan Decisioning System
        </Typography>
        <Box sx={{ display: 'flex', gap: 1.5 }}>
          {navLinks.map(link => {
            const isActive = location.pathname === link.path;
            return (
              <Button
                key={link.path}
                component={Link}
                to={link.path}
                variant={isActive ? 'contained' : 'text'}
                disableElevation
                sx={{
                  fontWeight: 500,
                  bgcolor: isActive ? 'white' : 'transparent',
                  color: isActive ? 'primary.main' : 'white',
                  boxShadow: isActive ? '0 2px 8px 0 rgba(255,255,255,0.10)' : 'none',
                  border: isActive ? '2px solid #fff' : '2px solid transparent',
                  px: 2.5,
                  py: 1.2,
                  fontSize: '1rem',
                  transition: 'all 0.2s',
                  '&:hover': {
                    bgcolor: 'white',
                    color: 'primary.main',
                    border: '2px solid #fff',
                    boxShadow: '0 0 0 4px rgba(255,255,255,0.25)',
                  },
                }}
              >
                {link.label}
              </Button>
            );
          })}
          <IconButton
            onClick={handleLogout}
            sx={{
              ml: 2,
              color: 'white',
              border: '2px solid transparent',
              bgcolor: 'transparent',
              transition: 'all 0.2s',
              '&:hover': {
                bgcolor: 'white',
                color: 'primary.main',
                border: '2px solid #fff',
                boxShadow: '0 0 0 4px rgba(255,255,255,0.25)',
              },
            }}
          >
            <LogoutIcon />
          </IconButton>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navigation;
