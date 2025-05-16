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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
} from '@mui/material';
import { experimentApi } from '../services/api';

interface Experiment {
  id: string;
  name: string;
  description: string;
  model_type: string;
  start_date: string;
  end_date?: string;
  status: 'running' | 'completed' | 'failed';
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
  };
}

const ExperimentManager: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [newExperiment, setNewExperiment] = useState({
    name: '',
    description: '',
    model_type: '',
  });

  const handleCreateExperiment = async () => {
    if (!newExperiment.name || !newExperiment.model_type) {
      setError('Please fill in all required fields');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(false);

    try {
      const result = await experimentApi.createExperiment(newExperiment) as Experiment;
      setExperiments((prev: Experiment[]) => [...prev, result]);
      setSuccess(true);
      setNewExperiment({ name: '', description: '', model_type: '' });
    } catch (err: any) {
      setError(err.message || 'An error occurred while creating the experiment');
    } finally {
      setLoading(false);
    }
  };

  const handleEndExperiment = async (experimentId: string) => {
    setLoading(true);
    setError(null);

    try {
      const result = await experimentApi.endExperiment(experimentId) as Experiment;
      setExperiments((prev: Experiment[]) =>
        prev.map((exp: Experiment) =>
          exp.id === experimentId ? { ...exp, ...result } : exp
        )
      );
      setSuccess(true);
    } catch (err: any) {
      setError(err.message || 'An error occurred while ending the experiment');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'primary';
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto', mt: 4, mb: 4 }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          A/B Testing Manager
        </Typography>

        <Grid container spacing={3}>
          {/* Create New Experiment */}
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Create New Experiment
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Experiment Name"
                  value={newExperiment.name}
                  onChange={(e) =>
                    setNewExperiment((prev) => ({
                      ...prev,
                      name: e.target.value,
                    }))
                  }
                  required
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth required>
                  <InputLabel>Model Type</InputLabel>
                  <Select
                    value={newExperiment.model_type}
                    label="Model Type"
                    onChange={(e) =>
                      setNewExperiment((prev) => ({
                        ...prev,
                        model_type: e.target.value,
                      }))
                    }
                  >
                    <MenuItem value="risk">Risk Assessment</MenuItem>
                    <MenuItem value="fraud">Fraud Detection</MenuItem>
                    <MenuItem value="credit">Credit Scoring</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Description"
                  multiline
                  rows={3}
                  value={newExperiment.description}
                  onChange={(e) =>
                    setNewExperiment((prev) => ({
                      ...prev,
                      description: e.target.value,
                    }))
                  }
                />
              </Grid>
              <Grid item xs={12}>
                <Button
                  variant="contained"
                  onClick={handleCreateExperiment}
                  disabled={loading}
                >
                  {loading ? <CircularProgress size={24} /> : 'Create Experiment'}
                </Button>
              </Grid>
            </Grid>
          </Grid>

          {error && (
            <Grid item xs={12}>
              <Alert severity="error">{error}</Alert>
            </Grid>
          )}

          {success && (
            <Grid item xs={12}>
              <Alert severity="success">
                Operation completed successfully!
              </Alert>
            </Grid>
          )}

          {/* Experiments List */}
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Active Experiments
            </Typography>
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Name</TableCell>
                    <TableCell>Model Type</TableCell>
                    <TableCell>Start Date</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Metrics</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {experiments.map((experiment) => (
                    <TableRow key={experiment.id}>
                      <TableCell>{experiment.name}</TableCell>
                      <TableCell>{experiment.model_type}</TableCell>
                      <TableCell>
                        {new Date(experiment.start_date).toLocaleDateString()}
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={experiment.status}
                          color={getStatusColor(experiment.status)}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        {experiment.metrics && (
                          <Box>
                            <Typography variant="body2">
                              Accuracy: {(experiment.metrics.accuracy * 100).toFixed(2)}%
                            </Typography>
                            <Typography variant="body2">
                              F1 Score: {(experiment.metrics.f1_score * 100).toFixed(2)}%
                            </Typography>
                          </Box>
                        )}
                      </TableCell>
                      <TableCell>
                        {experiment.status === 'running' && (
                          <Button
                            variant="outlined"
                            color="error"
                            size="small"
                            onClick={() => handleEndExperiment(experiment.id)}
                            disabled={loading}
                          >
                            End Experiment
                          </Button>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
};

export default ExperimentManager; 