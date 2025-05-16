import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface LoanApplication {
  id: string;
  status: 'pending' | 'approved' | 'rejected';
  applicantName: string;
  amount: number;
  purpose: string;
  riskScore: number;
  createdAt: string;
  updatedAt: string;
}

interface LoanState {
  applications: LoanApplication[];
  currentApplication: LoanApplication | null;
  loading: boolean;
  error: string | null;
}

const initialState: LoanState = {
  applications: [],
  currentApplication: null,
  loading: false,
  error: null,
};

const loanSlice = createSlice({
  name: 'loan',
  initialState,
  reducers: {
    fetchApplicationsStart: state => {
      state.loading = true;
      state.error = null;
    },
    fetchApplicationsSuccess: (state, action: PayloadAction<LoanApplication[]>) => {
      state.loading = false;
      state.applications = action.payload;
    },
    fetchApplicationsFailure: (state, action: PayloadAction<string>) => {
      state.loading = false;
      state.error = action.payload;
    },
    setCurrentApplication: (state, action: PayloadAction<LoanApplication>) => {
      state.currentApplication = action.payload;
    },
    updateApplicationStatus: (
      state,
      action: PayloadAction<{ id: string; status: LoanApplication['status'] }>
    ) => {
      const application = state.applications.find(app => app.id === action.payload.id);
      if (application) {
        application.status = action.payload.status;
      }
      if (state.currentApplication?.id === action.payload.id) {
        state.currentApplication.status = action.payload.status;
      }
    },
  },
});

export const {
  fetchApplicationsStart,
  fetchApplicationsSuccess,
  fetchApplicationsFailure,
  setCurrentApplication,
  updateApplicationStatus,
} = loanSlice.actions;

export default loanSlice.reducer;
