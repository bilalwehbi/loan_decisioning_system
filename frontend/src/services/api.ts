import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: false, // Disable credentials for now
  timeout: 10000, // Add timeout
});

// Add API key to all requests
api.interceptors.request.use((config) => {
  const apiKey = localStorage.getItem('apiKey');
  if (apiKey && config.headers) {
    config.headers['X-API-Key'] = apiKey;
  }
  return config;
});

// Add response interceptor for better error handling
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      console.error('API Error Response:', {
        status: error.response.status,
        data: error.response.data,
        headers: error.response.headers,
      });

      // Handle specific error cases
      if (error.response.status === 401) {
        // Handle unauthorized error (e.g., invalid API key)
        console.error('Authentication error: Invalid or missing API key');
      } else if (error.response.status === 500) {
        // Handle server error
        console.error('Server error:', error.response.data);
      }
    } else if (error.request) {
      // The request was made but no response was received
      console.error('API Error Request:', error.request);
    } else {
      // Something happened in setting up the request that triggered an Error
      console.error('API Error:', error.message);
    }

    // Return a more user-friendly error
    return Promise.reject({
      message: error.response?.data?.detail || error.message || 'An unexpected error occurred',
      status: error.response?.status,
      data: error.response?.data
    });
  }
);

export interface CreditData {
  credit_score: number;
  delinquencies: number;
  inquiries_last_6m: number;
  tradelines: number;
  utilization: number;
  payment_history_score: number;
  credit_age_months: number;
  credit_mix_score: number;
}

export interface BankingData {
  avg_monthly_income: number;
  income_stability_score: number;
  spending_pattern_score: number;
  transaction_count: number;
  avg_account_balance: number;
  overdraft_frequency: number;
  savings_rate: number;
  recurring_payments: Array<{ description: string; amount: number }>;
}

export interface BehavioralData {
  application_completion_time: number;
  form_fill_pattern: Record<string, number>;
  device_trust_score: number;
  location_risk_score: number;
  digital_footprint_score: number;
  social_media_presence: Record<string, number>;
}

export interface EmploymentData {
  employer: string;
  job_title: string;
  start_date: string;
  end_date?: string;
  income: number;
  employment_status: string;
}

export interface ApplicationData {
  device_id: string;
  ip_address: string;
  email_domain: string;
  phone_number: string;
  application_timestamp: string;
  device_fingerprint: Record<string, any>;
  browser_fingerprint: Record<string, any>;
  network_info: Record<string, any>;
  location_data: Record<string, any>;
}

export interface LoanApplication {
  loan_amount: number;
  loan_term: number;
  loan_purpose: string;
  credit_data: CreditData;
  banking_data: BankingData;
  behavioral_data: BehavioralData;
  employment_data: EmploymentData;
  application_data: ApplicationData;
}

export interface RiskScore {
  score: number;
  probability_default: number;
  risk_segment: string;
  top_factors: Array<Record<string, number>>;
  fraud_risk_score: number;
  fraud_flags: string[];
  explanation: Record<string, string>;
}

export interface ModelMetrics {
  model_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc_roc: number;
  timestamp: string;
}

export interface PerformanceMetrics {
  total_applications: number;
  approval_rate: number;
  default_rate: number;
  fraud_rate: number;
  average_processing_time: number;
  model_performance: ModelMetrics[];
}

export interface ComplianceReport {
  report_id: string;
  report_type: string;
  start_date: string;
  end_date: string;
  total_decisions: number;
  approval_rates: Record<string, number>;
  risk_distribution: Record<string, number>;
  fraud_incidents: number;
  compliance_issues: string[];
  recommendations: string[];
}

export const loanApi = {
  assessLoanApplication: async (application: LoanApplication): Promise<RiskScore> => {
    const response = await api.post<RiskScore>('/api/v1/assess', application);
    return response.data;
  },

  assessLoanEnhanced: async (application: LoanApplication): Promise<RiskScore> => {
    const response = await api.post<RiskScore>('/api/v1/assess/enhanced', application);
    return response.data;
  },

  getApplicationHistory: async (startDate: string, endDate: string, status?: string) => {
    const response = await api.get('/api/v1/applications/history', {
      params: { start_date: startDate, end_date: endDate, status }
    });
    return response.data;
  },

  getModelMetrics: async (days: number = 7): Promise<ModelMetrics[]> => {
    const response = await api.get<ModelMetrics[]>('/api/v1/models/metrics', { params: { days } });
    return response.data;
  },

  retrainModels: async (modelType?: string, force: boolean = false) => {
    const response = await api.post('/api/v1/models/retrain', null, {
      params: { model_type: modelType, force }
    });
    return response.data;
  },

  updateScoringThresholds: async (thresholds: Record<string, number>) => {
    const response = await api.post('/api/v1/config/thresholds', thresholds);
    return response.data;
  },

  getPerformanceMetrics: async (
    startDate: string,
    endDate: string,
    metricType: string = 'all'
  ): Promise<PerformanceMetrics> => {
    const response = await api.get<PerformanceMetrics>('/api/v1/monitoring/performance', {
      params: { start_date: startDate, end_date: endDate, metric_type: metricType }
    });
    return response.data;
  },

  getComplianceReport: async (
    startDate: string,
    endDate: string,
    reportType: string = 'full'
  ): Promise<ComplianceReport> => {
    const response = await api.get<ComplianceReport>('/api/v1/compliance/report', {
      params: { start_date: startDate, end_date: endDate, report_type: reportType }
    });
    return response.data;
  }
};

export const validationApi = {
  validatePaystub: async (applicationId: string, paystubImage: File) => {
    const formData = new FormData();
    formData.append('paystub_image', paystubImage);
    formData.append('application_id', applicationId);
    const response = await api.post('/api/v1/validate/paystub', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  },

  validateBankTransactions: async (applicationId: string, transactions: any[]) => {
    const response = await api.post('/api/v1/validate/bank-transactions', {
      application_id: applicationId,
      transactions
    });
    return response.data;
  },

  validateEmployment: async (employmentData: EmploymentData) => {
    const response = await api.post('/api/v1/validate/employment', employmentData);
    return response.data;
  }
};

export const experimentApi = {
  createExperiment: async (experimentData: any) => {
    const response = await api.post('/api/v1/experiments', experimentData);
    return response.data;
  },

  getExperimentResults: async (experimentId: string) => {
    const response = await api.get(`/api/v1/experiments/${experimentId}`);
    return response.data;
  },

  endExperiment: async (experimentId: string) => {
    const response = await api.post(`/api/v1/experiments/${experimentId}/end`);
    return response.data;
  }
};

export const fraudApi = {
  checkFraud: async (data: any) => {
    const response = await api.post('/fraud/check', data);
    return response.data;
  }
};

export default api; 