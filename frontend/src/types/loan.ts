import { CreditData, BankingData, BehavioralData, EmploymentData, ApplicationData } from '../services/api';

export interface PersonalInfo {
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
  address: string;
  city: string;
  state: string;
  zipCode: string;
}

export interface FinancialInfo {
  annualIncome: number;
  employmentStatus: string;
  employerName: string;
  employmentDuration: number;
  otherIncome: number;
  creditScore: number;
  delinquencies: number;
  inquiriesLast6m: number;
  tradelines: number;
  utilization: number;
  paymentHistoryScore: number;
  creditAgeMonths: number;
  creditMixScore: number;
}

export interface LoanDetails {
  loanAmount: number;
  loanPurpose: string;
  loanTerm: number;
  collateralType: string;
  collateralValue: number;
}

export interface Document {
  id: string;
  name: string;
  type: string;
  file: File;
}

export interface LoanApplicationData {
  personalInfo: PersonalInfo;
  financialInfo: FinancialInfo;
  loanDetails: LoanDetails;
  documents: Document[];
  creditData: CreditData;
  bankingData: BankingData;
  behavioralData: BehavioralData;
  employmentData: EmploymentData;
  applicationData: ApplicationData;
}

export interface LoanApplicationState {
  step: number;
  data: LoanApplicationData;
  isSubmitting: boolean;
  error: string | null;
  success: boolean;
}
