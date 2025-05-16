# Loan Decisioning System Frontend

A modern, responsive web application for managing loan applications and risk assessment.

## Features

- Role-based access control (Admin, Loan Officer, Risk Analyst)
- Real-time loan application processing
- Risk assessment and fraud detection visualization
- Admin panel for model management and A/B testing
- Dark/light mode support
- Responsive design for all devices

## Prerequisites

- Node.js (v16 or higher)
- npm (v7 or higher)

## Installation

1. Install dependencies:
   ```bash
   npm install
   ```

2. Create a `.env` file in the root directory with the following variables:
   ```
   REACT_APP_API_URL=http://localhost:8000
   REACT_APP_ENV=development
   ```

3. Start the development server:
   ```bash
   npm start
   ```

## Available Scripts

- `npm start` - Runs the app in development mode
- `npm test` - Launches the test runner
- `npm run build` - Builds the app for production
- `npm run lint` - Runs ESLint
- `npm run format` - Formats code with Prettier

## Project Structure

```
src/
  ├── assets/        # Static assets
  ├── components/    # Reusable components
  ├── hooks/         # Custom React hooks
  ├── layouts/       # Layout components
  ├── pages/         # Page components
  ├── services/      # API services
  ├── store/         # Redux store
  ├── styles/        # Global styles
  ├── types/         # TypeScript types
  └── utils/         # Utility functions
```

## Development

### Code Style

- Follow the TypeScript style guide
- Use functional components with hooks
- Implement proper error handling
- Write unit tests for components
- Use proper type definitions

### State Management

- Use Redux Toolkit for global state
- Use React Context for theme and auth state
- Use local state for component-specific state

### API Integration

- Use Axios for API calls
- Implement proper error handling
- Use TypeScript interfaces for API responses

## Testing

- Write unit tests using Jest and React Testing Library
- Test components in isolation
- Mock API calls
- Test user interactions

## Deployment

1. Build the production bundle:
   ```bash
   npm run build
   ```

2. The build artifacts will be stored in the `build/` directory

## Contributing

1. Create a feature branch
2. Make your changes
3. Write tests
4. Submit a pull request

## License

MIT 