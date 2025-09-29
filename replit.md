# Digital Pain Translator

## Overview
This is a healthcare application designed for non-verbal pain assessment using facial analysis and caregiver input. The application uses MediaPipe for facial feature detection and provides a Digital Pain Scale for healthcare professionals and caregivers.

## Project Architecture
- **Frontend**: React with TypeScript, Vite build system
- **Backend**: Express.js server with API routes
- **Database**: PostgreSQL with Drizzle ORM (optional, currently uses memory storage)
- **UI**: Radix UI components with Tailwind CSS
- **Camera**: MediaPipe for facial landmark detection

## Current State
- ✅ Development environment configured
- ✅ Frontend and backend integrated
- ✅ Vite dev server running on port 5000
- ✅ Deployment configuration set up
- ✅ Privacy consent modal implemented
- ❌ Database not yet provisioned (using memory storage)

## Recent Changes
- Fixed `import.meta.dirname` compatibility issues for Node.js
- Configured server to bind to `0.0.0.0` for Replit environment
- Set up development workflow for frontend/backend integration
- Configured deployment for autoscale target

## Key Features
- Camera-based facial analysis for pain assessment
- Caregiver input interface
- Privacy-first design with local processing
- Assessment history tracking
- HIPAA compliant design considerations

## Running the Application
The application runs via the "Dev Server" workflow which starts both the backend API and frontend development server on port 5000.

## Tech Stack
- React 18 with TypeScript
- Express.js backend
- Vite for build tooling
- Drizzle ORM for database operations
- MediaPipe for facial analysis
- Radix UI + Tailwind CSS for UI
- React Query for API state management