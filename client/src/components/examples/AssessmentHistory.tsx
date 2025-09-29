import AssessmentHistory from '../AssessmentHistory';

// Mock some data in localStorage for demo
const mockAssessments = [
  {
    id: '1',
    timestamp: new Date(Date.now() - 3600000), // 1 hour ago
    painScore: 7.2,
    confidence: 85,
    topContributors: ['Brow Furrowing', 'Grimacing', 'Restlessness'],
    facialFeatures: {
      mouthOpen: 0.6,
      eyeClosureAvg: 0.4,
      browFurrowAvg: 0.8,
      headTiltVar: 0.3,
      microMovementVar: 0.5
    },
    caregiverInputs: {
      grimace: 4,
      breathing: 3,
      restlessness: 4,
      gestures: ['clench', 'shake']
    }
  },
  {
    id: '2',
    timestamp: new Date(Date.now() - 7200000), // 2 hours ago
    painScore: 4.1,
    confidence: 72,
    topContributors: ['Breathing Pattern', 'Micro Movements'],
    facialFeatures: {
      mouthOpen: 0.3,
      eyeClosureAvg: 0.2,
      browFurrowAvg: 0.4,
      headTiltVar: 0.2,
      microMovementVar: 0.6
    },
    caregiverInputs: {
      grimace: 2,
      breathing: 3,
      restlessness: 2,
      gestures: []
    }
  },
  {
    id: '3',
    timestamp: new Date(Date.now() - 14400000), // 4 hours ago
    painScore: 2.8,
    confidence: 68,
    topContributors: ['Restlessness', 'Eye Closure'],
    facialFeatures: {
      mouthOpen: 0.1,
      eyeClosureAvg: 0.3,
      browFurrowAvg: 0.2,
      headTiltVar: 0.1,
      microMovementVar: 0.3
    },
    caregiverInputs: {
      grimace: 1,
      breathing: 1,
      restlessness: 3,
      gestures: []
    }
  }
];

// Store mock data
if (typeof window !== 'undefined') {
  localStorage.setItem('pain_assessments', JSON.stringify(mockAssessments));
}

export default function AssessmentHistoryExample() {
  return (
    <div className="space-y-4">
      <AssessmentHistory />
      
      <div className="p-4 bg-muted/50 rounded-lg text-sm">
        <p className="font-medium mb-2">Demo Note:</p>
        <p>This component shows mock assessment data. In the real application, this would display actual recorded assessments from the pain analysis tool.</p>
      </div>
    </div>
  );
}