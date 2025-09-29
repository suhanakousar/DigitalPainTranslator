import { useState } from 'react';
import PainScoreDisplay from '../PainScoreDisplay';
import type { PainAssessmentResult } from '@/utils/painInference';

export default function PainScoreDisplayExample() {
  const [showResult, setShowResult] = useState(true);

  // Mock result for demonstration
  const mockResult: PainAssessmentResult = {
    painScore: 6.7,
    confidence: 78,
    topContributors: ['Brow Furrowing', 'Grimacing', 'Restlessness'],
    recommendations: [
      'Consider mild pain management interventions',
      'Assess need for positioning changes', 
      'Monitor breathing and comfort levels',
      'Focus on breathing support and positioning'
    ]
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <button 
          onClick={() => setShowResult(!showResult)}
          className="px-3 py-1 text-sm bg-primary text-primary-foreground rounded"
        >
          {showResult ? 'Hide Result' : 'Show Result'}
        </button>
      </div>
      
      <PainScoreDisplay
        result={showResult ? mockResult : null}
        onSave={() => console.log('Assessment saved')}
      />
    </div>
  );
}