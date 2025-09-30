import type { FacialFeatures, CaregiverInput } from '@shared/schema';
import { APIClient } from '@/services/apiClient';

const apiClient = new APIClient();
  
  // Caregiver input weights
  grimace: 2.8,
  breathing: 2.2,
  restlessness: 2.6,
  
  // Gesture weights
  clench: 1.5,
  point: 1.0,
  shake: 2.0,
  
  // Base bias
  bias: -1.0
};

export interface PainAssessmentResult {
  painScore: number;
  confidence: number;
  topContributors: string[];
  recommendations: string[];
}

export async function calculatePainScore(
  facialFeatures: FacialFeatures,
  caregiverInputs: CaregiverInput
): Promise<PainAssessmentResult> {
  try {
    // Send data to backend for inference
    const response = await apiClient.makeRequest('/api/infer/pain-score', {
      method: 'POST',
      body: JSON.stringify({
        facialFeatures,
        caregiverInputs
      })
    });

    return response;

  // Add gesture contributions
  caregiverInputs.gestures.forEach(gesture => {
    switch (gesture) {
      case 'clench':
        contributions['Hand Clenching'] = PAIN_MODEL_WEIGHTS.clench;
        break;
      case 'point':
        contributions['Pointing Gesture'] = PAIN_MODEL_WEIGHTS.point;
        break;
      case 'shake':
        contributions['Head Shaking'] = PAIN_MODEL_WEIGHTS.shake;
        break;
    }
  });

  // Calculate total score
  const totalScore = Object.values(contributions).reduce((sum, val) => sum + val, 0) + PAIN_MODEL_WEIGHTS.bias;
  
  // Apply sigmoid function to constrain to 0-10 range
  const painScore = Math.max(0, Math.min(10, (1 / (1 + Math.exp(-totalScore))) * 10));

  // Calculate confidence based on consistency of signals
  const nonZeroContributions = Object.values(contributions).filter(val => val > 0.1);
  const confidence = Math.min(95, Math.max(45, 
    (nonZeroContributions.length / Object.keys(contributions).length) * 100
  ));

  // Get top 3 contributors
  const sortedContributions = Object.entries(contributions)
    .filter(([_, value]) => value > 0.1)
    .sort(([_, a], [__, b]) => b - a)
    .slice(0, 3)
    .map(([name, _]) => name);

  // Generate recommendations based on pain level
  const recommendations = generateRecommendations(painScore, sortedContributions);

  return {
    painScore: Math.round(painScore * 10) / 10,
    confidence: Math.round(confidence),
    topContributors: sortedContributions,
    recommendations
  };
}

function generateRecommendations(painScore: number, topContributors: string[]): string[] {
  const recommendations: string[] = [];

  if (painScore < 3) {
    recommendations.push("Continue monitoring patient comfort");
    recommendations.push("Consider non-pharmacological interventions");
  } else if (painScore < 6) {
    recommendations.push("Consider mild pain management interventions");
    recommendations.push("Assess need for positioning changes");
    recommendations.push("Monitor breathing and comfort levels");
  } else if (painScore < 8) {
    recommendations.push("Administer prescribed pain medication");
    recommendations.push("Contact healthcare provider for assessment");
    recommendations.push("Consider immediate comfort measures");
  } else {
    recommendations.push("Urgent: Contact healthcare provider immediately");
    recommendations.push("Administer emergency pain protocol if available");
    recommendations.push("Document all observations and interventions");
  }

  // Add specific recommendations based on top contributors
  if (topContributors.includes('Breathing Pattern')) {
    recommendations.push("Focus on breathing support and positioning");
  }
  if (topContributors.includes('Restlessness')) {
    recommendations.push("Consider environmental modifications for comfort");
  }
  if (topContributors.includes('Brow Furrowing')) {
    recommendations.push("Assess for headache or concentration-related discomfort");
  }

  return recommendations.slice(0, 4); // Limit to 4 recommendations
}

export function getPainLevelColor(painScore: number): string {
  if (painScore <= 3) return 'text-chart-1'; // Green
  if (painScore <= 6) return 'text-chart-2'; // Yellow/Orange
  return 'text-chart-3'; // Red
}

export function getPainLevelBgColor(painScore: number): string {
  if (painScore <= 3) return 'bg-chart-1/10';
  if (painScore <= 6) return 'bg-chart-2/10';
  return 'bg-chart-3/10';
}