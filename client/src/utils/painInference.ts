// client/src/utils/painInference.ts
import type { FacialFeatures, CaregiverInput } from '@shared/schema';
import { APIClient } from '@/services/apiClient';

const apiClient = new APIClient();

/**
 * Weights used by the local pain inference fallback.
 * You can tune these values as needed.
 */
const PAIN_MODEL_WEIGHTS: {
  // caregiver inputs
  grimace: number;
  breathing: number;
  restlessness: number;

  // gestures
  clench: number;
  point: number;
  shake: number;

  // bias
  bias: number;
} = {
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

/**
 * Calculate pain score by first attempting a backend inference.
 * If the backend call fails or returns an unexpected result, we fall back
 * to a local heuristic-based inference using PAIN_MODEL_WEIGHTS.
 */
export async function calculatePainScore(
  facialFeatures: FacialFeatures,
  caregiverInputs: CaregiverInput
): Promise<PainAssessmentResult> {
  // Try backend inference first
  try {
    const response = await apiClient.makeRequest('/api/infer/pain-score', {
      method: 'POST',
      body: JSON.stringify({
        facialFeatures,
        caregiverInputs
      })
    });

    // Validate the response shape loosely
    if (
      response &&
      typeof response.painScore === 'number' &&
      typeof response.confidence === 'number' &&
      Array.isArray(response.topContributors) &&
      Array.isArray(response.recommendations)
    ) {
      return response as PainAssessmentResult;
    }
    // If response doesn't match expected shape, fall through to local inference
  } catch (err) {
    // Backend may be unreachable or returned an error — we'll do local inference below
    // (Do not throw here; fallback is intended)
    // console.debug('Backend inference failed, using local fallback', err);
  }

  // -------- Local inference fallback --------
  // contributions mapping from descriptive name -> numeric contribution
  const contributions: Record<string, number> = {
    'Grimace': 0,
    'Breathing Pattern': 0,
    'Restlessness': 0,
    'Brow Furrowing': 0,
    'Hand Clenching': 0,
    'Pointing Gesture': 0,
    'Head Shaking': 0
  };

  // Safe access to facial feature numbers (the schema may vary; we coerce if present)
  const ff = facialFeatures as unknown as Record<string, unknown>;
  const ci = caregiverInputs as unknown as Record<string, unknown>;

  // Grimace (caregiver-reported or computed)
  const grimaceVal = (ci.grimace ?? ff.grimace ?? 0) as number;
  if (typeof grimaceVal === 'number' && grimaceVal > 0) {
    contributions['Grimace'] = grimaceVal * PAIN_MODEL_WEIGHTS.grimace;
  }

  // Breathing pattern (numeric severity expected, 0..1 or similar)
  const breathingVal = (ci.breathing ?? ff.breathing ?? 0) as number;
  if (typeof breathingVal === 'number' && breathingVal > 0) {
    contributions['Breathing Pattern'] = breathingVal * PAIN_MODEL_WEIGHTS.breathing;
  }

  // Restlessness
  const restlessnessVal = (ci.restlessness ?? ff.restlessness ?? 0) as number;
  if (typeof restlessnessVal === 'number' && restlessnessVal > 0) {
    contributions['Restlessness'] = restlessnessVal * PAIN_MODEL_WEIGHTS.restlessness;
  }

  // Brow furrowing — try multiple possible field names safely
  const browVal =
    (ff.browFurrow as number | undefined) ??
    (ff.brow as number | undefined) ??
    0;
  if (typeof browVal === 'number' && browVal > 0) {
    contributions['Brow Furrowing'] = browVal * 1.6; // local tuning factor
  }

  // Gesture contributions from caregiverInputs.gestures (safe-guarding for missing fields)
  const gestures = Array.isArray((ci.gestures ?? caregiverInputs.gestures) as unknown)
    ? ((ci.gestures ?? caregiverInputs.gestures) as string[])
    : [];

  gestures.forEach((gesture) => {
    switch (gesture) {
      case 'clench':
        contributions['Hand Clenching'] = (contributions['Hand Clenching'] || 0) + PAIN_MODEL_WEIGHTS.clench;
        break;
      case 'point':
        contributions['Pointing Gesture'] = (contributions['Pointing Gesture'] || 0) + PAIN_MODEL_WEIGHTS.point;
        break;
      case 'shake':
        contributions['Head Shaking'] = (contributions['Head Shaking'] || 0) + PAIN_MODEL_WEIGHTS.shake;
        break;
      default:
        break;
    }
  });

  // Sum contributions
  const totalScore = Object.values(contributions).reduce((sum, val) => sum + val, 0) + PAIN_MODEL_WEIGHTS.bias;

  // Sigmoid scaled to 0-10
  const rawSig = 1 / (1 + Math.exp(-totalScore));
  const painScore = Math.max(0, Math.min(10, rawSig * 10));

  // Confidence heuristic: fraction of active contributors, clamped
  const nonZeroContributions = Object.values(contributions).filter((v) => v > 0.1);
  const confidence = Math.min(
    95,
    Math.max(
      45,
      (nonZeroContributions.length / Math.max(1, Object.keys(contributions).length)) * 100
    )
  );

  // Top 3 contributors
  const sortedContributions = Object.entries(contributions)
    .filter(([, value]) => value > 0.1)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 3)
    .map(([name]) => name);

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
    recommendations.push('Continue monitoring patient comfort');
    recommendations.push('Consider non-pharmacological interventions');
  } else if (painScore < 6) {
    recommendations.push('Consider mild pain management interventions');
    recommendations.push('Assess need for positioning changes');
    recommendations.push('Monitor breathing and comfort levels');
  } else if (painScore < 8) {
    recommendations.push('Administer prescribed pain medication');
    recommendations.push('Contact healthcare provider for assessment');
    recommendations.push('Consider immediate comfort measures');
  } else {
    recommendations.push('Urgent: Contact healthcare provider immediately');
    recommendations.push('Administer emergency pain protocol if available');
    recommendations.push('Document all observations and interventions');
  }

  // Add specific recommendations based on top contributors
  if (topContributors.includes('Breathing Pattern')) {
    recommendations.push('Focus on breathing support and positioning');
  }
  if (topContributors.includes('Restlessness')) {
    recommendations.push('Consider environmental modifications for comfort');
  }
  if (topContributors.includes('Brow Furrowing')) {
    recommendations.push('Assess for headache or concentration-related discomfort');
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
