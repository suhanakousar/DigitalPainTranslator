import type { PainAssessment } from '@shared/schema';

const STORAGE_KEY = 'pain_assessments';

export function savePainAssessment(assessment: Omit<PainAssessment, 'id' | 'timestamp'>): void {
  const assessments = getPainAssessments();
  const newAssessment: PainAssessment = {
    ...assessment,
    id: crypto.randomUUID(),
    timestamp: new Date(),
  };
  
  assessments.push(newAssessment);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(assessments));
}

export function getPainAssessments(): PainAssessment[] {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (!stored) return [];
  
  try {
    const assessments = JSON.parse(stored);
    return assessments.map((a: any) => ({
      ...a,
      timestamp: new Date(a.timestamp)
    }));
  } catch {
    return [];
  }
}

export function clearPainAssessments(): void {
  localStorage.removeItem(STORAGE_KEY);
}

export function exportToCSV(): string {
  const assessments = getPainAssessments();
  if (assessments.length === 0) return '';

  const headers = [
    'ID',
    'Timestamp',
    'Pain Score',
    'Confidence %',
    'Mouth Open',
    'Eye Closure Avg',
    'Brow Furrow Avg',
    'Head Tilt Var',
    'Micro Movement Var',
    'Grimace',
    'Breathing',
    'Restlessness',
    'Gestures',
    'Top Contributors'
  ];

  const rows = assessments.map(assessment => [
    assessment.id,
    assessment.timestamp.toISOString(),
    assessment.painScore.toString(),
    assessment.confidence.toString(),
    assessment.facialFeatures.mouthOpen.toFixed(3),
    assessment.facialFeatures.eyeClosureAvg.toFixed(3),
    assessment.facialFeatures.browFurrowAvg.toFixed(3),
    assessment.facialFeatures.headTiltVar.toFixed(3),
    assessment.facialFeatures.microMovementVar.toFixed(3),
    assessment.caregiverInputs.grimace.toString(),
    assessment.caregiverInputs.breathing.toString(),
    assessment.caregiverInputs.restlessness.toString(),
    assessment.caregiverInputs.gestures.join(';'),
    assessment.topContributors.join(';')
  ]);

  return [headers, ...rows]
    .map(row => row.map(cell => `"${cell}"`).join(','))
    .join('\n');
}

export function downloadCSV(): void {
  const csv = exportToCSV();
  if (!csv) {
    alert('No data to export');
    return;
  }

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  
  link.setAttribute('href', url);
  link.setAttribute('download', `pain_assessments_${new Date().toISOString().split('T')[0]}.csv`);
  link.style.visibility = 'hidden';
  
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}