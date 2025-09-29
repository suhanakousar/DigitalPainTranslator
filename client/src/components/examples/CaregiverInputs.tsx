import { useState } from 'react';
import CaregiverInputs from '../CaregiverInputs';
import type { CaregiverInput } from '@shared/schema';

export default function CaregiverInputsExample() {
  const [inputs, setInputs] = useState<CaregiverInput>({
    grimace: 2,
    breathing: 1,
    restlessness: 3,
    gestures: ['clench']
  });

  return (
    <div className="space-y-4">
      <CaregiverInputs
        value={inputs}
        onChange={(newInputs) => {
          console.log('Caregiver inputs updated:', newInputs);
          setInputs(newInputs);
        }}
      />
      
      <div className="p-4 bg-muted/50 rounded-lg">
        <h3 className="font-medium mb-2">Current Values (Demo)</h3>
        <pre className="text-xs">{JSON.stringify(inputs, null, 2)}</pre>
      </div>
    </div>
  );
}