import { useState } from 'react';
import CameraCapture from '../CameraCapture';
import type { FacialFeatures } from '@shared/schema';

export default function CameraCaptureExample() {
  const [isActive, setIsActive] = useState(false);
  const [lastFeatures, setLastFeatures] = useState<FacialFeatures | null>(null);

  return (
    <div className="space-y-4">
      <CameraCapture
        onFeaturesDetected={(features) => {
          console.log('Features detected:', features);
          setLastFeatures(features);
        }}
        isActive={isActive}
        onToggle={() => setIsActive(!isActive)}
      />
      
      {lastFeatures && (
        <div className="p-4 bg-muted/50 rounded-lg">
          <h3 className="font-medium mb-2">Latest Features (Demo)</h3>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>Mouth Open: {(lastFeatures.mouthOpen * 100).toFixed(1)}%</div>
            <div>Eye Closure: {(lastFeatures.eyeClosureAvg * 100).toFixed(1)}%</div>
            <div>Brow Furrow: {(lastFeatures.browFurrowAvg * 100).toFixed(1)}%</div>
            <div>Head Tilt: {(lastFeatures.headTiltVar * 100).toFixed(1)}%</div>
          </div>
        </div>
      )}
    </div>
  );
}