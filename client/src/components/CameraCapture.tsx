import { useRef, useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Camera, CameraOff, Activity } from 'lucide-react';
import { FacialLandmarkDetector } from '@/utils/facialLandmarks';
import type { FacialFeatures } from '@shared/schema';

interface CameraCaptureProps {
  onFeaturesDetected: (features: FacialFeatures) => void;
  isActive: boolean;
  onToggle: () => void;
}

export default function CameraCapture({ onFeaturesDetected, isActive, onToggle }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [detector, setDetector] = useState<FacialLandmarkDetector | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentFeatures, setCurrentFeatures] = useState<FacialFeatures | null>(null);

  useEffect(() => {
    if (isActive && videoRef.current && canvasRef.current) {
      initializeCamera();
    } else if (!isActive && detector) {
      detector.stop();
      setDetector(null);
      setIsReady(false);
    }

    return () => {
      if (detector) {
        detector.stop();
      }
    };
  }, [isActive]);

  const initializeCamera = async () => {
    try {
      setError(null);
      
      if (!videoRef.current || !canvasRef.current) return;

      const newDetector = new FacialLandmarkDetector(
        videoRef.current,
        canvasRef.current,
        (features) => {
          setCurrentFeatures(features);
          onFeaturesDetected(features);
        }
      );

      await newDetector.start();
      setDetector(newDetector);
      setIsReady(true);
    } catch (err) {
      setError('Failed to access camera. Please ensure camera permissions are granted.');
      console.error('Camera initialization error:', err);
    }
  };

  const getFeatureStatus = (value: number, threshold = 0.3) => {
    return value > threshold ? 'Active' : 'Normal';
  };

  const getFeatureColor = (value: number, threshold = 0.3) => {
    return value > threshold ? 'destructive' : 'secondary';
  };

  return (
    <Card className="w-full" data-testid="card-camera-capture">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-lg font-medium">Camera Analysis</CardTitle>
        <div className="flex items-center gap-2">
          {isReady && (
            <Badge variant="secondary" className="text-xs">
              <Activity className="w-3 h-3 mr-1" />
              Detecting
            </Badge>
          )}
          <Button
            variant={isActive ? "destructive" : "default"}
            size="sm"
            onClick={onToggle}
            data-testid="button-camera-toggle"
          >
            {isActive ? <CameraOff className="w-4 h-4" /> : <Camera className="w-4 h-4" />}
            {isActive ? 'Stop' : 'Start'}
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="relative bg-muted rounded-lg overflow-hidden">
          <video
            ref={videoRef}
            className="w-full h-64 object-cover"
            autoPlay
            muted
            playsInline
            style={{ display: isActive ? 'block' : 'none' }}
            data-testid="video-camera-feed"
          />
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-64"
            width={640}
            height={480}
            style={{ display: isActive ? 'block' : 'none' }}
            data-testid="canvas-landmarks"
          />
          {!isActive && (
            <div className="flex items-center justify-center h-64 text-muted-foreground">
              <div className="text-center">
                <Camera className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>Click Start to begin facial analysis</p>
              </div>
            </div>
          )}
        </div>

        {error && (
          <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-lg">
            <p className="text-sm text-destructive">{error}</p>
          </div>
        )}

        {currentFeatures && (
          <div className="grid grid-cols-2 gap-2" data-testid="features-display">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Mouth Opening</span>
              <Badge variant={getFeatureColor(currentFeatures.mouthOpen)} className="text-xs">
                {getFeatureStatus(currentFeatures.mouthOpen)}
              </Badge>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Eye Closure</span>
              <Badge variant={getFeatureColor(currentFeatures.eyeClosureAvg)} className="text-xs">
                {getFeatureStatus(currentFeatures.eyeClosureAvg)}
              </Badge>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Brow Furrow</span>
              <Badge variant={getFeatureColor(currentFeatures.browFurrowAvg)} className="text-xs">
                {getFeatureStatus(currentFeatures.browFurrowAvg)}
              </Badge>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Head Movement</span>
              <Badge variant={getFeatureColor(currentFeatures.headTiltVar)} className="text-xs">
                {getFeatureStatus(currentFeatures.headTiltVar)}
              </Badge>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}