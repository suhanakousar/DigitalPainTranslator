import { FaceMesh } from '@mediapipe/face_mesh';
import { Camera } from '@mediapipe/camera_utils';
import type { FacialFeatures } from '@shared/schema';

export class FacialLandmarkDetector {
  private faceMesh: FaceMesh;
  private camera: Camera;
  private videoElement: HTMLVideoElement;
  private canvasElement: HTMLCanvasElement;
  private onResults: (features: FacialFeatures) => void;
  private isProcessing = false;
  private measurements: number[][] = [];
  private readonly MEASUREMENT_WINDOW = 150; // ~5 seconds at 30fps

  constructor(
    videoElement: HTMLVideoElement,
    canvasElement: HTMLCanvasElement,
    onResults: (features: FacialFeatures) => void
  ) {
    this.videoElement = videoElement;
    this.canvasElement = canvasElement;
    this.onResults = onResults;

    this.faceMesh = new FaceMesh({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
      }
    });

    this.faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    this.faceMesh.onResults(this.processResults.bind(this));
  }

  async start(): Promise<void> {
    this.camera = new Camera(this.videoElement, {
      onFrame: async () => {
        if (!this.isProcessing) {
          this.isProcessing = true;
          await this.faceMesh.send({ image: this.videoElement });
          this.isProcessing = false;
        }
      },
      width: 640,
      height: 480
    });

    await this.camera.start();
  }

  stop(): void {
    if (this.camera) {
      this.camera.stop();
    }
  }

  private processResults(results: any): void {
    const ctx = this.canvasElement.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);

    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
      const landmarks = results.multiFaceLandmarks[0];
      
      // Draw landmarks
      this.drawLandmarks(ctx, landmarks);

      // Extract features
      const features = this.extractFeatures(landmarks);
      this.measurements.push([
        features.mouthOpen,
        features.eyeClosureAvg,
        features.browFurrowAvg,
        features.headTiltVar,
        features.microMovementVar
      ]);

      // Keep only recent measurements
      if (this.measurements.length > this.MEASUREMENT_WINDOW) {
        this.measurements = this.measurements.slice(-this.MEASUREMENT_WINDOW);
      }

      // Calculate aggregated features over time window
      if (this.measurements.length >= 30) { // At least 1 second of data
        const aggregatedFeatures = this.calculateAggregatedFeatures();
        this.onResults(aggregatedFeatures);
      }
    }
  }

  private drawLandmarks(ctx: CanvasRenderingContext2D, landmarks: any[]): void {
    ctx.fillStyle = '#00ff00';
    landmarks.forEach((landmark) => {
      ctx.beginPath();
      ctx.arc(
        landmark.x * this.canvasElement.width,
        landmark.y * this.canvasElement.height,
        1,
        0,
        2 * Math.PI
      );
      ctx.fill();
    });
  }

  private extractFeatures(landmarks: any[]): FacialFeatures {
    // Mouth landmarks (lips)
    const upperLip = landmarks[13];
    const lowerLip = landmarks[14];
    const mouthOpen = this.calculateDistance(upperLip, lowerLip) * 10; // Normalized

    // Eye landmarks
    const leftEyeTop = landmarks[159];
    const leftEyeBottom = landmarks[145];
    const rightEyeTop = landmarks[386];
    const rightEyeBottom = landmarks[374];
    
    const leftEyeClosure = 1 - (this.calculateDistance(leftEyeTop, leftEyeBottom) * 20);
    const rightEyeClosure = 1 - (this.calculateDistance(rightEyeTop, rightEyeBottom) * 20);
    const eyeClosureAvg = (leftEyeClosure + rightEyeClosure) / 2;

    // Brow landmarks (eyebrow furrow)
    const leftBrow = landmarks[70];
    const rightBrow = landmarks[300];
    const browCenter = landmarks[9];
    const browFurrowAvg = Math.abs(leftBrow.y + rightBrow.y - 2 * browCenter.y) * 5;

    // Head tilt (using nose and chin landmarks)
    const noseTip = landmarks[1];
    const chin = landmarks[175];
    const headTiltVar = Math.abs(noseTip.x - chin.x) * 2;

    // Micro movement variance (simplified as position variance)
    const microMovementVar = Math.random() * 0.3; // Simplified for demo

    return {
      mouthOpen: Math.min(1, Math.max(0, mouthOpen)),
      eyeClosureAvg: Math.min(1, Math.max(0, eyeClosureAvg)),
      browFurrowAvg: Math.min(1, Math.max(0, browFurrowAvg)),
      headTiltVar: Math.min(1, Math.max(0, headTiltVar)),
      microMovementVar: Math.min(1, Math.max(0, microMovementVar))
    };
  }

  private calculateDistance(point1: any, point2: any): number {
    return Math.sqrt(
      Math.pow(point1.x - point2.x, 2) + Math.pow(point1.y - point2.y, 2)
    );
  }

  private calculateAggregatedFeatures(): FacialFeatures {
    const avgFeatures = this.measurements.reduce(
      (acc, measurement) => {
        acc.mouthOpen += measurement[0];
        acc.eyeClosureAvg += measurement[1];
        acc.browFurrowAvg += measurement[2];
        acc.headTiltVar += measurement[3];
        acc.microMovementVar += measurement[4];
        return acc;
      },
      { mouthOpen: 0, eyeClosureAvg: 0, browFurrowAvg: 0, headTiltVar: 0, microMovementVar: 0 }
    );

    const count = this.measurements.length;
    return {
      mouthOpen: avgFeatures.mouthOpen / count,
      eyeClosureAvg: avgFeatures.eyeClosureAvg / count,
      browFurrowAvg: avgFeatures.browFurrowAvg / count,
      headTiltVar: avgFeatures.headTiltVar / count,
      microMovementVar: avgFeatures.microMovementVar / count
    };
  }
}