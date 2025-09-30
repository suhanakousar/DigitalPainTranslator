import { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Activity, History, Settings, Brain, Shield } from 'lucide-react';
import ConsentModal from './ConsentModal';
import CameraCapture from './CameraCapture';
import CaregiverInputs from './CaregiverInputs';
import PainScoreDisplay from './PainScoreDisplay';
import AssessmentHistory from './AssessmentHistory';
import { calculatePainScore } from '@/utils/painInference';
import { savePainAssessment } from '@/utils/localStorage';
import type { FacialFeatures, CaregiverInput } from '@shared/schema';
import type { PainAssessmentResult } from '@/utils/painInference';

export default function MainInterface() {
  const [hasConsent, setHasConsent] = useState(false);
  const [showConsentModal, setShowConsentModal] = useState(true);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [currentFacialFeatures, setCurrentFacialFeatures] = useState<FacialFeatures | null>(null);
  const [caregiverInputs, setCaregiverInputs] = useState<CaregiverInput>({
    grimace: 0,
    breathing: 0,
    restlessness: 0,
    gestures: []
  });
  const [currentResult, setCurrentResult] = useState<PainAssessmentResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  // Calculate pain score when we have both facial features and caregiver inputs
  useEffect(() => {
    let mounted = true;

    async function processPainScore() {
      if (currentFacialFeatures && hasConsent) {
        setIsProcessing(true);
        try {
          const result = await calculatePainScore(currentFacialFeatures, caregiverInputs);
          if (mounted) {
            setCurrentResult(result);
          }
        } catch (error) {
          console.error('Failed to process pain score:', error);
          // You might want to show an error message to the user here
        } finally {
          if (mounted) {
            setIsProcessing(false);
          }
        }
      }
    }
    
    processPainScore();

    return () => {
      mounted = false;
    };
  }, [currentFacialFeatures, caregiverInputs, hasConsent]);

  const handleConsentAccept = () => {
    setHasConsent(true);
    setShowConsentModal(false);
  };

  const handleConsentDecline = () => {
    setShowConsentModal(false);
    // Could redirect or show alternative interface
  };

  const handleFeaturesDetected = (features: FacialFeatures) => {
    setCurrentFacialFeatures(features);
  };

  const handleSaveAssessment = () => {
    if (currentResult && currentFacialFeatures) {
      savePainAssessment({
        facialFeatures: currentFacialFeatures,
        caregiverInputs,
        painScore: currentResult.painScore,
        confidence: currentResult.confidence,
        topContributors: currentResult.topContributors
      });
      
      // Show success feedback
      alert('Assessment saved successfully!');
    }
  };

  const resetAssessment = () => {
    setCaregiverInputs({
      grimace: 0,
      breathing: 0,
      restlessness: 0,
      gestures: []
    });
    setCurrentFacialFeatures(null);
    setCurrentResult(null);
    setIsCameraActive(false);
  };

  return (
    <div className="min-h-screen bg-background" data-testid="main-interface">
      {/* Header */}
      <header className="border-b bg-card/50 backdrop-blur supports-[backdrop-filter]:bg-card/50">
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-primary/10 p-2 rounded-lg">
                <Brain className="w-6 h-6 text-primary" />
              </div>
              <div>
                <h1 className="text-xl font-semibold">Digital Pain Translator</h1>
                <p className="text-sm text-muted-foreground">AI-Powered Pain Assessment Tool</p>
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="flex items-center gap-1">
                <Shield className="w-3 h-3" />
                Privacy First
              </Badge>
              {isProcessing && (
                <Badge variant="secondary" className="flex items-center gap-1">
                  <Activity className="w-3 h-3 animate-pulse" />
                  Processing
                </Badge>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        {hasConsent ? (
          <Tabs defaultValue="assessment" className="space-y-6">
            <TabsList className="grid w-full grid-cols-3" data-testid="tabs-main">
              <TabsTrigger value="assessment">Assessment</TabsTrigger>
              <TabsTrigger value="history">History</TabsTrigger>
              <TabsTrigger value="settings">Settings</TabsTrigger>
            </TabsList>

            <TabsContent value="assessment" className="space-y-6">
              {/* Assessment Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Left Column - Inputs */}
                <div className="space-y-6">
                  <CameraCapture
                    onFeaturesDetected={handleFeaturesDetected}
                    isActive={isCameraActive}
                    onToggle={() => setIsCameraActive(!isCameraActive)}
                  />
                  
                  <CaregiverInputs
                    value={caregiverInputs}
                    onChange={setCaregiverInputs}
                  />
                </div>

                {/* Right Column - Results */}
                <div className="space-y-6">
                  <PainScoreDisplay
                    result={currentResult}
                    onSave={handleSaveAssessment}
                  />

                  {/* Quick Actions */}
                  <Card>
                    <CardContent className="p-4">
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={resetAssessment}
                          className="flex-1"
                          data-testid="button-reset-assessment"
                        >
                          New Assessment
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => window.location.reload()}
                          className="flex-1"
                          data-testid="button-restart"
                        >
                          Restart App
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>

              {/* Assessment Status */}
              <Card className="bg-muted/30">
                <CardContent className="p-4">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                    <div>
                      <div className="text-lg font-semibold">
                        {currentFacialFeatures ? 'Active' : 'Inactive'}
                      </div>
                      <div className="text-xs text-muted-foreground">Facial Analysis</div>
                    </div>
                    <div>
                      <div className="text-lg font-semibold">
                        {Object.values(caregiverInputs).some(v => Array.isArray(v) ? v.length > 0 : v > 0) ? 'Complete' : 'Pending'}
                      </div>
                      <div className="text-xs text-muted-foreground">Caregiver Input</div>
                    </div>
                    <div>
                      <div className="text-lg font-semibold">
                        {currentResult ? currentResult.painScore : 'N/A'}
                      </div>
                      <div className="text-xs text-muted-foreground">Current Score</div>
                    </div>
                    <div>
                      <div className="text-lg font-semibold">
                        {currentResult ? `${currentResult.confidence}%` : 'N/A'}
                      </div>
                      <div className="text-xs text-muted-foreground">Confidence</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="history">
              <AssessmentHistory />
            </TabsContent>

            <TabsContent value="settings">
              <Card>
                <CardContent className="p-6">
                  <div className="space-y-4">
                    <h3 className="text-lg font-medium">Application Settings</h3>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Privacy Mode</span>
                        <Badge variant="secondary">Enabled</Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Local Storage</span>
                        <Badge variant="secondary">Active</Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Camera Permission</span>
                        <Badge variant="secondary">Granted</Badge>
                      </div>
                    </div>
                    
                    <div className="pt-4 border-t">
                      <Button
                        variant="outline"
                        onClick={() => {
                          setHasConsent(false);
                          setShowConsentModal(true);
                        }}
                        data-testid="button-revoke-consent"
                      >
                        Revoke Camera Consent
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        ) : (
          <div className="flex items-center justify-center min-h-96">
            <Card className="w-full max-w-md text-center">
              <CardContent className="p-6">
                <Brain className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                <h2 className="text-lg font-medium mb-2">Camera Access Required</h2>
                <p className="text-sm text-muted-foreground mb-4">
                  To perform pain assessment, camera access is required for facial analysis.
                </p>
                <Button onClick={() => setShowConsentModal(true)} data-testid="button-show-consent">
                  Grant Camera Access
                </Button>
              </CardContent>
            </Card>
          </div>
        )}
      </main>

      {/* Consent Modal */}
      <ConsentModal
        isOpen={showConsentModal}
        onAccept={handleConsentAccept}
        onDecline={handleConsentDecline}
      />
    </div>
  );
}