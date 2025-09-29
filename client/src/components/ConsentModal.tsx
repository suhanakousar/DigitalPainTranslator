import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Shield, Camera, Database, Lock } from 'lucide-react';

interface ConsentModalProps {
  isOpen: boolean;
  onAccept: () => void;
  onDecline: () => void;
}

export default function ConsentModal({ isOpen, onAccept, onDecline }: ConsentModalProps) {
  const [hasReadPrivacy, setHasReadPrivacy] = useState(false);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center p-4 z-50" data-testid="modal-consent">
      <Card className="w-full max-w-2xl">
        <CardHeader className="text-center">
          <div className="flex justify-center mb-4">
            <div className="bg-primary/10 p-3 rounded-full">
              <Shield className="w-8 h-8 text-primary" />
            </div>
          </div>
          <CardTitle className="text-2xl">Privacy & Consent</CardTitle>
          <CardDescription className="text-base">
            Digital Pain Translator requires camera access for facial analysis
          </CardDescription>
        </CardHeader>
        
        <CardContent className="space-y-6">
          <div className="grid gap-4">
            <div className="flex items-start gap-3">
              <Camera className="w-5 h-5 text-primary mt-0.5" />
              <div>
                <h3 className="font-medium text-sm">Camera Access</h3>
                <p className="text-sm text-muted-foreground">
                  We need access to your camera to analyze facial micro-expressions for pain assessment.
                </p>
              </div>
            </div>
            
            <div className="flex items-start gap-3">
              <Lock className="w-5 h-5 text-primary mt-0.5" />
              <div>
                <h3 className="font-medium text-sm">Local Processing Only</h3>
                <p className="text-sm text-muted-foreground">
                  All video processing happens locally in your browser. No video data is transmitted or stored on servers.
                </p>
              </div>
            </div>
            
            <div className="flex items-start gap-3">
              <Database className="w-5 h-5 text-primary mt-0.5" />
              <div>
                <h3 className="font-medium text-sm">Local Storage</h3>
                <p className="text-sm text-muted-foreground">
                  Assessment results are stored locally on your device. You can export or delete them at any time.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-muted/50 p-4 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Badge variant="secondary" className="text-xs">Privacy First</Badge>
              <Badge variant="secondary" className="text-xs">HIPAA Compliant Design</Badge>
            </div>
            <p className="text-sm text-muted-foreground">
              This tool is designed for healthcare professionals and caregivers. All processing is done locally 
              to ensure maximum privacy and compliance with healthcare privacy standards.
            </p>
          </div>

          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="privacy-read"
              checked={hasReadPrivacy}
              onChange={(e) => setHasReadPrivacy(e.target.checked)}
              className="rounded border-border"
              data-testid="checkbox-privacy-read"
            />
            <label htmlFor="privacy-read" className="text-sm">
              I understand the privacy policy and consent to camera access for pain assessment
            </label>
          </div>

          <div className="flex gap-3 pt-2">
            <Button 
              variant="outline" 
              onClick={onDecline}
              className="flex-1"
              data-testid="button-consent-decline"
            >
              Decline
            </Button>
            <Button 
              onClick={onAccept}
              disabled={!hasReadPrivacy}
              className="flex-1"
              data-testid="button-consent-accept"
            >
              Accept & Continue
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}