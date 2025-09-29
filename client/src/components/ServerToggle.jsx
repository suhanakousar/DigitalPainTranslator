/**
 * Server Toggle Component - Controls local-only vs server-assisted mode
 * Implements explicit consent flow for server-assisted inference
 */
import React, { useState, useEffect } from 'react';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from '@/components/ui/alert-dialog';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { AlertTriangle, Server, Smartphone, Shield, Info } from 'lucide-react';
import apiClient from '@/services/apiClient';
import wsClient from '@/services/wsClient';

const ServerToggle = ({ onModeChange, className }) => {
  const [isServerMode, setIsServerMode] = useState(false);
  const [showConsentDialog, setShowConsentDialog] = useState(false);
  const [consentGiven, setConsentGiven] = useState(false);
  const [serverStatus, setServerStatus] = useState({ connected: false, loading: true });
  const [wsStatus, setWsStatus] = useState({ connected: false });

  // Load consent state from localStorage on mount
  useEffect(() => {
    const savedConsent = localStorage.getItem('pain-translator-server-consent');
    const savedMode = localStorage.getItem('pain-translator-inference-mode');
    
    if (savedConsent === 'granted') {
      setConsentGiven(true);
      if (savedMode === 'server') {
        setIsServerMode(true);
        testServerConnection();
      }
    }
  }, []);

  // Test server connectivity
  const testServerConnection = async () => {
    setServerStatus({ connected: false, loading: true });
    
    try {
      const result = await apiClient.testConnection();
      const wsResult = await wsClient.testConnection();
      
      setServerStatus({ 
        connected: result.connected, 
        loading: false, 
        error: result.error 
      });
      
      setWsStatus({ 
        connected: wsResult.connected, 
        error: wsResult.error 
      });
      
      if (!result.connected && isServerMode) {
        // Fallback to local mode if server is unavailable
        handleModeChange(false, true);
      }
    } catch (error) {
      setServerStatus({ 
        connected: false, 
        loading: false, 
        error: error.message 
      });
    }
  };

  // Handle mode change with consent check
  const handleToggle = (checked) => {
    if (checked && !consentGiven) {
      setShowConsentDialog(true);
    } else if (checked && consentGiven) {
      testServerConnection().then(() => {
        if (serverStatus.connected) {
          handleModeChange(true);
        }
      });
    } else {
      handleModeChange(false);
    }
  };

  // Handle consent and mode change
  const handleConsent = () => {
    setConsentGiven(true);
    setShowConsentDialog(false);
    
    // Save consent to localStorage
    localStorage.setItem('pain-translator-server-consent', 'granted');
    localStorage.setItem('pain-translator-consent-timestamp', new Date().toISOString());
    
    // Test connection and enable server mode
    testServerConnection().then(() => {
      if (serverStatus.connected) {
        handleModeChange(true);
      }
    });
  };

  // Handle mode change and notify parent
  const handleModeChange = (serverMode, isAutoFallback = false) => {
    setIsServerMode(serverMode);
    localStorage.setItem('pain-translator-inference-mode', serverMode ? 'server' : 'local');
    
    if (onModeChange) {
      onModeChange({
        mode: serverMode ? 'server' : 'local',
        serverAvailable: serverStatus.connected,
        wsAvailable: wsStatus.connected,
        isAutoFallback
      });
    }
  };

  // Revoke consent
  const revokeConsent = () => {
    setConsentGiven(false);
    setIsServerMode(false);
    localStorage.removeItem('pain-translator-server-consent');
    localStorage.removeItem('pain-translator-consent-timestamp');
    localStorage.setItem('pain-translator-inference-mode', 'local');
    
    wsClient.disconnect();
    
    if (onModeChange) {
      onModeChange({
        mode: 'local',
        serverAvailable: false,
        wsAvailable: false,
        consentRevoked: true
      });
    }
  };

  return (
    <>
      <Card className={className}>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-sm">
            <Shield className="h-4 w-4" />
            Inference Mode
          </CardTitle>
          <CardDescription className="text-xs">
            Control how pain assessment is performed
          </CardDescription>
        </CardHeader>
        
        <CardContent className="space-y-4">
          {/* Mode Toggle */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Label htmlFor="server-mode" className="text-sm font-medium">
                Server-assisted Mode
              </Label>
              {isServerMode && (
                <Badge variant="secondary" className="text-xs">
                  {serverStatus.connected ? 'Connected' : 'Disconnected'}
                </Badge>
              )}
            </div>
            
            <Switch
              id="server-mode"
              checked={isServerMode}
              onCheckedChange={handleToggle}
              disabled={serverStatus.loading || (isServerMode && !serverStatus.connected)}
            />
          </div>

          {/* Current Mode Display */}
          <div className="flex items-center gap-2 p-2 rounded-md bg-muted">
            {isServerMode ? (
              <>
                <Server className="h-4 w-4 text-blue-600" />
                <span className="text-sm">Server-assisted inference</span>
                {serverStatus.connected && wsStatus.connected && (
                  <Badge variant="outline" className="text-xs ml-auto">
                    Real-time
                  </Badge>
                )}
              </>
            ) : (
              <>
                <Smartphone className="h-4 w-4 text-green-600" />
                <span className="text-sm">Local-only processing</span>
                <Badge variant="outline" className="text-xs ml-auto">
                  Private
                </Badge>
              </>
            )}
          </div>

          {/* Connection Status */}
          {consentGiven && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span>API Connection:</span>
                <Badge variant={serverStatus.connected ? 'default' : 'destructive'}>
                  {serverStatus.loading ? 'Testing...' : (serverStatus.connected ? 'Online' : 'Offline')}
                </Badge>
              </div>
              
              <div className="flex items-center justify-between text-xs">
                <span>WebSocket:</span>
                <Badge variant={wsStatus.connected ? 'default' : 'destructive'}>
                  {wsStatus.connected ? 'Connected' : 'Disconnected'}
                </Badge>
              </div>
              
              {(!serverStatus.connected || !wsStatus.connected) && (
                <button
                  onClick={testServerConnection}
                  className="text-xs text-blue-600 hover:text-blue-800 underline"
                  disabled={serverStatus.loading}
                >
                  Test Connection
                </button>
              )}
            </div>
          )}

          {/* Error Display */}
          {serverStatus.error && (
            <div className="flex items-start gap-2 p-2 rounded-md bg-red-50 border border-red-200">
              <AlertTriangle className="h-4 w-4 text-red-600 mt-0.5 flex-shrink-0" />
              <div className="text-xs text-red-800">
                <div className="font-medium">Connection Error</div>
                <div className="mt-1">{serverStatus.error}</div>
              </div>
            </div>
          )}

          {/* Privacy Controls */}
          {consentGiven && (
            <div className="pt-2 border-t">
              <button
                onClick={revokeConsent}
                className="text-xs text-gray-600 hover:text-gray-800 underline"
              >
                Revoke server consent
              </button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Consent Dialog */}
      <AlertDialog open={showConsentDialog} onOpenChange={setShowConsentDialog}>
        <AlertDialogContent className="max-w-md">
          <AlertDialogHeader>
            <AlertDialogTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5" />
              Server-Assisted Mode Consent
            </AlertDialogTitle>
            <AlertDialogDescription className="space-y-3 text-sm">
              <div>
                You're about to enable server-assisted pain assessment. This mode provides:
              </div>
              
              <ul className="list-disc list-inside space-y-1 text-xs">
                <li>Enhanced ML model accuracy</li>
                <li>Real-time WebSocket inference</li>
                <li>Advanced explainability features</li>
                <li>Assessment history storage</li>
              </ul>
              
              <div className="bg-amber-50 border border-amber-200 rounded-md p-3">
                <div className="flex items-start gap-2">
                  <Info className="h-4 w-4 text-amber-600 mt-0.5 flex-shrink-0" />
                  <div className="text-xs text-amber-800">
                    <div className="font-medium">Privacy Notice</div>
                    <div className="mt-1">
                      When enabled, facial features and assessment data will be sent to the server for processing. 
                      No raw video data is transmitted. You can return to local-only mode at any time.
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="bg-red-50 border border-red-200 rounded-md p-3">
                <div className="flex items-start gap-2">
                  <AlertTriangle className="h-4 w-4 text-red-600 mt-0.5 flex-shrink-0" />
                  <div className="text-xs text-red-800">
                    <div className="font-medium">Research Use Only</div>
                    <div className="mt-1">
                      This system is a research prototype and should not be used for clinical diagnosis 
                      or medical decision-making without proper validation.
                    </div>
                  </div>
                </div>
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setShowConsentDialog(false)}>
              Cancel
            </AlertDialogCancel>
            <AlertDialogAction onClick={handleConsent}>
              I Understand & Consent
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
};

export default ServerToggle;