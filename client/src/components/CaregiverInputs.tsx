import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { User, Wind, Activity, Hand, MapPin, RotateCcw } from 'lucide-react';
import type { CaregiverInput } from '@shared/schema';

interface CaregiverInputsProps {
  value: CaregiverInput;
  onChange: (inputs: CaregiverInput) => void;
}

export default function CaregiverInputs({ value, onChange }: CaregiverInputsProps) {
  const updateSlider = (field: keyof Pick<CaregiverInput, 'grimace' | 'breathing' | 'restlessness'>, newValue: number[]) => {
    onChange({
      ...value,
      [field]: newValue[0]
    });
  };

  const toggleGesture = (gesture: 'clench' | 'point' | 'shake') => {
    const newGestures = value.gestures.includes(gesture)
      ? value.gestures.filter(g => g !== gesture)
      : [...value.gestures, gesture];
    
    onChange({
      ...value,
      gestures: newGestures
    });
  };

  const resetInputs = () => {
    onChange({
      grimace: 0,
      breathing: 0,
      restlessness: 0,
      gestures: []
    });
  };

  return (
    <Card className="w-full" data-testid="card-caregiver-inputs">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
        <CardTitle className="text-lg font-medium flex items-center gap-2">
          <User className="w-5 h-5" />
          Caregiver Assessment
        </CardTitle>
        <Button
          variant="outline"
          size="sm"
          onClick={resetInputs}
          data-testid="button-reset-inputs"
        >
          <RotateCcw className="w-4 h-4 mr-1" />
          Reset
        </Button>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Pain Indicators Sliders */}
        <div className="space-y-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="grimace-slider" className="text-sm font-medium flex items-center gap-2">
                <User className="w-4 h-4" />
                Grimacing
              </Label>
              <Badge variant="outline" className="text-xs" data-testid="badge-grimace-value">
                {value.grimace}/5
              </Badge>
            </div>
            <Slider
              id="grimace-slider"
              min={0}
              max={5}
              step={1}
              value={[value.grimace]}
              onValueChange={(val) => updateSlider('grimace', val)}
              className="w-full"
              data-testid="slider-grimace"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>None</span>
              <span>Severe</span>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="breathing-slider" className="text-sm font-medium flex items-center gap-2">
                <Wind className="w-4 h-4" />
                Breathing Pattern
              </Label>
              <Badge variant="outline" className="text-xs" data-testid="badge-breathing-value">
                {value.breathing}/5
              </Badge>
            </div>
            <Slider
              id="breathing-slider"
              min={0}
              max={5}
              step={1}
              value={[value.breathing]}
              onValueChange={(val) => updateSlider('breathing', val)}
              className="w-full"
              data-testid="slider-breathing"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Normal</span>
              <span>Labored</span>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="restlessness-slider" className="text-sm font-medium flex items-center gap-2">
                <Activity className="w-4 h-4" />
                Restlessness
              </Label>
              <Badge variant="outline" className="text-xs" data-testid="badge-restlessness-value">
                {value.restlessness}/5
              </Badge>
            </div>
            <Slider
              id="restlessness-slider"
              min={0}
              max={5}
              step={1}
              value={[value.restlessness]}
              onValueChange={(val) => updateSlider('restlessness', val)}
              className="w-full"
              data-testid="slider-restlessness"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Calm</span>
              <span>Very Restless</span>
            </div>
          </div>
        </div>

        {/* Gesture Buttons */}
        <div className="space-y-3">
          <Label className="text-sm font-medium">Observed Gestures</Label>
          <div className="grid grid-cols-3 gap-2">
            <Button
              variant={value.gestures.includes('clench') ? 'default' : 'outline'}
              size="sm"
              onClick={() => toggleGesture('clench')}
              className="flex items-center gap-2"
              data-testid="button-gesture-clench"
            >
              <Hand className="w-4 h-4" />
              Clench
            </Button>
            
            <Button
              variant={value.gestures.includes('point') ? 'default' : 'outline'}
              size="sm"
              onClick={() => toggleGesture('point')}
              className="flex items-center gap-2"
              data-testid="button-gesture-point"
            >
              <MapPin className="w-4 h-4" />
              Point
            </Button>
            
            <Button
              variant={value.gestures.includes('shake') ? 'default' : 'outline'}
              size="sm"
              onClick={() => toggleGesture('shake')}
              className="flex items-center gap-2"
              data-testid="button-gesture-shake"
            >
              <RotateCcw className="w-4 h-4" />
              Shake
            </Button>
          </div>
          
          {value.gestures.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {value.gestures.map((gesture) => (
                <Badge key={gesture} variant="secondary" className="text-xs">
                  {gesture}
                </Badge>
              ))}
            </div>
          )}
        </div>

        {/* Quick Summary */}
        <div className="bg-muted/50 p-3 rounded-lg">
          <h4 className="text-sm font-medium mb-2">Assessment Summary</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Grimace:</span>
              <span className="font-medium">{value.grimace}/5</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Breathing:</span>
              <span className="font-medium">{value.breathing}/5</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Restlessness:</span>
              <span className="font-medium">{value.restlessness}/5</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Gestures:</span>
              <span className="font-medium">{value.gestures.length}</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}