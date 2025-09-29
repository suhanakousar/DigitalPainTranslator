import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { AlertTriangle, CheckCircle, AlertCircle, TrendingUp, Save } from 'lucide-react';
import { getPainLevelColor, getPainLevelBgColor } from '@/utils/painInference';
import type { PainAssessmentResult } from '@/utils/painInference';

interface PainScoreDisplayProps {
  result: PainAssessmentResult | null;
  onSave?: () => void;
}

export default function PainScoreDisplay({ result, onSave }: PainScoreDisplayProps) {
  if (!result) {
    return (
      <Card className="w-full" data-testid="card-pain-score-empty">
        <CardContent className="flex items-center justify-center h-64 text-center">
          <div className="space-y-2">
            <TrendingUp className="w-12 h-12 mx-auto text-muted-foreground opacity-50" />
            <p className="text-muted-foreground">Start assessment to see pain score</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const getPainLevelIcon = (score: number) => {
    if (score <= 3) return <CheckCircle className="w-6 h-6 text-chart-1" />;
    if (score <= 6) return <AlertCircle className="w-6 h-6 text-chart-2" />;
    return <AlertTriangle className="w-6 h-6 text-chart-3" />;
  };

  const getPainLevelText = (score: number) => {
    if (score <= 3) return 'Mild';
    if (score <= 6) return 'Moderate';
    return 'Severe';
  };

  const getUrgencyLevel = (score: number) => {
    if (score <= 3) return 'Monitor';
    if (score <= 6) return 'Assess';
    return 'Urgent';
  };

  return (
    <Card className="w-full" data-testid="card-pain-score-result">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
        <CardTitle className="text-lg font-medium">Pain Assessment Result</CardTitle>
        {onSave && (
          <Button
            variant="outline"
            size="sm"
            onClick={onSave}
            data-testid="button-save-assessment"
          >
            <Save className="w-4 h-4 mr-1" />
            Save
          </Button>
        )}
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Main Pain Score Display */}
        <div className={`text-center p-6 rounded-lg ${getPainLevelBgColor(result.painScore)}`}>
          <div className="flex items-center justify-center gap-3 mb-2">
            {getPainLevelIcon(result.painScore)}
            <div>
              <div className={`text-4xl font-bold ${getPainLevelColor(result.painScore)}`} data-testid="text-pain-score">
                {result.painScore}
              </div>
              <div className="text-sm text-muted-foreground">out of 10</div>
            </div>
          </div>
          
          <div className="flex items-center justify-center gap-2">
            <Badge 
              variant={result.painScore <= 3 ? 'secondary' : result.painScore <= 6 ? 'outline' : 'destructive'}
              className="text-sm"
            >
              {getPainLevelText(result.painScore)} Pain
            </Badge>
            <Badge variant="outline" className="text-sm">
              {getUrgencyLevel(result.painScore)}
            </Badge>
          </div>
        </div>

        {/* Confidence Score */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Confidence Level</span>
            <span className="text-sm font-bold" data-testid="text-confidence-score">
              {result.confidence}%
            </span>
          </div>
          <Progress value={result.confidence} className="h-2" />
          <p className="text-xs text-muted-foreground">
            {result.confidence >= 80 ? 'High confidence in assessment' :
             result.confidence >= 60 ? 'Moderate confidence - consider additional observation' :
             'Low confidence - recommend extended observation period'}
          </p>
        </div>

        {/* Top Contributing Factors */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium">Top Contributing Factors</h4>
          <div className="space-y-2" data-testid="list-contributors">
            {result.topContributors.map((contributor, index) => (
              <div key={contributor} className="flex items-center gap-2">
                <Badge variant="outline" className="text-xs w-6 h-6 rounded-full p-0 flex items-center justify-center">
                  {index + 1}
                </Badge>
                <span className="text-sm">{contributor}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Recommendations */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium">Recommended Actions</h4>
          <div className="space-y-2" data-testid="list-recommendations">
            {result.recommendations.map((recommendation, index) => (
              <div key={index} className="flex items-start gap-2 p-2 bg-muted/30 rounded-md">
                <div className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                <span className="text-sm">{recommendation}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Quick Action Buttons */}
        <div className="grid grid-cols-2 gap-2 pt-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => console.log('Document assessment')}
            data-testid="button-document-assessment"
          >
            Document
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => console.log('Contact provider')}
            data-testid="button-contact-provider"
          >
            Contact Provider
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}