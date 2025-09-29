import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Download, Trash2, Clock, TrendingUp } from 'lucide-react';
import { getPainAssessments, clearPainAssessments, downloadCSV } from '@/utils/localStorage';
import { getPainLevelColor } from '@/utils/painInference';
import type { PainAssessment } from '@shared/schema';

export default function AssessmentHistory() {
  const [assessments, setAssessments] = useState<PainAssessment[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadAssessments();
  }, []);

  const loadAssessments = () => {
    setIsLoading(true);
    const data = getPainAssessments();
    setAssessments(data.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()));
    setIsLoading(false);
  };

  const handleClearHistory = () => {
    if (window.confirm('Are you sure you want to clear all assessment history? This action cannot be undone.')) {
      clearPainAssessments();
      setAssessments([]);
    }
  };

  const handleExportCSV = () => {
    try {
      downloadCSV();
    } catch (error) {
      console.error('Export failed:', error);
      alert('Failed to export data. Please try again.');
    }
  };

  const formatTime = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(new Date(date));
  };

  const getPainLevelText = (score: number) => {
    if (score <= 3) return 'Mild';
    if (score <= 6) return 'Moderate';
    return 'Severe';
  };

  if (isLoading) {
    return (
      <Card className="w-full" data-testid="card-history-loading">
        <CardContent className="flex items-center justify-center h-32">
          <div className="text-muted-foreground">Loading assessment history...</div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full" data-testid="card-assessment-history">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
        <CardTitle className="text-lg font-medium flex items-center gap-2">
          <Clock className="w-5 h-5" />
          Assessment History
        </CardTitle>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-xs">
            {assessments.length} records
          </Badge>
          {assessments.length > 0 && (
            <>
              <Button
                variant="outline"
                size="sm"
                onClick={handleExportCSV}
                data-testid="button-export-csv"
              >
                <Download className="w-4 h-4 mr-1" />
                Export CSV
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleClearHistory}
                data-testid="button-clear-history"
              >
                <Trash2 className="w-4 h-4 mr-1" />
                Clear
              </Button>
            </>
          )}
        </div>
      </CardHeader>

      <CardContent>
        {assessments.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <TrendingUp className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p>No assessments recorded yet</p>
            <p className="text-sm">Start an assessment to see history here</p>
          </div>
        ) : (
          <div className="space-y-4">
            {/* Quick Stats */}
            <div className="grid grid-cols-3 gap-4 p-4 bg-muted/30 rounded-lg">
              <div className="text-center">
                <div className="text-2xl font-bold">
                  {(assessments.reduce((sum, a) => sum + a.painScore, 0) / assessments.length).toFixed(1)}
                </div>
                <div className="text-xs text-muted-foreground">Avg Pain Score</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold">
                  {Math.round(assessments.reduce((sum, a) => sum + a.confidence, 0) / assessments.length)}%
                </div>
                <div className="text-xs text-muted-foreground">Avg Confidence</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold">{assessments.length}</div>
                <div className="text-xs text-muted-foreground">Total Assessments</div>
              </div>
            </div>

            {/* Assessment Table */}
            <div className="border rounded-lg">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Time</TableHead>
                    <TableHead>Pain Score</TableHead>
                    <TableHead>Level</TableHead>
                    <TableHead>Confidence</TableHead>
                    <TableHead>Top Factor</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody data-testid="table-assessments">
                  {assessments.slice(0, 10).map((assessment) => (
                    <TableRow key={assessment.id}>
                      <TableCell className="font-mono text-sm">
                        {formatTime(assessment.timestamp)}
                      </TableCell>
                      <TableCell>
                        <span className={`font-bold ${getPainLevelColor(assessment.painScore)}`}>
                          {assessment.painScore}
                        </span>
                      </TableCell>
                      <TableCell>
                        <Badge 
                          variant={
                            assessment.painScore <= 3 ? 'secondary' : 
                            assessment.painScore <= 6 ? 'outline' : 
                            'destructive'
                          }
                          className="text-xs"
                        >
                          {getPainLevelText(assessment.painScore)}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <span className="text-sm">{assessment.confidence}%</span>
                          <div className="w-16 bg-muted rounded-full h-1.5">
                            <div 
                              className="bg-primary h-1.5 rounded-full" 
                              style={{ width: `${assessment.confidence}%` }}
                            />
                          </div>
                        </div>
                      </TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        {assessment.topContributors[0] || 'N/A'}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>

            {assessments.length > 10 && (
              <p className="text-xs text-muted-foreground text-center">
                Showing latest 10 assessments. Export CSV to view all {assessments.length} records.
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}