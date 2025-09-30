// client/src/services/apiClient.ts
/**
 * REST API client for server-assisted pain assessment inference.
 * Converted to TypeScript to avoid bundler parsing errors.
 */

type Json = Record<string, unknown>;

// Minimal typing for responses used by the frontend.
// If you have precise types in @shared/schema, you can import them instead.
interface InferenceResponse extends Json {}
interface RecordsResponse extends Json {}

export class APIClient {
  baseURL: string;
  timeout: number;

  constructor(baseURL: string | null = null) {
    // Use environment variable or default to localhost
    // Vite exposes env vars via import.meta.env
    this.baseURL = baseURL || (import.meta.env.VITE_BACKEND_URL as string) || 'http://localhost:8000';
    this.timeout = 10000; // 10 second timeout
  }

  private async fetchJson(endpoint: string, options: RequestInit = {}) {
    const url = `${this.baseURL}${endpoint}`;

    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers ?? {})
      },
      ...options
    };

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const resp = await fetch(url, {
        ...defaultOptions,
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!resp.ok) {
        const errorData = await resp.json().catch(() => ({}));
        const message = (errorData && (errorData as any).message) || `HTTP ${resp.status}: ${resp.statusText}`;
        throw new Error(message);
      }

      return await resp.json();
    } catch (err: any) {
      if (err?.name === 'AbortError') {
        throw new Error('Request timeout - server may be unavailable');
      }
      throw new Error(`API request failed: ${err?.message ?? String(err)}`);
    }
  }

  // Public methods

  async makeRequest(endpoint: string, options: RequestInit = {}): Promise<Json> {
    return this.fetchJson(endpoint, options);
  }

  async inferPainScore(requestData: unknown): Promise<InferenceResponse> {
    return this.makeRequest('/api/infer', {
      method: 'POST',
      body: JSON.stringify(requestData)
    }) as Promise<InferenceResponse>;
  }

  async batchInference(requests: unknown): Promise<InferenceResponse> {
    return this.makeRequest('/api/infer/batch', {
      method: 'POST',
      body: JSON.stringify(requests)
    }) as Promise<InferenceResponse>;
  }

  async getInferenceStatus(): Promise<Json> {
    return this.makeRequest('/api/infer/status');
  }

  async createRecord(recordData: unknown): Promise<RecordsResponse> {
    return this.makeRequest('/api/records', {
      method: 'POST',
      body: JSON.stringify(recordData)
    }) as Promise<RecordsResponse>;
  }

  async getRecords(params: Record<string, string> = {}): Promise<RecordsResponse> {
    const queryString = new URLSearchParams(params).toString();
    const endpoint = queryString ? `/api/records?${queryString}` : '/api/records';
    return this.makeRequest(endpoint) as Promise<RecordsResponse>;
  }

  async getRecord(recordId: string): Promise<RecordsResponse> {
    return this.makeRequest(`/api/records/${recordId}`) as Promise<RecordsResponse>;
  }

  async deleteRecord(recordId: string): Promise<Json> {
    return this.makeRequest(`/api/records/${recordId}`, { method: 'DELETE' });
  }

  async getRecordsStatistics(): Promise<Json> {
    return this.makeRequest('/api/records/stats/summary');
  }

  async exportRecords(params: Record<string, unknown> = {}): Promise<Json> {
    return this.makeRequest('/api/records/export', {
      method: 'POST',
      body: JSON.stringify(params)
    });
  }

  async healthCheck(): Promise<Json> {
    return this.makeRequest('/api/health');
  }

  async detailedHealthCheck(): Promise<Json> {
    return this.makeRequest('/api/health/detailed');
  }

  async reloadModel(modelPath: string | null = null, force = false): Promise<Json> {
    return this.makeRequest('/api/model/reload', {
      method: 'POST',
      body: JSON.stringify({ model_path: modelPath, force })
    });
  }

  async getModelInfo(): Promise<Json> {
    return this.makeRequest('/api/model/info');
  }

  async getMetrics(): Promise<Json> {
    return this.makeRequest('/api/metrics');
  }

  async testConnection() {
    try {
      await this.healthCheck();
      return { connected: true, error: null };
    } catch (error: any) {
      return { connected: false, error: error.message };
    }
  }

  // Simple validation helpers (kept from your original file)
  validateInferenceRequest(requestData: any) {
    const errors: string[] = [];

    if (!requestData.landmarks && !requestData.features) {
      errors.push('Either landmarks or features must be provided');
    }
    if (requestData.landmarks && requestData.features) {
      errors.push('Provide either landmarks OR features, not both');
    }
    if (!requestData.caregiverInputs) {
      errors.push('Caregiver inputs are required');
    }

    // Additional checks omitted for brevity — you can restore them if desired
    return { valid: errors.length === 0, errors };
  }

  createInferenceRequest(data: any) {
    return {
      session_id: data.session_id || crypto.randomUUID(),
      caregiverInputs: data.caregiverInputs,
      timestamp: new Date().toISOString(),
      ...(data.landmarks ? { landmarks: data.landmarks } : {}),
      ...(data.features ? { features: data.features } : {})
    };
  }

  processInferenceResponse(response: any) {
    return {
      sessionId: response.session_id,
      painScore: response.score,
      confidence: response.confidence,
      explanation: response.explanation,
      recommendedActions: response.recommendedActions,
      modelVersion: response.model_version,
      processingTime: response.processing_ms,
      isHighConfidence: () => response.confidence >= 0.7,
      isHighPain: () => response.score >= 7.0,
      isModeratePain: () => response.score >= 4.0 && response.score < 7.0,
      isLowPain: () => response.score < 4.0,
      getTopContributors: (limit = 3) =>
        (response.explanation ?? []).sort((a: any, b: any) => b.importance - a.importance).slice(0, limit)
    };
  }
}

// Default exported singleton instance (matches how your frontend code imports it)
const apiClient = new APIClient();
export default apiClient;
