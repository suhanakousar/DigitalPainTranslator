/**
 * REST API client for server-assisted pain assessment inference.
 * Handles communication with the FastAPI backend.
 */

class APIClient {
  constructor(baseURL = null) {
    // Use environment variable or default to localhost
    this.baseURL = baseURL || process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
    this.timeout = 10000; // 10 second timeout
  }

  /**
   * Make HTTP request with error handling
   */
  async makeRequest(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    
    const defaultOptions = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      timeout: this.timeout,
      ...options
    };

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);
      
      const response = await fetch(url, {
        ...defaultOptions,
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      if (error.name === 'AbortError') {
        throw new Error('Request timeout - server may be unavailable');
      }
      
      // Network or parsing errors
      throw new Error(`API request failed: ${error.message}`);
    }
  }

  /**
   * Perform pain assessment inference
   */
  async inferPainScore(requestData) {
    return await this.makeRequest('/api/infer', {
      method: 'POST',
      body: JSON.stringify(requestData)
    });
  }

  /**
   * Perform batch inference on multiple requests
   */
  async batchInference(requests) {
    return await this.makeRequest('/api/infer/batch', {
      method: 'POST',
      body: JSON.stringify(requests)
    });
  }

  /**
   * Get inference service status
   */
  async getInferenceStatus() {
    return await this.makeRequest('/api/infer/status');
  }

  /**
   * Create a new assessment record
   */
  async createRecord(recordData) {
    return await this.makeRequest('/api/records', {
      method: 'POST',
      body: JSON.stringify(recordData)
    });
  }

  /**
   * Get assessment records with optional filtering
   */
  async getRecords(params = {}) {
    const queryString = new URLSearchParams(params).toString();
    const endpoint = queryString ? `/api/records?${queryString}` : '/api/records';
    return await this.makeRequest(endpoint);
  }

  /**
   * Get a specific record by ID
   */
  async getRecord(recordId) {
    return await this.makeRequest(`/api/records/${recordId}`);
  }

  /**
   * Delete a record by ID
   */
  async deleteRecord(recordId) {
    return await this.makeRequest(`/api/records/${recordId}`, {
      method: 'DELETE'
    });
  }

  /**
   * Get records statistics
   */
  async getRecordsStatistics() {
    return await this.makeRequest('/api/records/stats/summary');
  }

  /**
   * Export records with filtering
   */
  async exportRecords(params = {}) {
    return await this.makeRequest('/api/records/export', {
      method: 'POST',
      body: JSON.stringify(params)
    });
  }

  /**
   * Health check
   */
  async healthCheck() {
    return await this.makeRequest('/api/health');
  }

  /**
   * Detailed health check
   */
  async detailedHealthCheck() {
    return await this.makeRequest('/api/health/detailed');
  }

  /**
   * Reload model
   */
  async reloadModel(modelPath = null, force = false) {
    return await this.makeRequest('/api/model/reload', {
      method: 'POST',
      body: JSON.stringify({
        model_path: modelPath,
        force: force
      })
    });
  }

  /**
   * Get model information
   */
  async getModelInfo() {
    return await this.makeRequest('/api/model/info');
  }

  /**
   * Get service metrics
   */
  async getMetrics() {
    return await this.makeRequest('/api/metrics');
  }

  /**
   * Test server connectivity
   */
  async testConnection() {
    try {
      await this.healthCheck();
      return { connected: true, error: null };
    } catch (error) {
      return { connected: false, error: error.message };
    }
  }

  /**
   * Validate request data before sending
   */
  validateInferenceRequest(requestData) {
    const errors = [];

    // Check that either landmarks or features are provided
    if (!requestData.landmarks && !requestData.features) {
      errors.push('Either landmarks or features must be provided');
    }

    if (requestData.landmarks && requestData.features) {
      errors.push('Provide either landmarks OR features, not both');
    }

    // Validate caregiver inputs
    if (!requestData.caregiverInputs) {
      errors.push('Caregiver inputs are required');
    } else {
      const caregiver = requestData.caregiverInputs;
      
      if (typeof caregiver.grimace !== 'number' || caregiver.grimace < 0 || caregiver.grimace > 5) {
        errors.push('Grimace must be a number between 0 and 5');
      }
      
      if (typeof caregiver.breathing !== 'number' || caregiver.breathing < 0 || caregiver.breathing > 5) {
        errors.push('Breathing must be a number between 0 and 5');
      }
      
      if (typeof caregiver.restlessness !== 'number' || caregiver.restlessness < 0 || caregiver.restlessness > 5) {
        errors.push('Restlessness must be a number between 0 and 5');
      }

      if (caregiver.gestures && !Array.isArray(caregiver.gestures)) {
        errors.push('Gestures must be an array');
      }
    }

    // Validate features if provided
    if (requestData.features) {
      const features = requestData.features;
      const requiredFeatures = ['mouthOpen', 'eyeClosureAvg', 'browFurrowAvg', 'headTiltVar', 'microMovementVar'];
      
      for (const feature of requiredFeatures) {
        if (typeof features[feature] !== 'number' || features[feature] < 0 || features[feature] > 1) {
          errors.push(`${feature} must be a number between 0 and 1`);
        }
      }
    }

    // Validate landmarks if provided
    if (requestData.landmarks) {
      if (!Array.isArray(requestData.landmarks)) {
        errors.push('Landmarks must be an array');
      } else {
        for (let i = 0; i < requestData.landmarks.length; i++) {
          const frame = requestData.landmarks[i];
          if (!Array.isArray(frame)) {
            errors.push(`Landmarks frame ${i} must be an array`);
            continue;
          }
          
          for (let j = 0; j < frame.length; j++) {
            const landmark = frame[j];
            if (!landmark || typeof landmark.x !== 'number' || typeof landmark.y !== 'number' || typeof landmark.z !== 'number') {
              errors.push(`Landmark ${j} in frame ${i} must have x, y, z coordinates`);
              break;
            }
          }
        }
      }
    }

    return {
      valid: errors.length === 0,
      errors: errors
    };
  }

  /**
   * Create a properly formatted inference request
   */
  createInferenceRequest(data) {
    const request = {
      session_id: data.session_id || crypto.randomUUID(),
      caregiverInputs: data.caregiverInputs,
      timestamp: new Date().toISOString()
    };

    if (data.landmarks) {
      request.landmarks = data.landmarks;
    } else if (data.features) {
      request.features = data.features;
    }

    return request;
  }

  /**
   * Handle server response and extract useful information
   */
  processInferenceResponse(response) {
    return {
      sessionId: response.session_id,
      painScore: response.score,
      confidence: response.confidence,
      explanation: response.explanation,
      recommendedActions: response.recommendedActions,
      modelVersion: response.model_version,
      processingTime: response.processing_ms,
      
      // Helper methods
      isHighConfidence: () => response.confidence >= 0.7,
      isHighPain: () => response.score >= 7.0,
      isModeratePain: () => response.score >= 4.0 && response.score < 7.0,
      isLowPain: () => response.score < 4.0,
      
      getTopContributors: (limit = 3) => 
        response.explanation
          .sort((a, b) => b.importance - a.importance)
          .slice(0, limit)
    };
  }
}

// Create singleton instance
const apiClient = new APIClient();

export default apiClient;