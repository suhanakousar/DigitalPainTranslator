/**
 * WebSocket client for real-time pain assessment inference.
 * Provides low-latency communication with the FastAPI backend.
 */

class WebSocketClient {
  constructor(wsURL = null) {
    this.wsURL = wsURL || process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws/infer';
    this.ws = null;
    this.isConnected = false;
    this.sessionId = crypto.randomUUID();
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000; // Start with 1 second
    this.maxReconnectDelay = 30000; // Max 30 seconds
    this.heartbeatInterval = null;
    this.heartbeatTimeout = null;
    
    // Event handlers
    this.onConnectionOpen = null;
    this.onConnectionClose = null;
    this.onInferenceResult = null;
    this.onError = null;
    this.onReconnecting = null;
  }

  /**
   * Connect to WebSocket server
   */
  connect() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return Promise.resolve();
    }

    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.wsURL);
        
        this.ws.onopen = (event) => {
          console.log('WebSocket connected');
          this.isConnected = true;
          this.reconnectAttempts = 0;
          this.reconnectDelay = 1000;
          
          // Start heartbeat
          this.startHeartbeat();
          
          if (this.onConnectionOpen) {
            this.onConnectionOpen(event);
          }
          
          resolve();
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event);
        };

        this.ws.onclose = (event) => {
          console.log('WebSocket disconnected:', event.code, event.reason);
          this.isConnected = false;
          this.stopHeartbeat();
          
          if (this.onConnectionClose) {
            this.onConnectionClose(event);
          }

          // Attempt reconnection if not closed intentionally
          if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = (event) => {
          console.error('WebSocket error:', event);
          
          if (this.onError) {
            this.onError(new Error('WebSocket connection error'));
          }
          
          reject(new Error('WebSocket connection failed'));
        };

      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect() {
    if (this.ws) {
      this.reconnectAttempts = this.maxReconnectAttempts; // Prevent reconnection
      this.stopHeartbeat();
      this.ws.close(1000, 'User initiated disconnect');
      this.ws = null;
      this.isConnected = false;
    }
  }

  /**
   * Send inference request via WebSocket
   */
  async sendInferenceRequest(requestData) {
    if (!this.isConnected) {
      throw new Error('WebSocket not connected');
    }

    const message = {
      type: 'inference',
      session_id: this.sessionId,
      data: requestData,
      timestamp: new Date().toISOString()
    };

    return new Promise((resolve, reject) => {
      try {
        this.ws.send(JSON.stringify(message));
        
        // Set up one-time response handler
        const originalHandler = this.onInferenceResult;
        this.onInferenceResult = (response) => {
          // Restore original handler
          this.onInferenceResult = originalHandler;
          
          if (response.session_id === this.sessionId) {
            resolve(response);
          }
          
          // Call original handler if it exists
          if (originalHandler) {
            originalHandler(response);
          }
        };
        
        // Set timeout for response
        setTimeout(() => {
          this.onInferenceResult = originalHandler;
          reject(new Error('WebSocket inference timeout'));
        }, 10000);
        
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Handle incoming WebSocket messages
   */
  handleMessage(event) {
    try {
      const message = JSON.parse(event.data);
      
      switch (message.type) {
        case 'connection':
          console.log('WebSocket welcome:', message.message);
          break;
          
        case 'inference_result':
          if (this.onInferenceResult) {
            this.onInferenceResult(message.data);
          }
          break;
          
        case 'pong':
          // Heartbeat response received
          if (this.heartbeatTimeout) {
            clearTimeout(this.heartbeatTimeout);
            this.heartbeatTimeout = null;
          }
          break;
          
        case 'error':
          console.error('WebSocket server error:', message);
          if (this.onError) {
            this.onError(new Error(message.message || 'Server error'));
          }
          break;
          
        default:
          console.warn('Unknown WebSocket message type:', message.type);
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
      if (this.onError) {
        this.onError(new Error('Invalid message format'));
      }
    }
  }

  /**
   * Start heartbeat mechanism
   */
  startHeartbeat() {
    this.stopHeartbeat(); // Clear any existing heartbeat
    
    this.heartbeatInterval = setInterval(() => {
      if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
        const pingMessage = {
          type: 'ping',
          session_id: this.sessionId,
          timestamp: new Date().toISOString()
        };
        
        this.ws.send(JSON.stringify(pingMessage));
        
        // Set timeout to detect if server is not responding
        this.heartbeatTimeout = setTimeout(() => {
          console.warn('Heartbeat timeout - connection may be lost');
          this.ws.close();
        }, 5000);
      }
    }, 30000); // Send ping every 30 seconds
  }

  /**
   * Stop heartbeat mechanism
   */
  stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
    
    if (this.heartbeatTimeout) {
      clearTimeout(this.heartbeatTimeout);
      this.heartbeatTimeout = null;
    }
  }

  /**
   * Schedule reconnection attempt
   */
  scheduleReconnect() {
    this.reconnectAttempts++;
    
    if (this.reconnectAttempts > this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    console.log(`Reconnecting in ${this.reconnectDelay}ms (attempt ${this.reconnectAttempts})`);
    
    if (this.onReconnecting) {
      this.onReconnecting(this.reconnectAttempts, this.maxReconnectAttempts);
    }

    setTimeout(() => {
      this.connect().catch((error) => {
        console.error('Reconnection failed:', error);
        
        // Exponential backoff
        this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
        
        // Schedule next attempt
        this.scheduleReconnect();
      });
    }, this.reconnectDelay);
  }

  /**
   * Get connection status
   */
  getStatus() {
    return {
      connected: this.isConnected,
      readyState: this.ws ? this.ws.readyState : WebSocket.CLOSED,
      sessionId: this.sessionId,
      reconnectAttempts: this.reconnectAttempts,
      wsURL: this.wsURL
    };
  }

  /**
   * Update session ID
   */
  setSessionId(sessionId) {
    this.sessionId = sessionId;
  }

  /**
   * Test WebSocket connectivity
   */
  async testConnection() {
    try {
      await this.connect();
      return { connected: true, error: null };
    } catch (error) {
      return { connected: false, error: error.message };
    }
  }

  /**
   * Send custom message
   */
  sendMessage(type, data) {
    if (!this.isConnected) {
      throw new Error('WebSocket not connected');
    }

    const message = {
      type: type,
      session_id: this.sessionId,
      data: data,
      timestamp: new Date().toISOString()
    };

    this.ws.send(JSON.stringify(message));
  }

  /**
   * Set event handlers
   */
  setEventHandlers(handlers) {
    this.onConnectionOpen = handlers.onOpen || null;
    this.onConnectionClose = handlers.onClose || null;
    this.onInferenceResult = handlers.onInferenceResult || null;
    this.onError = handlers.onError || null;
    this.onReconnecting = handlers.onReconnecting || null;
  }
}

// Create singleton instance
const wsClient = new WebSocketClient();

export default wsClient;