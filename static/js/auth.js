// Authentication utility class
class AuthService {
    constructor() {
        this.baseURL = window.location.origin;
        this.token = localStorage.getItem('access_token');
    }

    // Initiate Google OAuth login
    async initiateGoogleLogin() {
        try {
            this.showLoading();
            
            const response = await fetch(`${this.baseURL}/auth/google/login`);
            const data = await response.json();
            
            if (data.authorization_url) {
                // Redirect to Google OAuth
                window.location.href = data.authorization_url;
            } else {
                throw new Error('No authorization URL received');
            }
        } catch (error) {
            this.hideLoading();
            this.showError('Failed to initiate Google login: ' + error.message);
            console.error('Failed to initiate login:', error);
        }
    }

    // Handle OAuth callback (call this on your callback page)
    handleCallback() {
        const urlParams = new URLSearchParams(window.location.search);
        const token = urlParams.get('token');
        const email = urlParams.get('email');
        const name = urlParams.get('name');
        const error = urlParams.get('error');

        if (error) {
            this.showError('Authentication failed: ' + error);
            return null;
        }

        if (token) {
            this.setToken(token);
            return { token, email, name };
        }
        return null;
    }

    // Set authentication token
    setToken(token) {
        this.token = token;
        localStorage.setItem('access_token', token);
    }

    // Get authentication token
    getToken() {
        return this.token || localStorage.getItem('access_token');
    }

    // Remove authentication token
    logout() {
        this.token = null;
        localStorage.removeItem('access_token');
        localStorage.removeItem('user_email');
        localStorage.removeItem('user_name');
    }

    // Check if user is authenticated
    isAuthenticated() {
        return !!this.getToken();
    }

    // Get current user info
    async getCurrentUser() {
        const token = this.getToken();
        if (!token) throw new Error('No token available');

        const response = await fetch(`${this.baseURL}/auth/google/user`, {
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            if (response.status === 401) {
                this.logout();
                throw new Error('Authentication expired');
            }
            throw new Error('Failed to get user info');
        }

        return response.json();
    }

    // Make authenticated API calls
    async apiCall(endpoint, options = {}) {
        const token = this.getToken();
        if (!token) throw new Error('No token available');

        const config = {
            ...options,
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json',
                ...options.headers
            }
        };

        const response = await fetch(`${this.baseURL}${endpoint}`, config);
        
        if (response.status === 401) {
            this.logout();
            throw new Error('Authentication expired');
        }

        return response;
    }

    // Show loading overlay
    showLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.remove('hidden');
        }
    }

    // Hide loading overlay
    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.add('hidden');
        }
    }

    // Show error message
    showError(message) {
        // Create error notification
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-notification';
        errorDiv.innerHTML = `
            <div class="error-content">
                <span class="error-icon">⚠️</span>
                <span class="error-message">${message}</span>
                <button class="error-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        
        // Add error styles if not already present
        if (!document.getElementById('error-styles')) {
            const style = document.createElement('style');
            style.id = 'error-styles';
            style.textContent = `
                .error-notification {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: #fee;
                    border: 1px solid #fcc;
                    border-radius: 8px;
                    padding: 16px;
                    max-width: 400px;
                    z-index: 1001;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                    animation: slideIn 0.3s ease;
                }
                
                .error-content {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                }
                
                .error-icon {
                    font-size: 18px;
                }
                
                .error-message {
                    flex: 1;
                    color: #dc2626;
                    font-size: 14px;
                }
                
                .error-close {
                    background: none;
                    border: none;
                    font-size: 18px;
                    color: #dc2626;
                    cursor: pointer;
                    padding: 0;
                    width: 20px;
                    height: 20px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                
                @keyframes slideIn {
                    from {
                        transform: translateX(100%);
                        opacity: 0;
                    }
                    to {
                        transform: translateX(0);
                        opacity: 1;
                    }
                }
            `;
            document.head.appendChild(style);
        }
        
        document.body.appendChild(errorDiv);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentElement) {
                errorDiv.remove();
            }
        }, 5000);
    }

    // Show success message
    showSuccess(message) {
        const successDiv = document.createElement('div');
        successDiv.className = 'success-notification';
        successDiv.innerHTML = `
            <div class="success-content">
                <span class="success-icon">✅</span>
                <span class="success-message">${message}</span>
                <button class="success-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        
        // Add success styles if not already present
        if (!document.getElementById('success-styles')) {
            const style = document.createElement('style');
            style.id = 'success-styles';
            style.textContent = `
                .success-notification {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: #f0f9ff;
                    border: 1px solid #bae6fd;
                    border-radius: 8px;
                    padding: 16px;
                    max-width: 400px;
                    z-index: 1001;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                    animation: slideIn 0.3s ease;
                }
                
                .success-content {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                }
                
                .success-icon {
                    font-size: 18px;
                }
                
                .success-message {
                    flex: 1;
                    color: #0369a1;
                    font-size: 14px;
                }
                
                .success-close {
                    background: none;
                    border: none;
                    font-size: 18px;
                    color: #0369a1;
                    cursor: pointer;
                    padding: 0;
                    width: 20px;
                    height: 20px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
            `;
            document.head.appendChild(style);
        }
        
        document.body.appendChild(successDiv);
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            if (successDiv.parentElement) {
                successDiv.remove();
            }
        }, 3000);
    }
}

// Export singleton instance
window.authService = new AuthService();