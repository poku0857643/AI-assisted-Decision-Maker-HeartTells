# Frontend Google OAuth Integration

## Backend Endpoints

### Authentication Flow
- `GET /auth/google/login` - Get Google authorization URL
- `GET /auth/google/callback` - API callback (returns JSON)
- `GET /auth/google/callback/frontend` - Frontend callback (redirects with token)
- `GET /auth/google/user` - Get current user (requires Bearer token)

## Frontend Integration Examples

### React/JavaScript Example

```javascript
// auth.js - Authentication utility
class AuthService {
  constructor() {
    this.baseURL = 'http://localhost:8000';
    this.token = localStorage.getItem('access_token');
  }

  // Initiate Google OAuth login
  async initiateGoogleLogin() {
    try {
      const response = await fetch(`${this.baseURL}/auth/google/login`);
      const data = await response.json();
      
      // Redirect to Google OAuth
      window.location.href = data.authorization_url;
    } catch (error) {
      console.error('Failed to initiate login:', error);
    }
  }

  // Handle OAuth callback (call this on your callback page)
  handleCallback() {
    const urlParams = new URLSearchParams(window.location.search);
    const token = urlParams.get('token');
    const email = urlParams.get('email');
    const name = urlParams.get('name');

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
}

// Export singleton instance
export const authService = new AuthService();
```

### React Components Example

```jsx
// LoginButton.jsx
import React from 'react';
import { authService } from './auth';

const LoginButton = () => {
  const handleLogin = () => {
    authService.initiateGoogleLogin();
  };

  return (
    <button onClick={handleLogin} className="google-login-btn">
      Sign in with Google
    </button>
  );
};

export default LoginButton;
```

```jsx
// AuthCallback.jsx - Handle OAuth callback
import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { authService } from './auth';

const AuthCallback = () => {
  const navigate = useNavigate();

  useEffect(() => {
    const handleCallback = async () => {
      try {
        const result = authService.handleCallback();
        
        if (result) {
          console.log('Login successful:', result);
          navigate('/dashboard'); // Redirect to dashboard
        } else {
          console.error('No token received');
          navigate('/login?error=no_token');
        }
      } catch (error) {
        console.error('Callback error:', error);
        navigate('/login?error=callback_failed');
      }
    };

    handleCallback();
  }, [navigate]);

  return <div>Processing login...</div>;
};

export default AuthCallback;
```

```jsx
// UserProfile.jsx - Display current user
import React, { useState, useEffect } from 'react';
import { authService } from './auth';

const UserProfile = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchUser = async () => {
      try {
        const userData = await authService.getCurrentUser();
        setUser(userData);
      } catch (error) {
        console.error('Failed to fetch user:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchUser();
  }, []);

  const handleLogout = () => {
    authService.logout();
    window.location.href = '/login';
  };

  if (loading) return <div>Loading...</div>;
  if (!user) return <div>Not logged in</div>;

  return (
    <div>
      <h2>Welcome, {user.email}</h2>
      <button onClick={handleLogout}>Logout</button>
    </div>
  );
};

export default UserProfile;
```

### App Router Setup

```jsx
// App.jsx - React Router setup
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LoginButton from './LoginButton';
import AuthCallback from './AuthCallback';
import UserProfile from './UserProfile';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/login" element={<LoginButton />} />
        <Route path="/auth/callback" element={<AuthCallback />} />
        <Route path="/dashboard" element={<UserProfile />} />
        <Route path="/" element={<LoginButton />} />
      </Routes>
    </Router>
  );
}

export default App;
```

## Configuration

### Backend Configuration
Set these environment variables:
```env
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback/frontend
SECRET_KEY=your_secret_key
```

### Google Cloud Console Setup
1. Create project in Google Cloud Console
2. Enable Google+ API
3. Create OAuth 2.0 credentials
4. Add authorized redirect URIs:
   - `http://localhost:8000/auth/google/callback`
   - `http://localhost:8000/auth/google/callback/frontend`

### Frontend CORS Origins
The backend is configured to allow these origins:
- `http://localhost:3000`
- `http://localhost:3002`
- `http://127.0.0.1:3000`
- `http://127.0.0.1:3002`
- `http://localhost:8080`

## API Usage Patterns

### Making Authenticated Requests
```javascript
// Example API call
try {
  const response = await authService.apiCall('/api/protected-endpoint', {
    method: 'POST',
    body: JSON.stringify({ data: 'example' })
  });
  
  const result = await response.json();
  console.log(result);
} catch (error) {
  console.error('API call failed:', error);
}
```

### Error Handling
```javascript
// Handle authentication errors
try {
  const user = await authService.getCurrentUser();
} catch (error) {
  if (error.message === 'Authentication expired') {
    // Redirect to login
    window.location.href = '/login';
  } else {
    console.error('Error:', error);
  }
}
```