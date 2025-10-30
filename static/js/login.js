// Login page functionality
document.addEventListener('DOMContentLoaded', function() {
    // Check if user is already authenticated
    if (authService.isAuthenticated()) {
        // Redirect to dashboard or home page
        window.location.href = '/dashboard';
        return;
    }

    // Handle OAuth callback if present
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has('token') || urlParams.has('error')) {
        handleOAuthCallback();
        return;
    }

    // Initialize login form handlers
    initializeLoginForm();
});

function handleOAuthCallback() {
    const result = authService.handleCallback();
    
    if (result) {
        // Store user data
        localStorage.setItem('user_email', result.email);
        localStorage.setItem('user_name', result.name);
        
        authService.showSuccess('Login successful! Redirecting...');
        
        // Redirect to dashboard after a short delay
        setTimeout(() => {
            window.location.href = '/dashboard';
        }, 1500);
    } else {
        // Handle error case
        const urlParams = new URLSearchParams(window.location.search);
        const error = urlParams.get('error');
        
        if (error) {
            authService.showError('Login failed: ' + error);
        } else {
            authService.showError('Login failed: No authentication token received');
        }
        
        // Clean up URL
        window.history.replaceState({}, document.title, window.location.pathname);
    }
}

function initializeLoginForm() {
    // Google login button
    const googleBtn = document.getElementById('google-login-btn');
    if (googleBtn) {
        googleBtn.addEventListener('click', async function(e) {
            e.preventDefault();
            await authService.initiateGoogleLogin();
        });
    }

    // Email form submission
    const emailForm = document.querySelector('.email-form');
    if (emailForm) {
        emailForm.addEventListener('submit', function(e) {
            e.preventDefault();
            handleEmailLogin();
        });
    }

    // Form validation
    const emailInput = document.getElementById('email');
    const passwordInput = document.getElementById('password');
    
    if (emailInput) {
        emailInput.addEventListener('blur', validateEmail);
        emailInput.addEventListener('input', clearValidationError);
    }
    
    if (passwordInput) {
        passwordInput.addEventListener('blur', validatePassword);
        passwordInput.addEventListener('input', clearValidationError);
    }
}

async function handleEmailLogin() {
    const email = document.getElementById('email').value.trim();
    const password = document.getElementById('password').value;
    const rememberMe = document.getElementById('remember').checked;

    // Clear previous errors
    clearFormErrors();

    // Validate inputs
    if (!validateEmailLogin(email, password)) {
        return;
    }

    try {
        authService.showLoading();

        // TODO: Implement email/password authentication endpoint
        const response = await fetch(`${authService.baseURL}/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                email,
                password,
                remember_me: rememberMe
            })
        });

        const data = await response.json();

        if (response.ok) {
            // Store token and user data
            authService.setToken(data.access_token);
            localStorage.setItem('user_email', data.user.email);
            localStorage.setItem('user_name', data.user.name || '');

            authService.showSuccess('Login successful! Redirecting...');
            
            setTimeout(() => {
                window.location.href = '/dashboard';
            }, 1500);
        } else {
            throw new Error(data.detail || 'Login failed');
        }
    } catch (error) {
        authService.showError('Login failed: ' + error.message);
        console.error('Email login error:', error);
    } finally {
        authService.hideLoading();
    }
}

function validateEmailLogin(email, password) {
    let isValid = true;

    if (!email) {
        showFieldError('email', 'Email is required');
        isValid = false;
    } else if (!isValidEmail(email)) {
        showFieldError('email', 'Please enter a valid email address');
        isValid = false;
    }

    if (!password) {
        showFieldError('password', 'Password is required');
        isValid = false;
    } else if (password.length < 6) {
        showFieldError('password', 'Password must be at least 6 characters');
        isValid = false;
    }

    return isValid;
}

function validateEmail() {
    const email = this.value.trim();
    if (email && !isValidEmail(email)) {
        showFieldError('email', 'Please enter a valid email address');
    } else {
        clearFieldError('email');
    }
}

function validatePassword() {
    const password = this.value;
    if (password && password.length < 6) {
        showFieldError('password', 'Password must be at least 6 characters');
    } else {
        clearFieldError('password');
    }
}

function clearValidationError() {
    clearFieldError(this.id);
}

function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

function showFieldError(fieldId, message) {
    const field = document.getElementById(fieldId);
    const formGroup = field.closest('.form-group');
    
    // Remove existing error
    clearFieldError(fieldId);
    
    // Add error styling
    field.style.borderColor = '#ef4444';
    field.style.backgroundColor = '#fef2f2';
    
    // Add error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'field-error';
    errorDiv.style.cssText = `
        color: #ef4444;
        font-size: 0.875rem;
        margin-top: 4px;
        display: flex;
        align-items: center;
        gap: 4px;
    `;
    errorDiv.innerHTML = `<span>⚠️</span> ${message}`;
    
    formGroup.appendChild(errorDiv);
}

function clearFieldError(fieldId) {
    const field = document.getElementById(fieldId);
    const formGroup = field.closest('.form-group');
    const existingError = formGroup.querySelector('.field-error');
    
    if (existingError) {
        existingError.remove();
    }
    
    // Reset field styling
    field.style.borderColor = '';
    field.style.backgroundColor = '';
}

function clearFormErrors() {
    const errors = document.querySelectorAll('.field-error');
    errors.forEach(error => error.remove());
    
    const fields = document.querySelectorAll('#email, #password');
    fields.forEach(field => {
        field.style.borderColor = '';
        field.style.backgroundColor = '';
    });
}

// Handle keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Enter key on login form
    if (e.key === 'Enter' && (e.target.id === 'email' || e.target.id === 'password')) {
        e.preventDefault();
        handleEmailLogin();
    }
});

// Add some UI enhancements
function addUIEnhancements() {
    // Add focus effects
    const inputs = document.querySelectorAll('input[type="email"], input[type="password"]');
    inputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
        });
        
        input.addEventListener('blur', function() {
            this.parentElement.classList.remove('focused');
        });
    });
}

// Initialize UI enhancements
addUIEnhancements();