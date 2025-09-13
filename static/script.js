// Configuration - Auto-detect API URL
const API_URL = (() => {
    const ports = [5001, 5002, 5003, 8000, 8080];
    const currentPort = window.location.port || '5001';
    const host = window.location.hostname; // Use the same host as the frontend
    return `http://${host}:${currentPort}/api`;
})();
const FRAME_INTERVAL = 200; // Process a frame every 200ms (configurable via backend)
const ALERT_DURATION = 3000; // Alert stays visible for 3 seconds
const DROWSY_COUNTER_THRESHOLD = 3; // Count of consecutive drowsy detections to trigger alert

// Global variables
let video = document.getElementById('videoElement');
let drowsyOverlay = document.getElementById('drowsyOverlay');
let startBtn = document.getElementById('startBtn');
let calibrateBtn = document.getElementById('calibrateBtn');
let stopBtn = document.getElementById('stopBtn');
let eyeStatus = document.getElementById('eyeStatus');
let yawnStatus = document.getElementById('yawnStatus');
let faceStatus = document.getElementById('faceStatus');
let earValue = document.getElementById('earValue');
let thresholdValue = document.getElementById('thresholdValue');
let eyeStatusIndicator = document.getElementById('eyeStatusIndicator');
let yawnStatusIndicator = document.getElementById('yawnStatusIndicator');
let faceStatusIndicator = document.getElementById('faceStatusIndicator');
let statusMessage = document.getElementById('statusMessage');
let serverStatus = document.getElementById('serverStatus');
let serverStatusIndicator = document.getElementById('serverStatusIndicator');
let sensitivitySlider = document.getElementById('sensitivitySlider');
let sensitivityValue = document.getElementById('sensitivityValue');
let thresholdSlider = document.getElementById('thresholdSlider');
let nightModeToggle = document.getElementById('nightModeToggle');
let soundToggle = document.getElementById('soundToggle');
let alertCount = document.getElementById('alertCount');
let monitorTime = document.getElementById('monitorTime');
let avgEAR = document.getElementById('avgEAR');
let faceConfidence = document.getElementById('faceConfidence');
let resetStatsBtn = document.getElementById('resetStatsBtn');
let toggleTroubleshoot = document.getElementById('toggleTroubleshoot');
let troubleshootContent = document.getElementById('troubleshootContent');
let testConnectionBtn = document.getElementById('testConnectionBtn');
let resetAppBtn = document.getElementById('resetAppBtn');
let darkModeToggle = document.getElementById('darkModeToggle');
let calibrationProgress = document.getElementById('calibrationProgress');
let calibrationTime = document.getElementById('calibrationTime');
let progressBarFill = document.querySelector('.progress-bar-fill');

// Statistics tracking
let stats = {
    alertCount: 0,
    monitorStartTime: null,
    earValues: [],
    faceConfidenceValues: []
};

let stream = null;
let processingInterval = null;
let drowsyCounter = 0;
let alertActive = false;
let alertTimeout = null;
let monitorTimeInterval = null;

// Initialize the application
window.addEventListener('load', () => {
    startBtn.addEventListener('click', startDetection);
    calibrateBtn.addEventListener('click', calibrateEyes);
    stopBtn.addEventListener('click', stopDetection);
    
    // System Controls Functionality
    sensitivitySlider.addEventListener('input', updateSensitivity);
    thresholdSlider.addEventListener('input', updateThreshold);
    nightModeToggle.addEventListener('change', toggleNightMode);
    soundToggle.addEventListener('change', toggleSound);
    
    // Statistics Functionality
    resetStatsBtn.addEventListener('click', resetStatistics);
    
    // Troubleshooting Functionality
    toggleTroubleshoot.addEventListener('click', toggleTroubleshootingPanel);
    testConnectionBtn.addEventListener('click', testConnection);
    resetAppBtn.addEventListener('click', resetApplication);
    
    // Dark Mode Functionality
    darkModeToggle.addEventListener('change', toggleDarkMode);
    
    // Initialize values
    updateSensitivity();
    updateThreshold();
    
    // Check if backend is available
    checkBackendStatus();
    
    // Apply stored preferences (if any)
    loadUserPreferences();
});

// Load user preferences from localStorage
function loadUserPreferences() {
    // Dark mode preference
    const darkModePreference = localStorage.getItem('darkMode');
    if (darkModePreference !== null) {
        darkModeToggle.checked = darkModePreference === 'true';
        toggleDarkMode();
    }
    
    // Sound preference
    const soundPreference = localStorage.getItem('soundEnabled');
    if (soundPreference !== null) {
        soundToggle.checked = soundPreference === 'true';
    }
    
    // Night mode preference
    const nightModePreference = localStorage.getItem('nightMode');
    if (nightModePreference !== null) {
        nightModeToggle.checked = nightModePreference === 'true';
    }
    
    // Sensitivity preference
    const sensitivityPreference = localStorage.getItem('sensitivity');
    if (sensitivityPreference !== null) {
        sensitivitySlider.value = sensitivityPreference;
        updateSensitivity();
    }
    
    // Threshold preference
    const thresholdPreference = localStorage.getItem('threshold');
    if (thresholdPreference !== null) {
        thresholdSlider.value = thresholdPreference;
        updateThreshold();
    }
}

// Check if the backend server is running
async function checkBackendStatus() {
    try {
        // Updated to use the new API status endpoint
        const response = await fetch(`${API_URL}/status`);
        const data = await response.json();
        
        if (data.status === 'API is running') {
            updateStatusMessage('Connected to backend server', 'success');
            updateServerStatus(true);
            
            if (!data.model_loaded) {
                updateStatusMessage('Warning: Model not loaded - yawning detection will use fallback method', 'warning');
                // Don't disable buttons - app works without model
            }
        }
    } catch (err) {
        console.error('Error connecting to backend:', err);
        updateStatusMessage('Cannot connect to backend server. Make sure it is running at the same address as this page.', 'error');
        updateServerStatus(false);
        startBtn.disabled = true;
        calibrateBtn.disabled = true;
    }
}

// Update server connection status
function updateServerStatus(connected) {
    serverStatus.textContent = connected ? 'Connected' : 'Disconnected';
    serverStatusIndicator.className = 'status-indicator ' + (connected ? 'status-ok' : 'status-warning');
}

// Update status message with animation
function updateStatusMessage(message, type = 'info') {
    if (!statusMessage) return;
    
    statusMessage.textContent = message;
    statusMessage.classList.remove('hidden', 'scale-95', 'opacity-0');
    
    // Set background color based on type
    statusMessage.className = 'px-4 py-3 rounded-lg mb-4 shadow-lg transform transition-all duration-300';
    
    if (type === 'error') {
        statusMessage.classList.add('bg-red-900', 'text-white');
    } else if (type === 'success') {
        statusMessage.classList.add('bg-green-900', 'text-white');
    } else if (type === 'warning') {
        statusMessage.classList.add('bg-yellow-900', 'text-white');
    } else {
        statusMessage.classList.add('bg-blue-900', 'text-white');
    }
    
    // Hide the message after 5 seconds
    setTimeout(() => {
        statusMessage.classList.add('scale-95', 'opacity-0');
        setTimeout(() => {
            statusMessage.classList.add('hidden');
        }, 300);
    }, 5000);
}

// Start the webcam and drowsiness detection
async function startDetection() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.style.transform = 'scaleX(-1)'; // Mirror the video
        
        startBtn.disabled = true;
        calibrateBtn.disabled = true;
        stopBtn.disabled = false;
        
        updateStatusMessage('Detection started. System is monitoring for drowsiness.', 'info');
        
        // Start processing frames at regular intervals
        processingInterval = setInterval(processFrame, FRAME_INTERVAL);
        
        // Start monitoring time
        stats.monitorStartTime = Date.now();
        monitorTimeInterval = setInterval(updateMonitorTime, 1000);
    } catch (err) {
        console.error('Error accessing webcam:', err);
        updateStatusMessage('Error accessing webcam. Please ensure you have a camera connected and have granted permission.', 'error');
    }
}

// Stop detection and release resources
function stopDetection() {
    clearInterval(processingInterval);
    clearInterval(monitorTimeInterval);
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    
    video.srcObject = null;
    stream = null;
    
    startBtn.disabled = false;
    calibrateBtn.disabled = false;
    stopBtn.disabled = true;
    
    updateStatusMessage('Detection stopped', 'info');
    
    // Reset status
    updateStatus({
        face_detected: false,
        eyes_closed: false,
        yawning: false
    });
    
    // Calculate final statistics
    if (stats.earValues.length > 0) {
        const avgEarValue = stats.earValues.reduce((sum, val) => sum + val, 0) / stats.earValues.length;
        avgEAR.textContent = avgEarValue.toFixed(2);
    }
    
    if (stats.faceConfidenceValues.length > 0) {
        const avgConfidence = stats.faceConfidenceValues.reduce((sum, val) => sum + val, 0) / stats.faceConfidenceValues.length;
        faceConfidence.textContent = `${Math.round(avgConfidence)}%`;
    }
}

// Calibrate the eye threshold by collecting frames of closed eyes
async function calibrateEyes() {
    try {
        if (!stream) {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
        }
        
        updateStatusMessage('Calibration started. Please close your eyes lightly for 5 seconds...', 'info');
        calibrateBtn.disabled = true;
        startBtn.disabled = true;
        
        // Show calibration progress bar
        calibrationProgress.classList.remove('hidden');
        
        let frames = [];
        let calibrationDuration = 5000; // 5 seconds
        let frameInterval = 200; // Collect a frame every 200ms
        let startTime = Date.now();
        
        // Collect frames during calibration period
        let calibrationInterval = setInterval(async () => {
            const elapsedTime = Date.now() - startTime;
            const remainingTime = Math.ceil((calibrationDuration - elapsedTime) / 1000);
            const progressPercentage = (elapsedTime / calibrationDuration) * 100;
            
            // Update progress bar
            progressBarFill.style.width = `${progressPercentage}%`;
            calibrationTime.textContent = `${remainingTime}s`;
            
            if (elapsedTime >= calibrationDuration) {
                clearInterval(calibrationInterval);
                
                // Hide calibration progress bar
                calibrationProgress.classList.add('hidden');
                
                updateStatusMessage('Processing calibration data...', 'info');
                
                // Send collected frames to server for calibration
                try {
                    const response = await fetch(`${API_URL}/calibrate`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ frames })
                    });
                    
                    const data = await response.json();
                    
                    if (data.threshold) {
                        thresholdSlider.value = data.threshold;
                        thresholdValue.textContent = data.threshold.toFixed(3);
                        updateStatusMessage(`Calibration complete! Eye threshold set to ${data.threshold.toFixed(3)}`, 'success');
                        
                        // Save to preferences
                        localStorage.setItem('threshold', data.threshold);
                    } else {
                        updateStatusMessage('Calibration failed: ' + (data.error || 'Unknown error'), 'error');
                    }
                } catch (err) {
                    console.error('Error during calibration:', err);
                    updateStatusMessage(`Error during calibration: ${err.message}. Please try again.`, 'error');
                }
                
                calibrateBtn.disabled = false;
                startBtn.disabled = false;
                
                // Stop the stream if detection hasn't started
                if (!processingInterval) {
                    stream.getTracks().forEach(track => track.stop());
                    stream = null;
                    video.srcObject = null;
                }
            } else {
                // Update status with countdown
                updateStatusMessage(`Calibrating... Keep eyes closed for ${remainingTime} more seconds`, 'info');
                
                // Capture current frame
                const canvas = document.createElement('canvas');
                const context = canvas.getContext('2d');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                // Add frame to collection
                frames.push(canvas.toDataURL('image/jpeg', 0.8));
            }
        }, frameInterval);
        
    } catch (err) {
        console.error('Error during calibration:', err);
        updateStatusMessage('Error accessing webcam for calibration. Please ensure you have a camera connected and have granted permission.', 'error');
        calibrateBtn.disabled = false;
        startBtn.disabled = false;
    }
}

// Process a single frame and send to backend for drowsiness detection
async function processFrame() {
    if (!video.videoWidth) return; // Video not ready yet
    
    // Capture current frame
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to base64 and send to backend
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    
    try {
        // Add current threshold and night mode settings to request
        const response = await fetch(`${API_URL}/detect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                image: imageData,
                eye_threshold: parseFloat(thresholdSlider.value),
                night_mode: nightModeToggle.checked,
                sensitivity: parseInt(sensitivitySlider.value)
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            console.error('Error from backend:', data.error);
            updateStatusMessage(`Detection error: ${data.error}`, 'warning');
            return;
        }
        
        // Update status indicators
        updateStatus(data);
        
        // Update statistics with memory management
        if (data.eye_aspect_ratio !== undefined) {
            stats.earValues.push(data.eye_aspect_ratio);
            
            // Keep only last 100 values to prevent memory leak (configurable)
            const maxValues = 100; // This could be fetched from backend config
            if (stats.earValues.length > maxValues) {
                stats.earValues = stats.earValues.slice(-maxValues);
            }
            
            // Calculate rolling average of last 10 EAR values
            const recentEarValues = stats.earValues.slice(-10);
            const avgEarValue = recentEarValues.reduce((sum, val) => sum + val, 0) / recentEarValues.length;
            avgEAR.textContent = avgEarValue.toFixed(2);
        }
        
        if (data.face_confidence !== undefined) {
            stats.faceConfidenceValues.push(data.face_confidence);
            
            // Keep only last 100 values to prevent memory leak (configurable)
            const maxValues = 100; // This could be fetched from backend config
            if (stats.faceConfidenceValues.length > maxValues) {
                stats.faceConfidenceValues = stats.faceConfidenceValues.slice(-maxValues);
            }
            
            faceConfidence.textContent = `${Math.round(data.face_confidence)}%`;
        }
        
        // Handle drowsiness alert
        if (data.drowsy) {
            drowsyCounter++;
            if (drowsyCounter >= DROWSY_COUNTER_THRESHOLD && !alertActive) {
                triggerAlert();
            }
        } else {
            drowsyCounter = 0;
        }
        
    } catch (err) {
        console.error('Error processing frame:', err);
        updateStatusMessage('Connection to backend lost. Try restarting the application.', 'error');
        stopDetection();
    }
}

// Update the UI with current detection status
function updateStatus(data) {
    // Update face detection status
    faceStatus.textContent = data.face_detected ? 'Detected' : 'Not detected';
    faceStatusIndicator.className = 'status-indicator ' + (data.face_detected ? 'status-ok' : 'status-warning');
    
    if (data.face_detected) {
        // Update eye status
        eyeStatus.textContent = data.eyes_closed ? 'Closed' : 'Open';
        eyeStatusIndicator.className = 'status-indicator ' + (data.eyes_closed ? 'status-warning' : 'status-ok');
        
        if (data.eye_aspect_ratio !== undefined) {
            earValue.textContent = data.eye_aspect_ratio.toFixed(3);
        }
        
        // Update yawn status
        yawnStatus.textContent = data.yawning ? 'Yawning' : 'Normal';
        yawnStatusIndicator.className = 'status-indicator ' + (data.yawning ? 'status-warning' : 'status-ok');
        
        // Update threshold if provided
        if (data.eye_threshold !== undefined) {
            thresholdValue.textContent = data.eye_threshold.toFixed(3);
        }
    } else {
        // Reset statuses when face not detected
        eyeStatus.textContent = 'N/A';
        yawnStatus.textContent = 'N/A';
        earValue.textContent = 'N/A';
        eyeStatusIndicator.className = 'status-indicator status-warning';
        yawnStatusIndicator.className = 'status-indicator status-warning';
    }
}

// Show the drowsy alert
function triggerAlert() {
    // Check if sound is enabled
    if (soundToggle.checked) {
        // Play alert sound
        const audio = new Audio('data:audio/wav;base64,UklGRnQHAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YU8HAACAgICAgICAgICAgICAgICAgICAgICAgIA9PT09PT09PT09PT09PT09PT09PT09PYCAgICAgICAgICAgICAgICAgICAgICAgD09PT09PT09PT09PT09PT09PT09PT09PYCAgICAgICAgICAgICAgICAgICAgICAgD09PT09PT09PT09PT09PT09PT09PT09PYCAgICAgICAgICAgICAgICAgICAgICAgD09PT09PT09PT09PT09PT09PT09PT09PYCAgICAgICAgICAgICAgICAgICAgICAgD09PT09PT09PT09PT09PT09PT09PT09PT09PT2AgICAgICAgICAgICAgICAgICAgICAPT09PT09PT09PT09PT09PT09PT09PT09PT09PYCAgICAgICAgICAgICAgICAgICAgIA9PT09PT09PT09PT09PT09PT09PT09PT09PT09gICAgICAgICAgICAgICAgICAgICAgD09PT09PT09PT09PT09PT09PT09PT09PT09PT2AgICAgICAgICAgICAgICAgICAgICA9PT09PT09PT09PT09PT09PT09PT09gICAgICAgICAgICAgICAgICAgICAgICA9PT09PT09PT09PT09PT09PT09PT09gICAgICAgICAgICAgICAgICAgICAgICA9PT09PT09PT09PT09PT09PT09PT09gICAgICAgICAgICAgICAgICAgICAgICA9PT09PT09PT09PT09PT09PT09PT09gICAgICAgICAgICAgICAgICAgICAgICA9PT09PT09PT09PT09PT09PT09PT09gICAgICAgICAgICAgICAgICAgICAgID09PT09PT09PT09PT09PT09PT09PT2AgICAgICAgICAgICAgICAgICAgICAgPT09PT09PT09PT09PT09PT09PT09PYCAgICAgICAgICAgICAgICAgICAgICAgD09PT09PT09PT09PT09PT09PT09PYCAgICAgICAgICAgICAgICAgICAgICAgIA9PT09PT09PT09PT09PT09PT09PT2AgICAgICAgICAgICAgICAgICAgICAgICA9PT09PYCAgICAgICAgICAgICAgICAgICAgICAgPT09PT2AgICAgICAgICAgICAgICAgICAgICAgID09PT09gICAgICAgICAgICAgICAgICAgICAgICA9PT09PYCAgICAgICAgICAgICAgICAgICAgICAgPT09PT2AgICAgICAgICAgICAgICAgICAgICAgID09PT09PT09PT09PT09PT09PT09PT09PT09PT2AgICAgICAgICAgICAgICAgICAgICAPT09PT09PT09PT09PT09PT09PT09PT09PT09PYCAgICAgICAgICAgICAgICAgICAgIA9PT09PT09PT09PT09PT09PT09PT09PT09PT09gICAgICAgICAgICAgICAgICAgICAgD09PT09PT09PT09PT09PT09PT09PT09PT09PT2AgICAgICAgICAgICAgICAgICAgICA9PT09PT09PT09PT09PT09PT09PT09PT09PT09gICAgICAgICAgICAgICAgICAgICAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAgICAgICAgICAgICAgICAgICAgICAgIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA');
        audio.play();
    }
    
    // Show visual alert with animation
    drowsyOverlay.classList.add('alert-active');
    alertActive = true;
    
    // Clear previous timeout if exists
    if (alertTimeout) {
        clearTimeout(alertTimeout);
    }
    
    // Hide alert after duration
    alertTimeout = setTimeout(() => {
        drowsyOverlay.classList.remove('alert-active');
        alertActive = false;
    }, ALERT_DURATION);
    
    // Update alert count
    stats.alertCount++;
    alertCount.textContent = stats.alertCount;
    
    // Apply pulse animation to alert counter
    alertCount.parentElement.classList.add('pulse-animation');
    setTimeout(() => {
        alertCount.parentElement.classList.remove('pulse-animation');
    }, 2000);
}

// System Controls Functions
function updateSensitivity() {
    const value = parseInt(sensitivitySlider.value);
    let sensitivityText = '';
    
    // Map sensitivity values to text
    switch (value) {
        case 1:
            sensitivityText = 'Very Low';
            break;
        case 2:
            sensitivityText = 'Low';
            break;
        case 3:
            sensitivityText = 'Medium';
            break;
        case 4:
            sensitivityText = 'High';
            break;
        case 5:
            sensitivityText = 'Very High';
            break;
    }
    
    sensitivityValue.textContent = sensitivityText;
    
    // Save preference
    localStorage.setItem('sensitivity', value);
}

function updateThreshold() {
    const value = parseFloat(thresholdSlider.value);
    thresholdValue.textContent = value.toFixed(3);
    
    // Save preference
    localStorage.setItem('threshold', value);
}

function toggleNightMode() {
    // Save preference
    localStorage.setItem('nightMode', nightModeToggle.checked);
}

function toggleSound() {
    // Save preference
    localStorage.setItem('soundEnabled', soundToggle.checked);
}

// Statistics Functions
function updateMonitorTime() {
    if (!stats.monitorStartTime) return;
    
    const elapsed = Math.floor((Date.now() - stats.monitorStartTime) / 1000);
    const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
    const seconds = (elapsed % 60).toString().padStart(2, '0');
    
    monitorTime.textContent = `${minutes}:${seconds}`;
}

function resetStatistics() {
    stats.alertCount = 0;
    stats.earValues = [];
    stats.faceConfidenceValues = [];
    
    alertCount.textContent = '0';
    avgEAR.textContent = '0.00';
    faceConfidence.textContent = '0%';
    
    if (stats.monitorStartTime) {
        stats.monitorStartTime = Date.now(); // Reset monitor start time if monitoring
    }
    
    updateStatusMessage('Statistics reset', 'success');
}

// Troubleshooting Functions
function toggleTroubleshootingPanel() {
    const isHidden = troubleshootContent.classList.contains('hidden');
    
    if (isHidden) {
        troubleshootContent.classList.remove('hidden');
        toggleTroubleshoot.innerHTML = '<i class="fas fa-chevron-up"></i>';
    } else {
        troubleshootContent.classList.add('hidden');
        toggleTroubleshoot.innerHTML = '<i class="fas fa-chevron-down"></i>';
    }
}

async function testConnection() {
    updateStatusMessage('Testing connection to backend...', 'info');
    
    try {
        const response = await fetch(`${API_URL}/status`);
        const data = await response.json();
        
        if (data.status === 'API is running') {
            updateStatusMessage('Connection successful! Backend server is running.', 'success');
            updateServerStatus(true);
            
            // Enable buttons if they were disabled
            startBtn.disabled = false;
            calibrateBtn.disabled = false;
        } else {
            updateStatusMessage('Connection test failed. Unexpected response from server.', 'error');
            updateServerStatus(false);
        }
    } catch (err) {
        console.error('Error testing connection:', err);
        updateStatusMessage('Connection test failed. Could not reach backend server.', 'error');
        updateServerStatus(false);
    }
}

function resetApplication() {
    // Stop any ongoing detection
    stopDetection();
    
    // Reset statistics
    resetStatistics();
    
    // Reset UI elements to defaults
    updateStatusMessage('Application reset complete. All settings returned to defaults.', 'success');
    
    // Clear stored preferences
    localStorage.clear();
    
    // Reset controls to defaults
    sensitivitySlider.value = 3;
    thresholdSlider.value = 0.25;
    nightModeToggle.checked = false;
    soundToggle.checked = true;
    darkModeToggle.checked = true;
    
    // Apply defaults
    updateSensitivity();
    updateThreshold();
    toggleDarkMode();
    
    // Test connection again
    setTimeout(checkBackendStatus, 500);
}

// Dark Mode Toggle
function toggleDarkMode() {
    const isDarkMode = darkModeToggle.checked;
    const body = document.body;
    
    if (isDarkMode) {
        // Dark mode (default)
        body.style.backgroundColor = '#0f0f0f';
        body.style.color = '#f8f8f8';
    } else {
        // Light mode
        body.style.backgroundColor = '#f8f8f8';
        body.style.color = '#0f0f0f';
        
        // Change other components for light mode
        document.querySelectorAll('.bg-black, .bg-gray-900, .bg-gray-800').forEach(el => {
            el.classList.remove('bg-black', 'bg-gray-900', 'bg-gray-800');
            el.classList.add('bg-white', 'shadow-md');
        });
    }
    
    // Save preference
    localStorage.setItem('darkMode', isDarkMode);
}