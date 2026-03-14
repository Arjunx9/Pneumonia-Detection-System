import React from 'react';
import './PredictionResult.css';

const PredictionResult = ({ prediction, onReset }) => {
  const isPneumonia = prediction.prediction === 'Pneumonia';
  const confidence = (prediction.probability * 100).toFixed(2);

  return (
    <div className="prediction-container">
      <div className="result-card">
        <div className="result-header">
          <h2>Prediction Result</h2>
          <button onClick={onReset} className="reset-button">
            ↻ New Analysis
          </button>
        </div>

        <div className="result-content">
          <div className="image-section">
            <img
              src={prediction.imagePreview}
              alt="X-ray"
              className="result-image"
            />
          </div>

          <div className="prediction-section">
            <div className={`prediction-badge ${isPneumonia ? 'pneumonia' : 'normal'}`}>
              <div className="badge-icon">
                {isPneumonia ? '⚠️' : '✅'}
              </div>
              <div className="badge-content">
                <h3>{prediction.prediction}</h3>
                <p className="confidence">
                  Confidence: {confidence}%
                </p>
              </div>
            </div>

            <div className="probabilities">
              <h4>Probability Breakdown:</h4>
              <div className="prob-bar-container">
                <div className="prob-item">
                  <span>Normal</span>
                  <div className="prob-bar">
                    <div
                      className="prob-fill normal"
                      style={{ width: `${prediction.all_probabilities.Normal * 100}%` }}
                    >
                      {(prediction.all_probabilities.Normal * 100).toFixed(2)}%
                    </div>
                  </div>
                </div>
                <div className="prob-item">
                  <span>Pneumonia</span>
                  <div className="prob-bar">
                    <div
                      className="prob-fill pneumonia"
                      style={{ width: `${prediction.all_probabilities.Pneumonia * 100}%` }}
                    >
                      {(prediction.all_probabilities.Pneumonia * 100).toFixed(2)}%
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {isPneumonia && (
              <div className="warning-message">
                <p>⚠️ <strong>Pneumonia Detected</strong></p>
                <p>Please consult with a healthcare professional for proper diagnosis and treatment.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionResult;
