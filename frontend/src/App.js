import React, { useState } from 'react';
import './App.css';
import ImageUploader from './components/ImageUploader';
import PredictionResult from './components/PredictionResult';
import XAIExplanations from './components/XAIExplanations';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [explanations, setExplanations] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePrediction = (predData) => {
    setPrediction(predData);
    setExplanations(null);
  };

  const handleExplanations = (explData) => {
    setExplanations(explData);
  };

  const handleLoading = (isLoading) => {
    setLoading(isLoading);
  };

  const handleError = (err) => {
    setError(err);
  };

  const handleReset = () => {
    setPrediction(null);
    setExplanations(null);
    setError(null);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🩺 Pneumonia Detection System</h1>
        <p className="subtitle">Explainable AI for Chest X-ray Analysis</p>
      </header>

      <main className="App-main">
        {error && (
          <div className="error-message">
            <p>❌ Error: {error}</p>
            <button onClick={handleReset}>Try Again</button>
          </div>
        )}

        {!prediction && (
          <ImageUploader
            onPrediction={handlePrediction}
            onLoading={handleLoading}
            onError={handleError}
          />
        )}

        {loading && (
          <div className="loading">
            <div className="spinner"></div>
            <p>Analyzing X-ray image...</p>
          </div>
        )}

        {prediction && (
          <>
            <PredictionResult
              prediction={prediction}
              onReset={handleReset}
            />
            <XAIExplanations
              prediction={prediction}
              explanations={explanations}
              onExplanations={handleExplanations}
              onLoading={handleLoading}
              onError={handleError}
            />
          </>
        )}
      </main>

      <footer className="App-footer">
        <p>Final Year Project - Explainable Deep Learning Framework for Pneumonia Detection</p>
      </footer>
    </div>
  );
}

export default App;
