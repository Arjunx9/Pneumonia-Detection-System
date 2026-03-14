import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './XAIExplanations.css';

const XAIExplanations = ({ prediction, explanations, onExplanations, onLoading, onError }) => {
  const [activeTab, setActiveTab] = useState('gradcam');
  const [loadingMethod, setLoadingMethod] = useState('');
  const lastImageRef = useRef(null);

  const fetchExplanation = async (method) => {
    if (explanations && explanations[method]) {
      setActiveTab(method);
      return;
    }

    try {
      setLoadingMethod(method);
      onLoading(true);
      onError(null);

      const formData = new FormData();
      const response = await fetch(prediction.imagePreview);
      const blob = await response.blob();
      formData.append('file', blob, 'image.jpg');

      let endpoint = '';
      switch (method) {
        case 'gradcam':
          endpoint = '/api/explain/gradcam';
          break;
        case 'shap':
          endpoint = '/api/explain/shap';
          break;
        case 'lime':
          endpoint = '/api/explain/lime';
          break;
        default:
          endpoint = '/api/explain/all';
      }

      const res = await axios.post(endpoint, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      onExplanations({
        ...explanations,
        [method]: res.data,
      });
      setActiveTab(method);
    } catch (err) {
      onError(err.response?.data?.error || `Failed to generate ${method.toUpperCase()} explanation`);
    } finally {
      setLoadingMethod('');
      onLoading(false);
    }
  };

  // Auto-fetch Grad-CAM when a new prediction arrives
  useEffect(() => {
    if (prediction && prediction.imagePreview !== lastImageRef.current) {
      lastImageRef.current = prediction.imagePreview;
      fetchExplanation('gradcam');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prediction]);

  const renderGradCAM = () => {
    if (!explanations?.gradcam) {
      return (
        <div className="explanation-placeholder">
          <p>Click "Generate Grad-CAM" to see the heatmap visualization</p>
        </div>
      );
    }

    const data = explanations.gradcam;
    return (
      <div className="gradcam-container">
        <div className="explanation-grid">
          <div className="explanation-item">
            <h4>Original X-ray</h4>
            <img src={data.original_image} alt="Original" />
          </div>
          <div className="explanation-item">
            <h4>Grad-CAM Heatmap</h4>
            <img src={data.heatmap} alt="Heatmap" />
          </div>
          <div className="explanation-item">
            <h4>Overlay Visualization</h4>
            <img src={data.overlay} alt="Overlay" />
          </div>
        </div>
        <div className="explanation-info">
          <p>
            <strong>Grad-CAM</strong> highlights the regions of the X-ray that are most important
            for the model's prediction. Warmer colors (red/yellow) indicate higher importance.
          </p>
        </div>
      </div>
    );
  };

  const renderSHAP = () => {
    if (!explanations?.shap) {
      return (
        <div className="explanation-placeholder">
          <p>Click "Generate SHAP" to see feature importance visualization</p>
        </div>
      );
    }

    const data = explanations.shap;
    return (
      <div className="shap-container">
        <div className="explanation-grid">
          <div className="explanation-item">
            <h4>Original X-ray</h4>
            <img src={data.original_image} alt="Original" />
          </div>
          <div className="explanation-item">
            <h4>SHAP Values</h4>
            <img src={data.shap_heatmap} alt="SHAP" />
          </div>
        </div>
        <div className="explanation-info">
          <p>
            <strong>SHAP</strong> (SHapley Additive exPlanations) shows the contribution of each
            pixel to the final prediction. Red areas increase the probability of the predicted class.
          </p>
        </div>
      </div>
    );
  };

  const renderLIME = () => {
    if (!explanations?.lime) {
      return (
        <div className="explanation-placeholder">
          <p>Click "Generate LIME" to see local interpretable explanations</p>
        </div>
      );
    }

    const data = explanations.lime;
    return (
      <div className="lime-container">
        <div className="explanation-item">
          <h4>LIME Explanation</h4>
          <img src={data.lime_explanation} alt="LIME" />
        </div>
        <div className="explanation-info">
          <p>
            <strong>LIME</strong> (Local Interpretable Model-agnostic Explanations) identifies
            which superpixels (regions) of the image are most important for the prediction.
            Green areas support the prediction, red areas oppose it.
          </p>
        </div>
      </div>
    );
  };

  return (
    <div className="xai-container">
      <div className="xai-card">
        <h2>🔍 Explainable AI (XAI) Explanations</h2>
        <p className="xai-subtitle">
          Understand why the model made this prediction using advanced XAI techniques
        </p>

        <div className="xai-tabs">
          <button
            className={`tab-button ${activeTab === 'gradcam' ? 'active' : ''}`}
            onClick={() => fetchExplanation('gradcam')}
            disabled={!!loadingMethod}
          >
            Grad-CAM
          </button>
          <button
            className={`tab-button ${activeTab === 'shap' ? 'active' : ''}`}
            onClick={() => fetchExplanation('shap')}
            disabled={!!loadingMethod}
          >
            SHAP
          </button>
          <button
            className={`tab-button ${activeTab === 'lime' ? 'active' : ''}`}
            onClick={() => fetchExplanation('lime')}
            disabled={!!loadingMethod}
          >
            LIME
          </button>
        </div>

        {loadingMethod && (
          <div className="explanation-loading">
            <div className="spinner"></div>
            <p>Generating {loadingMethod.toUpperCase()} explanation...</p>
            {loadingMethod === 'lime' && <p className="loading-subtext">This may take up to 30 seconds</p>}
          </div>
        )}

        <div className="explanation-content">
          {activeTab === 'gradcam' && renderGradCAM()}
          {activeTab === 'shap' && renderSHAP()}
          {activeTab === 'lime' && renderLIME()}
        </div>
      </div>
    </div>
  );
};

export default XAIExplanations;
