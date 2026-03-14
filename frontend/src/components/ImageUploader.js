import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './ImageUploader.css';

const ImageUploader = ({ onPrediction, onLoading, onError }) => {
  const [preview, setPreview] = useState(null);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Create preview (await so imagePreview is always set when we get the response)
    const imagePreview = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        setPreview(reader.result);
        resolve(reader.result);
      };
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(file);
    });

    // Upload and predict
    const formData = new FormData();
    formData.append('file', file);

    try {
      onLoading(true);
      onError(null);

      const response = await axios.post('/api/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      onPrediction({
        ...response.data,
        imagePreview,
      });
    } catch (err) {
      onError(err.response?.data?.error || 'Failed to process image');
      setPreview(null);
    } finally {
      onLoading(false);
    }
  }, [onPrediction, onLoading, onError]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif'],
    },
    maxFiles: 1,
  });

  return (
    <div className="uploader-container">
      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? 'active' : ''}`}
      >
        <input {...getInputProps()} />
        {preview ? (
          <div className="preview-container">
            <img src={preview} alt="Preview" className="preview-image" />
            <p className="preview-text">Click or drag to change image</p>
          </div>
        ) : (
          <div className="dropzone-content">
            <div className="upload-icon">📤</div>
            <p className="dropzone-text">
              {isDragActive
                ? 'Drop the X-ray image here'
                : 'Drag & drop a chest X-ray image here, or click to select'}
            </p>
            <p className="dropzone-hint">
              Supported formats: PNG, JPG, JPEG (Max 16MB)
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageUploader;
