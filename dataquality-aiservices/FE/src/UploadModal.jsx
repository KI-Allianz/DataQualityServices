import React, { useState, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import './UploadModal.css';

function UploadModal({
  show,
  onClose,
  onSubmit,
  setUploadedFile,
  setColumns,
  setTargetVariable,
  setCsvContent,
  setParsedCSV
}) {
  const { t } = useTranslation();
  const [selectedFile, setSelectedFile] = useState(null);
  const [useLLMFeatureInference, setUseLLMFeatureInference] = useState(false);
  const [useLLMPersonalDataDetection, setUseLLMPersonalDataDetection] = useState(false);
  const [localColumns, setLocalColumns] = useState([]);
  const [localTarget, setLocalTarget] = useState('');
  const [localParsedCSV, setLocalParsedCSV] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const isSubmittingRef = useRef(false);

  useEffect(() => {
    if (!show) {
      setSelectedFile(null);
      setLocalColumns([]);
      setLocalTarget('');
      setIsLoading(false);
      isSubmittingRef.current = false;
    }
  }, [show]);

  const handleFileChange = (e) => {
    if (isSubmittingRef.current) return;
    if (e.target.files.length > 0) {
      const file = e.target.files[0];
      processFile(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (isSubmittingRef.current) return;
    if (e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      processFile(file);
    }
  };

  const handleClose = () => {
    if (isSubmittingRef.current) return;
    onClose();
  };

  const processFile = (file) => {
    if (isSubmittingRef.current) return;
    setSelectedFile(file);
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target.result;
      const lines = text.trim().split(/\r?\n/);
      const headers = lines[0].split(',').map((header) => header.trim());
      const data = lines.slice(1).map((line) => {
        const values = line.split(',').map((value) => value.trim());
        return headers.reduce((acc, header, idx) => {
          acc[header] = values[idx] || '';
          return acc;
        }, {});
      });

      setLocalParsedCSV(data);  // lokal speichern, nicht global
      setLocalColumns(headers);
      setLocalTarget(headers[0]);
      setCsvContent(text);
    };
    reader.readAsText(file);
  };

  //Angepasst Test SL
  const handleSubmit = async () => {
    if (isSubmittingRef.current) return;

    if (!selectedFile || !localTarget) {
      alert("Bitte Datei und Zielvariable auswählen.");
      return;
    }

    isSubmittingRef.current = true;
    setIsLoading(true);
    setUploadedFile(selectedFile);
    setColumns(localColumns);
    setTargetVariable(localTarget);
    setParsedCSV(localParsedCSV);

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("target", localTarget);

    // LLM-Variablen korrekt an das Backend senden
    formData.append("use_llm_feature_type_inference", useLLMFeatureInference ? "true" : "false");
    formData.append("use_llm_personal_data_detection", useLLMPersonalDataDetection ? "true" : "false");

    try {
      const response = await fetch('/upload', {
        method: 'POST',
        body: formData
      });

      const responseText = await response.text();
      let result = {};

      try {
        result = responseText ? JSON.parse(responseText) : {};
      } catch {
        result = { error: responseText || "Ungültige Server-Antwort." };
      }

      if (!response.ok) {
        alert("Fehler beim Upload: " + (result?.error || "Unbekannter Fehler"));
        setIsLoading(false);
        return;
      }

      await onSubmit(result); // weitergeben an App.jsx

    } catch (error) {
      alert("Beim Hochladen der Datei ist ein Fehler aufgetreten:\n" + error.message);
    } finally {
      isSubmittingRef.current = false;
      setIsLoading(false);
    }
  };




  if (!show) return null;

  return (
    <div className="modal-overlay" onClick={handleClose}>
      <div
        className="modal upload-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <h2>{t('uploadTitle')}</h2>
          <button onClick={handleClose} className="close-btn" disabled={isLoading}>&times;</button>
        </div>

        <div
          className="dropzone"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          <p>{t('dragDrop')}</p>
          <p className="file-limit">{t('fileLimit')}</p>
          <label htmlFor="file-input" className="browse-btn">
            {t('browseFiles')}
          </label>
          <input
            id="file-input"
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            disabled={isLoading}
            style={{ display: 'none' }}
          />
        </div>

        {selectedFile && (
          <div className="uploaded-section">
            <div className="uploaded-file-info">
              <div>{selectedFile.name}</div>
              <div>{(selectedFile.size / 1024).toFixed(1)}KB</div>
            </div>

            <div className="upload-success">
              🙌 {t('dataUploaded')}
            </div>

            <div>
              <label htmlFor="target-select" className="select-label">
                {t('selectTargetVariable')}
              </label>
              <select
                id="target-select"
                className="select-target"
                value={localTarget}
                onChange={(e) => setLocalTarget(e.target.value)}
                disabled={isLoading}
              >
                {localColumns.map((col, idx) => (
                  <option key={idx} value={col}>{col}</option>
                ))}
              </select>
            </div>

            <div className="llm-settings-box">
              <div className="llm-settings-title">LLM Settings</div>

              <div className="toggle-row">
                <span>Use LLM for Feature Type Inference</span>
                <label className="switch">
                  <input
                    type="checkbox"
                    checked={useLLMFeatureInference}
                    disabled={isLoading}
                    onChange={() => {
                      const newValue = !useLLMFeatureInference;
                      setUseLLMFeatureInference(newValue);
                      console.log("LLM Feature Type Inference:", newValue);
                    }}
                  />
                  <span className="slider round"></span>
                </label>
              </div>

              <div className="toggle-row">
                <span>Use LLM for Personalized Data Detection</span>
                <label className="switch">
                  <input
                    type="checkbox"
                    checked={useLLMPersonalDataDetection}
                    disabled={isLoading}
                    onChange={() => {
                      const newValue = !useLLMPersonalDataDetection;
                      setUseLLMPersonalDataDetection(newValue);
                      console.log("LLM Personalized Data Detection:", newValue);
                    }}
                  />
                  <span className="slider round"></span>
                </label>
              </div>
            </div>


            <button className="submit-btn" onClick={handleSubmit} disabled={isLoading}>
              {isLoading ? (
                <>
                  <span className="spinner"></span> {t('submit')} {t('loading') || 'wird verarbeitet...'}
                </>
              ) : (
                t('submit')
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default UploadModal;
