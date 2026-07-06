import React, { useRef, useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import Sidebar from './Sidebar';
import UploadModal from './UploadModal';
import Plot from 'react-plotly.js';
import './App.css';

function App() {
  const { t, i18n } = useTranslation();

  const [jsonResponse, setJsonResponse] = useState(null);
  const inFlightUrlRef = useRef("");
  const hasUploadedRef = useRef(false);
  const [operationTouched, setOperationTouched] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [selectedOperation, setSelectedOperation] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [columns, setColumns] = useState([]);
  const [targetVariable, setTargetVariable] = useState('');
  const [activeTabs, setActiveTabs] = useState({});
  const [csvContent, setCsvContent] = useState('');
  const [parsedCSV, setParsedCSV] = useState([]);
  const [rowsToShow, setRowsToShow] = useState(10);
  const [rowsToShowImputation, setRowsToShowImputation] = useState(10);
  const [rowsToShowAnomaly, setRowsToShowAnomaly] = useState(10);

  // Download Tab
  const downloadItems = [
    { key: "anomalies", label: "anomalies.json" },
    { key: "feature_types", label: "feature_types.json" },
    { key: "imputation", label: "imputation.json" },
    { key: "personal_data", label: "personal_data.json" },
    { key: "datagraphs", label: "datagraphs.json" },
  ];

  const [originalTable, setOriginalTable] = useState(null);
  const [processedTable, setProcessedTable] = useState(null);

  const [downloadSelections, setDownloadSelections] = useState(
    downloadItems.reduce((acc, item) => {
      acc[item.key] = true;
      return acc;
    }, {})
  );

  const [zipFilename, setZipFilename] = useState("DCAT-AP_bundle_services.zip");
  const allSelected = downloadItems.every((it) => !!downloadSelections[it.key]);

  const handleSelectAllDownloads = () => {
    const nextValue = !allSelected;
    const next = downloadItems.reduce((acc, it) => {
      acc[it.key] = nextValue;
      return acc;
    }, {});
    setDownloadSelections(next);
  };

  // Download ZIP via Flask 
  const handleDownloadZip = async () => {
    try {
      const selectedFiles = downloadItems
        .filter((it) => !!downloadSelections[it.key])
        .map((it) => it.label);

      if (selectedFiles.length === 0) {
        alert("Bitte mindestens eine Datei auswählen.");
        return;
      }

      const cleanZipName = (zipFilename || "").trim() || "DCAT-AP_bundle_services.zip";

      if (selectedOperation === "model") {
        setJsonResponse(null);
        setModelPage("select");
        return;
      }


      const res = await fetch("/downloadZip", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          selected: selectedFiles,
          zip_name: cleanZipName,
        }),
      });

      const contentType = res.headers.get("content-type") || "";
      if (!res.ok) {
        if (contentType.includes("application/json")) {
          const err = await res.json();
          throw new Error(err?.error || "Download fehlgeschlagen.");
        }
        throw new Error(`Download fehlgeschlagen (HTTP ${res.status}).`);
      }

      const blob = await res.blob();

      const disposition = res.headers.get("content-disposition") || "";
      const match = disposition.match(/filename\*?=(?:UTF-8''|")?([^";]+)"?/i);
      const serverFilename = match ? decodeURIComponent(match[1]) : cleanZipName;
      const finalName = serverFilename.toLowerCase().endsWith(".zip")
        ? serverFilename
        : `${serverFilename}.zip`;

      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = finalName;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (e) {
      console.error(e);
      alert(e.message || "Download fehlgeschlagen.");
    }
  };


  // Model Tab
  const modelOptions = [
    { key: "rf", label: "Random Forest" },
    { key: "automl", label: "AutoML" },
    { key: "xgb", label: "xGBoost" },
  ];


  const [selectedModelKey, setSelectedModelKey] = useState("");
  const [modelPage, setModelPage] = useState("select");
  const [modelResult, setModelResult] = useState(null);
  const [modelLoading, setModelLoading] = useState(false);
  const [modelError, setModelError] = useState("");
  const [modelDownloadUrl, setModelDownloadUrl] = useState(null);
  const [showModelResultPopup, setShowModelResultPopup] = useState(false);


  const searchOptionsClassic = [
    { key: "none", label: "None (Baseline Model)" },
    { key: "random", label: "Random Search" },
    { key: "grid", label: "Grid Search" },
  ];

  const [selectedSearchKey, setSelectedSearchKey] = useState("none");

  // AutoML (AutoGluon)
  const automlPresetOptions = [
    { key: "good_quality", label: "good_quality" },
    { key: "high_quality", label: "high_quality" },
    { key: "best_quality", label: "best_quality" },
  ];

  const [selectedAutoMLPreset, setSelectedAutoMLPreset] = useState("good_quality");
  const [selectedAutoMLHpoKey, setSelectedAutoMLHpoKey] = useState("off");

  // Cross Val 
  const crossValOptions = [
    { key: "no", label: "No" },
    { key: "yes", label: "Yes" },
  ];

  const [selectedCrossValKey, setSelectedCrossValKey] = useState("no");

  const handleRunModel = async () => {
    try {
      setModelLoading(true);
      setModelError("");
      setModelResult(null);

      const target = targetVariable || jsonResponse?.target_variable || "";
      const time_or_tab = detectedCounts?.time_or_tab || "";


      const selectedModels = selectedModelKey
        ? [modelOptions.find((m) => m.key === selectedModelKey)?.label].filter(Boolean)
        : [];


      if (selectedModels.length === 0) {
        setModelError("Please select a model to continue.");
        setModelLoading(false);
        return;
      }

      const res = await fetch("/model", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_variable: target,
          time_or_tab: time_or_tab,
          selected_models: selectedModels,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.error || `Request failed (HTTP ${res.status})`);
      }

      setModelResult(data);
    } catch (e) {
      console.error(e);
      setModelError(e.message || "Model request failed.");
    } finally {
      setModelLoading(false);
    }
  };

  const [trainingTimeSec, setTrainingTimeSec] = useState(60);
  const [isTraining, setIsTraining] = useState(false);
  const [remainingSec, setRemainingSec] = useState(0);
  const [trainingResult, setTrainingResult] = useState(null);
  const [trainingError, setTrainingError] = useState("");

  const startTraining = async () => {
    if (!selectedModelKey) return;

    const t = Math.max(1, Math.min(3600, Number(trainingTimeSec) || 60));
    setTrainingTimeSec(t);

    setTrainingError("");
    setTrainingResult(null);
    setModelDownloadUrl(null);
    setShowModelResultPopup(false);
    setIsTraining(true);
    setRemainingSec(t);

    const intervalId = window.setInterval(() => {
      setRemainingSec((prev) => {
        if (prev <= 1) {
          window.clearInterval(intervalId);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);


    const simulatedFinishMs = Math.floor(t * 1000 * (0.3 + Math.random() * 0.6));

    try {
      const payload = {
        model_key: selectedModelKey,
        search_mode: selectedModelKey === "automl" ? selectedAutoMLHpoKey : selectedSearchKey,
        cross_validation: selectedCrossValKey === "yes",
        time_limit_s: t,
      };

      // AutoML-only params
      if (selectedModelKey === "automl") {
        payload.automl_preset = selectedAutoMLPreset;
        payload.num_trials = 20;
      }

      const res = await fetch("/model", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.error || `Request failed (HTTP ${res.status})`);
      }

      setTrainingResult(data);
      setModelDownloadUrl(data?.artifact?.download_url || null);
      setShowModelResultPopup(true);
    } catch (e) {
      setTrainingError(e?.message || "Training failed.");
    } finally {
      window.clearInterval(intervalId);
      setIsTraining(false);
      setRemainingSec(0);
    }
  };


  // Dynamische Höhen für Navbar & Footer erfassen
  const [layoutSizes, setLayoutSizes] = useState({
    navbar: 0,
    footer: 0,
  });

  useEffect(() => {
    function updateSizes() {
      const navbar = document.querySelector(".navbar")?.offsetHeight ?? 0;
      const footer = document.querySelector(".footer")?.offsetHeight ?? 0;

      setLayoutSizes({ navbar, footer });
    }

    updateSizes();
    window.addEventListener("resize", updateSizes);

    return () => window.removeEventListener("resize", updateSizes);
  }, []);


  const operations = [
    'original', 'inference', 'datagraphs', 'imputation', 'anomaly', 'personal',
    'summary', 'cleaned', /* 'model', */ 'downloads'
  ];

  const tabOptions = {
    original: ['rawData', 'seeStatistics'],
    inference: ['seeFeatureResults', 'seeExplanation'],
    datagraphs: ['boxplots', 'histograms', 'correlationMatrix', 'featureImportance'],
    imputation: ['imputationResults', 'seeExplanationImp'],
    anomaly: ['anomalyResults', 'seeExplanationAno'],
    personal: [],
    summary: [],
    cleaned: [],
    model: [],
    downloads: []
  };

  const operationItems = operations.map((op) => ({
    key: op,
    subItems: tabOptions[op] || [],
  }));

  const [detectedCounts, setDetectedCounts] = useState(null);

  //Cache im Frontend löschen
  useEffect(() => {
    const wait = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

    const reset = async () => {
      try {
        let lastError = null;

        for (let attempt = 1; attempt <= 10; attempt += 1) {
          try {
            const response = await fetch("/reset", {
              method: "POST",
              credentials: "include",
            });

            if (response.ok) {
              lastError = null;
              break;
            }

            lastError = new Error(`/reset failed with HTTP ${response.status}`);
          } catch (error) {
            lastError = error;
          }

          await wait(1000);
        }

        if (lastError) {
          throw lastError;
        }
      } catch (e) {
        console.error("Session reset failed:", e);
      } finally {
        if (hasUploadedRef.current) {
          return;
        }

        setJsonResponse(null);
        setSelectedOperation(null);
        setActiveTabs({});
        setUploadedFile(null);
        setColumns([]);
        setTargetVariable("");
        setCsvContent("");
        setParsedCSV([]);
        setUploadSuccess(false);
        setDetectedCounts({});
      }
    };
    reset();
  }, []);

  useEffect(() => {
    // nur laden, wenn Downloads gewählt UND ein Upload stattgefunden hat
    if (selectedOperation !== "downloads") return;
    if (!uploadSuccess) {
      setOriginalTable(null);
      setProcessedTable(null);
      return;
    }

    (async () => {
      try {
        const res = await fetch("/cleaned", {
          method: "GET",
          credentials: "include",
        });

        const data = await res.json();

        setOriginalTable(data?.original ?? null);
        setProcessedTable(data?.final ?? data?.processed ?? null);
      } catch (err) {
        console.error("Fehler beim Laden der Tabellen-Preview (/cleaned):", err);
        setOriginalTable(null);
        setProcessedTable(null);
      }
    })();
  }, [selectedOperation, uploadSuccess]);


  const handleOperationSelect = async (operation) => {
    setSelectedOperation(operation);
    setOperationTouched(true);

    if (!uploadSuccess && operation !== "downloads") {
      setJsonResponse(null);
      return;
    }

    const defaultTab = tabOptions[operation]?.[0];
    if (defaultTab) {
      setActiveTabs((prev) => ({ ...prev, [operation]: defaultTab }));
    }

    if (operation === "ReportRoutes" || operation === "ReportOverall") return;

    try {
      const url = operation === "datagraphs" && defaultTab
        ? `/datagraphs?sub=${encodeURIComponent(defaultTab)}`
        : `/${operation}`;

      if (inFlightUrlRef.current === url) {
        return;
      }
      inFlightUrlRef.current = url;

      const res = await fetch(url, {
        method: "GET",
        credentials: "include",
      });
      const data = await res.json();
      setJsonResponse(data);
    } catch (err) {
      console.error(`Fehler bei API-Aufruf für /${operation}:`, err);
      setJsonResponse({ error: "Fehler beim Laden der Daten." });
    } finally {
      inFlightUrlRef.current = "";
    }
  };


  const handleLanguageToggle = () => {
    i18n.changeLanguage(i18n.language === 'en' ? 'de' : 'en');
  };

  const handleDragOver = (e) => e.preventDefault();

  useEffect(() => {
    if (
      selectedOperation === 'number' && Array.isArray(jsonResponse?.dataframe?.data) && jsonResponse.dataframe.data.length > 0
    ) {
      setRowsToShowAnomaly(10);
    }
  }, [jsonResponse, selectedOperation]);

  useEffect(() => {
    if (selectedOperation === 'imputation' && Array.isArray(jsonResponse?.original?.data) && jsonResponse.original.data.length > 0) {
      setRowsToShowImputation(10);
    }
  }, [jsonResponse, selectedOperation]);

  useEffect(() => {
    if (
      selectedOperation === 'anomaly' && Array.isArray(jsonResponse?.anomalies?.data) && jsonResponse.anomalies.data.length > 0
    ) {
      setRowsToShowAnomaly(10);
    }
  }, [jsonResponse, selectedOperation]);

  useEffect(() => {
    const fetchCounts = async () => {
      try {
        const res = await fetch("/detectedCounts", {
          method: "GET",
          credentials: "include",
        });
        const data = await res.json();
        setDetectedCounts((prev) => ({ ...prev, ...data }));
      } catch (err) {
        console.error("Fehler beim Laden der detectedCounts:", err);
      }
    };

    if (uploadSuccess) fetchCounts();
  }, [uploadSuccess]);


  const availableOptions = [10, 25, 50].filter(n => n <= (jsonResponse?.dataframe?.data?.length || 0));
  if ((jsonResponse?.dataframe?.data?.length || 0) > 0) {
    availableOptions.push('all');
  }

  const translatePlotLabels = (labels, translations) => (
    Array.isArray(labels)
      ? labels.map((label) => translations[label] || label)
      : labels
  );

  const isMissingValue = (value) => (
    value === null || value === undefined || Number.isNaN(value)
  );

  const wasCellImputed = (rowIdx, columnKey) => {
    const originalValue = jsonResponse?.original?.data?.[rowIdx]?.[columnKey];
    const imputedValue = jsonResponse?.imputed?.data?.[rowIdx]?.[columnKey];

    return isMissingValue(originalValue) && !isMissingValue(imputedValue);
  };

  const cleanedPreviewTable = jsonResponse?.final ?? jsonResponse?.processed;

  return (
    <div className="app-container">
      <div className="navbar">
        <img src="/logo.svg" alt="Logo" className="logo" />
        <div className="nav-links">
        
          <button onClick={() => setShowUploadModal(true)} className="upload-btn">
            {t('upload')}
          </button>
          <button onClick={handleLanguageToggle} className="language-toggle-button">
            DE/EN
          </button>
        </div>
      </div>

      <Sidebar
        key={`${layoutSizes.navbar}-${layoutSizes.footer}`}
        items={operationItems}
        selectedItem={selectedOperation}
        layoutSizes={layoutSizes}
        onSelectItem={(key) => handleOperationSelect(key)}
        onSelectSubItem={async (parentKey, subKey) => {
          setSelectedOperation(parentKey);
          setOperationTouched(true);
          setActiveTabs((prev) => ({ ...prev, [parentKey]: subKey }));

          if (!uploadSuccess) {
            setJsonResponse(null);
            return;
          }


          const url =
            parentKey === "datagraphs"
              ? `/datagraphs?sub=${encodeURIComponent(subKey)}`
              : `/${parentKey}`;

          if (inFlightUrlRef.current === url) {
            return;
          }
          inFlightUrlRef.current = url;

          try {
            const res = await fetch(url, { method: "GET", credentials: "include" });
            const data = await res.json();
            setJsonResponse(data);
          } catch (err) {
            console.error(`Fehler bei API-Aufruf für ${url}:`, err);
            setJsonResponse({ error: "Fehler beim Laden der Daten." });
          } finally {
            inFlightUrlRef.current = "";
          }
        }}

        renderLabel={(key, isSubItem = false) => {
          if (isSubItem) return <span>{t(key)}</span>;

          const noBadgeKeys = ['original', 'datagraphs', 'summary', 'cleaned', 'model', 'downloads'];
          if (noBadgeKeys.includes(key)) return <span>{t(key)}</span>;

          const count = detectedCounts?.[key];
          if (typeof count !== "number" || count <= 0) return <span>{t(key)}</span>;

          return (
            <div className="sidebar-label">
              <span>{t(key)}</span>
              <span className="sidebar-detected">
                {count} {t('detected')}
              </span>
            </div>
          );
        }}

      />


      <div className="background" />

      {selectedOperation && (
        <div className="current-tab-label">
          <h3>
            {selectedOperation === 'downloads'
              ? t('Downloads')
              : activeTabs?.[selectedOperation]
                ? t(activeTabs[selectedOperation])
                : t(selectedOperation)}
          </h3>
        </div>
      )}


      <div className="scroll-content">

        <UploadModal
          show={showUploadModal}
          onClose={() => setShowUploadModal(false)}
          onSubmit={async () => {
            hasUploadedRef.current = true;
            setShowUploadModal(false);
            setUploadSuccess(true);
            setSelectedOperation("original");
            setActiveTabs((prev) => ({ ...prev, original: "rawData" }));
            inFlightUrlRef.current = "/original";

            try {
              const originalRes = await fetch("/original", {
                method: "GET",
                credentials: "include",
              });
              if (!originalRes.ok) {
                throw new Error(`/original failed with HTTP ${originalRes.status}`);
              }
              const originalData = await originalRes.json();
              setJsonResponse(originalData);
              inFlightUrlRef.current = "";

              const res = await fetch("/detectedCounts", {
                method: "GET",
                credentials: "include",
              });
              if (!res.ok) {
                throw new Error(`/detectedCounts failed with HTTP ${res.status}`);
              }
              const data = await res.json();
              setDetectedCounts(data && Object.keys(data).length ? data : null);
            } catch (err) {
              console.error("Fehler beim Laden nach dem Upload:", err);
              inFlightUrlRef.current = "";
              setDetectedCounts((prev) => prev ?? {});
            }
          }}
          setUploadedFile={setUploadedFile}
          setColumns={setColumns}
          setTargetVariable={setTargetVariable}
          setCsvContent={setCsvContent}
          setParsedCSV={setParsedCSV}
        />

        {selectedOperation === "downloads" && (
          <div className="summary-plot-section">
            <div className="plots-grid">
              <div className="downloads-container">
                <h3>Finished Data Processing</h3>

                {!uploadSuccess ? (
                  <div className="explanation-text">
                    Bitte zuerst einen Datensatz hochladen.
                  </div>
                ) : (
                  <>

                    {(originalTable?.data?.length > 0 || processedTable?.data?.length > 0) && (
                      <div className="downloads-preview-grid">
                        <div className="downloads-preview-panel">
                          <div className="downloads-preview-title">Original Table</div>
                          {originalTable?.data?.length > 0 && (
                            <div className="downloads-table-scroll">
                              <table className="csv-table">
                                <thead>
                                  <tr>
                                    {originalTable.columns.map((c, i) => (
                                      <th key={i}>{c}</th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody>
                                  {originalTable.data.slice(0, 10).map((row, r) => (
                                    <tr key={r}>
                                      {originalTable.columns.map((c, i) => (
                                        <td key={i}>{row[c]?.toString() ?? "-"}</td>
                                      ))}
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          )}
                        </div>

                        <div className="downloads-preview-panel">
                          <div className="downloads-preview-title">Processed Table</div>
                          {processedTable?.data?.length > 0 && (
                            <div className="downloads-table-scroll">
                              <table className="csv-table">
                                <thead>
                                  <tr>
                                    {processedTable.columns.map((c, i) => (
                                      <th key={i}>{c}</th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody>
                                  {processedTable.data.slice(0, 10).map((row, r) => (
                                    <tr key={r}>
                                      {processedTable.columns.map((c, i) => (
                                        <td key={i}>{row[c]?.toString() ?? "-"}</td>
                                      ))}
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    <div className="downloads-section">
                      <div className="downloads-section-title">Select services for upload to the KI-Allianz Data Platform.</div>
                      <div className="downloads-items">
                        {downloadItems.map((item) => (
                          <label key={item.key} className="downloads-item">
                            <input
                              type="checkbox"
                              checked={!!downloadSelections[item.key]}
                              onChange={(e) =>
                                setDownloadSelections((prev) => ({
                                  ...prev,
                                  [item.key]: e.target.checked,
                                }))
                              }
                            />
                            <span>{item.label}</span>
                          </label>
                        ))}
                      </div>
                    </div>

                    <div className="downloads-section">
                      <div className="downloads-section-title">ZIP filename</div>
                      <div className="downloads-zip-input">
                        <input
                          className="downloads-zip-name"
                          value={(zipFilename || "").replace(/\.zip$/i, "")}
                          onChange={(e) => {
                            const base = e.target.value.replace(/\.zip$/i, "");
                            setZipFilename(`${base}.zip`);
                          }}
                        />
                        <span className="downloads-zip-suffix">.zip</span>
                      </div>
                    </div>

                    <div className="downloads-actions">
                      <button
                        type="button"
                        className="report-download-button"
                        onClick={handleDownloadZip}
                      >
                        Download
                      </button>

                      <button
                        type="button"
                        className="report-download-button"
                        onClick={handleSelectAllDownloads}
                      >
                        {allSelected ? "Deselect all" : "Select all"}
                      </button>
                    </div>

                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {/* MODEL SUB PAGE - COMMENTED OUT */}
        {/* selectedOperation === "model" && (
          <div className="summary-plot-section">
            <div className="plots-grid">
              <div className="model-container">
                <h3>Select your Model</h3>

                {!uploadSuccess ? (
                  <div className="explanation-text">
                    Bitte zuerst einen Datensatz hochladen.
                  </div>
                ) : (
                  <>
                    {modelPage === "select" ? (
                      <>
                        <div className="model-section">
                          <div className="model-section-title">Choose one</div>

                          <div className="model-items">
                            {modelOptions.map((m) => (
                              <label key={m.key} className="model-item">
                                <input
                                  type="checkbox"
                                  checked={selectedModelKey === m.key}
                                  onChange={(e) => {
                                    setSelectedModelKey(e.target.checked ? m.key : "");
                                  }}
                                />
                                <span>{m.label}</span>
                              </label>
                            ))}
                          </div>
                        </div>

                        <div className="model-actions">
                          <button
                            type="button"
                            className="report-download-button"
                            onClick={() => setModelPage("next")}
                            disabled={!selectedModelKey}
                          >
                            Weiter
                          </button>
                        </div>
                      </>
                    ) : (
                      <>
                        {selectedModelKey !== "automl" ? (
                          <>
                            <div className="model-section-title">Search Strategy</div>
                            <div className="model-items">
                              {searchOptionsClassic.map((o) => (
                                <label key={o.key} className="model-item">
                                  <input
                                    type="checkbox"
                                    checked={selectedSearchKey === o.key}
                                    onChange={(e) => setSelectedSearchKey(e.target.checked ? o.key : "none")}
                                  />
                                  <span>{o.label}</span>
                                </label>
                              ))}
                            </div>
                          </>
                        ) : (
                          <>
                            <div className="model-section-title">AutoML (AutoGluon) Preset</div>
                            <div className="model-items">
                              {automlPresetOptions.map((o) => (
                                <label key={o.key} className="model-item">
                                  <input
                                    type="checkbox"
                                    checked={selectedAutoMLPreset === o.key}
                                    onChange={(e) => setSelectedAutoMLPreset(e.target.checked ? o.key : "best_quality")}
                                  />
                                  <span>{o.label}</span>
                                </label>
                              ))}
                            </div>

                            <div className="model-section-title" style={{ marginTop: "1rem" }}>
                              Hyperparameter Tuning
                            </div>
                            <div className="model-items">
                              {[
                                { key: "off", label: "Off (recommended default)" },
                                { key: "random", label: "Random Search" },
                                { key: "grid", label: "Grid Search" },
                              ].map((o) => (
                                <label key={o.key} className="model-item">
                                  <input
                                    type="checkbox"
                                    checked={selectedAutoMLHpoKey === o.key}
                                    onChange={(e) => setSelectedAutoMLHpoKey(e.target.checked ? o.key : "off")}
                                  />
                                  <span>{o.label}</span>
                                </label>
                              ))}
                            </div>
                          </>
                        )}

                        {(selectedModelKey === "rf" || selectedModelKey === "xgb") && (
                          <div className="model-section" style={{ marginTop: "1rem" }}>
                            <div className="model-section-title">Cross Validation</div>

                            <div className="model-items">
                              {crossValOptions.map((o) => (
                                <label key={o.key} className="model-item">
                                  <input
                                    type="checkbox"
                                    checked={selectedCrossValKey === o.key}
                                    onChange={(e) => {
                                      setSelectedCrossValKey(e.target.checked ? o.key : "no");
                                    }}
                                  />
                                  <span>{o.label}</span>
                                </label>
                              ))}
                            </div>
                          </div>
                        )}

                        <div className="model-section" style={{ marginTop: "1rem" }}>
                          <div className="model-section-title">Training time limit (seconds)</div>

                          <div className="model-time-input">
                            <input
                              type="number"
                              min={1}
                              max={3600}
                              value={trainingTimeSec}
                              onChange={(e) => setTrainingTimeSec(e.target.value)}
                              className="model-time-field"
                              disabled={isTraining}
                            />
                            <span className="model-time-hint">1–3600</span>
                          </div>
                        </div>

                        <div className="model-actions">
                          <button
                            type="button"
                            className="report-download-button"
                            onClick={() => setModelPage("select")}
                          >
                            Back
                          </button>

                          <button
                            type="button"
                            className="report-download-button"
                            disabled={!trainingResult}
                            onClick={() => setShowModelResultPopup(true)}
                          >
                            See Results
                          </button>

                          <button
                            type="button"
                            className="report-download-button"
                            onClick={startTraining}
                            disabled={isTraining || !selectedModelKey}
                          >
                            {isTraining ? "Training..." : "Start Training"}
                          </button>
                        </div>

                        {trainingError && (
                          <div className="model-error" style={{ marginTop: "1rem" }}>
                            {trainingError}
                          </div>
                        )}

                        {isTraining && (
                          <div className="model-loading-overlay">
                            <div className="model-loading-card">
                              <div className="model-loading-title">Training in progress</div>

                              <div className="model-loading-sub">
                                Time remaining: <b>{remainingSec}s</b>
                              </div>

                              <div className="model-loading-bar">
                                <div
                                  className="model-loading-bar-fill"
                                  style={{
                                    width: `${Math.max(
                                      0,
                                      Math.min(100, (remainingSec / Math.max(1, Number(trainingTimeSec) || 1)) * 100)
                                    )}%`,
                                  }}
                                />
                              </div>

                              <div className="model-loading-note">
                                The model is being trained. Please wait until the timer reaches zero or the training completes.
                              </div>
                            </div>
                          </div>
                        )}

                        {showModelResultPopup && trainingResult && (
                          <div className="model-popup-overlay">
                            <div className="model-popup">
                              <div className="model-popup-title">Model Results</div>

                              <pre className="model-popup-pre">
                                {JSON.stringify(trainingResult, null, 2)}
                              </pre>

                              <div className="model-popup-actions">
                                <button
                                  type="button"
                                  className="report-download-button"
                                  onClick={() => setShowModelResultPopup(false)}
                                >
                                  Back to Training
                                </button>

                                <button
                                  type="button"
                                  className="report-download-button"
	                                  disabled={!modelDownloadUrl}
	                                  onClick={() => {
	                                    window.location.href = modelDownloadUrl;
	                                  }}
                                >
                                  Download
                                </button>
                              </div>
                            </div>
                          </div>
                        )}
                      </>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        ) */}


        {selectedOperation === 'datagraphs' && activeTabs.datagraphs === 'boxplots' && jsonResponse?.boxplot && (
          <div className="summary-plot-section">
            <div className="plot-wrapper">
              <div className="plot-container">
                <Plot
                  data={jsonResponse.boxplot.data.map((trace) => {
                    const parseStringToArray = (str) => {
                      try {
                        return str
                          .replace(/[\[\]']+/g, '')
                          .split(/[\s,]+/)
                          .map((item) => isNaN(item) ? item : Number(item));
                      } catch {
                        return [];
                      }
                    };

                    return {
                      ...trace,
                      x: typeof trace.x === 'string' ? parseStringToArray(trace.x) : trace.x,
                      y: typeof trace.y === 'string' ? parseStringToArray(trace.y) : trace.y,
                      marker: {
                        color: '#1f77b4',
                        line: {
                          color: '#08519c',
                          width: 1.5
                        }
                      },
                      line: {
                        color: '#bf91e5ff',
                      },
                      boxpoints: 'outliers'
                    };
                  })}
                  layout={{
                    ...jsonResponse.boxplot.layout,
                    title: {
                      text: 'Boxplot of Numerical Features',
                      font: {
                        size: 20,
                        color: '#333'
                      },
                      x: 0.5,
                      xanchor: 'center'
                    },
                    boxmode: 'group',
                    plot_bgcolor: '#ffffff',
                    paper_bgcolor: '#ffffff',
                    font: { color: '#333' }
                  }}
                  style={{ width: '100%', height: '500px', border: '2px solid #ccc9c9ff' }}
                  config={{ responsive: true }}
                />
              </div>
            </div>
          </div>
        )}

        {selectedOperation === "datagraphs" &&
          activeTabs.datagraphs === "histograms" &&
          Array.isArray(jsonResponse?.histograms) &&
          jsonResponse.histograms.length > 0 && (
            <div className="summary-plot-section">
              <div className="plot-wrapper">
                {jsonResponse.histograms.map((histJson, index) => (
                  <div className="plot-container" key={`hist-${index}`}>
                    <Plot
                      data={(histJson?.data || []).map((trace) => {
                        const parseArray = (v) => {
                          if (!v || typeof v !== "string") return v;
                          try {
                            return v
                              .replace(/[\[\]'\n]+/g, "")
                              .split(/[\s,]+/)
                              .filter(Boolean)
                              .map((item) => (isNaN(item) ? item : Number(item)));
                          } catch {
                            return v;
                          }
                        };

                        return {
                          ...trace,
                          x: parseArray(trace.x),
                          y: parseArray(trace.y),


                          marker: {
                            ...(trace.marker || {}),
                            color: "#DFC8F2",
                            line: {
                              color: "rgba(120, 60, 170, 0.95)",
                              width: 1.2,
                            },
                          },
                        };
                      })}
                      layout={{
                        ...(histJson?.layout || {}),
                        margin: { l: 80, r: 40, t: 80, b: 80 },
                      }}
                      style={{ width: "100%", height: "450px", border: "2px solid #ccc9c9ff" }}
                      config={{ responsive: true }}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}




        {selectedOperation === "datagraphs" &&
          activeTabs.datagraphs === "correlationMatrix" &&
          jsonResponse?.correlation_matrix?.data &&
          Array.isArray(jsonResponse.correlation_matrix.data) &&
          jsonResponse.correlation_matrix.data.length > 0 && (
            <div className="summary-plot-section">
              <div className="plot-wrapper">
                <div className="plot-container">
                  <Plot
                    data={(jsonResponse.correlation_matrix.data || []).map((trace) => {
                      if (trace?.type === "heatmap" && trace?.z) {
                        const absZ = (trace.z || []).map((row) =>
                          Array.isArray(row)
                            ? row.map((v) => (typeof v === "number" ? Math.abs(v) : v))
                            : row
                        );

                        return {
                          ...trace,
                          z: absZ,
                          zmin: -1,
                          zmax: 1,
                          colorscale: [
                            [0.00, "#0b1b3a"],
                            [0.15, "#123a78"],
                            [0.30, "#1f77b4"],
                            [0.50, "#ffffff"],
                            [0.70, "#d8c3ff"],
                            [0.85, "#8b3fd1"],
                            [1.00, "#4b137a"],
                          ],
                          colorbar: {
                            ...(trace.colorbar || {}),
                            title: "|corr|",
                          },
                        };
                      }
                      return trace;
                    })}
                    layout={{
                      ...jsonResponse.correlation_matrix.layout,
                      coloraxis: {
                        ...(jsonResponse.correlation_matrix.layout?.coloraxis || {}),
                        colorscale: [
                          [0.00, "#0b1b3a"],
                          [0.15, "#123a78"],
                          [0.30, "#1f77b4"],
                          [0.50, "#ffffff"],
                          [0.70, "#d8c3ff"],
                          [0.85, "#8b3fd1"],
                          [1.00, "#4b137a"],
                        ],
                        cmin: -1,
                        cmax: 1,
                        colorbar: {
                          ...(jsonResponse.correlation_matrix.layout?.coloraxis?.colorbar || {}),
                          title: "|corr|",
                        },
                      },
                      margin:
                        jsonResponse.correlation_matrix.layout?.margin ?? {
                          l: 110,
                          r: 30,
                          t: 60,
                          b: 60,
                        },
                    }}
                    style={{ width: "100%", height: "700px", border: "2px solid #ccc9c9ff" }}
                    config={{ responsive: true }}
                  />
                </div>
              </div>
            </div>
          )}


        {selectedOperation === 'datagraphs'
          && activeTabs.datagraphs === 'featureImportance'
          && jsonResponse?.feature_importance?.data
          && Array.isArray(jsonResponse.feature_importance.data) && (
            <div className="summary-plot-section">
              <div className="plot-wrapper">
                <div className="plot-container">
                  <Plot
                    data={(jsonResponse.feature_importance.data || []).map((trace) => {
                      if (trace?.type === "bar") {
                        return {
                          ...trace,
                          marker: {
                            ...(trace.marker || {}),
                            color: "#bf91e5",
                            line: {
                              ...(trace.marker?.line || {}),
                              color: "#8b3fd1",
                              width: 1,
                            },
                          },
                        };
                      }
                      return trace;
                    })}
                    layout={{
                      ...jsonResponse.feature_importance.layout,
                      margin: { l: 180, r: 40, t: 80, b: 60 },
                    }}
                    style={{ width: '100%', height: '650px', border: '2px solid #ccc9c9ff' }}
                    config={{ responsive: true }}
                  />
                </div>
              </div>
            </div>
          )}


        {selectedOperation === 'original' && activeTabs.original === 'rawData' && jsonResponse?.dataframe?.data?.length > 0 && (
          <div className="summary-plot-section">
            <div className="table-wrapper">
              <div style={{ marginBottom: '1rem' }}>
                <label htmlFor="row-select" style={{ marginRight: '0.5rem' }}>
                  {t('toShow')}
                </label>
                <select
                  id="row-select"
                  value={rowsToShow}
                  onChange={(e) =>
                    setRowsToShow(
                      e.target.value === 'all' ? 'all' : parseInt(e.target.value)
                    )
                  }
                >
                  {[10, 25, 50].map((count) =>
                    jsonResponse.dataframe.data.length >= count ? (
                      <option key={count} value={count}>{count}</option>
                    ) : null
                  )}
                  <option value="all">{t('all')}</option>
                </select>
              </div>

              <div className="table-container">
                <table className="csv-table">
                  <thead>
                    <tr>
                      {jsonResponse.dataframe.columns.map((col, idx) => (
                        <th key={idx}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(rowsToShow === 'all'
                      ? jsonResponse.dataframe.data
                      : jsonResponse.dataframe.data.slice(0, rowsToShow)
                    ).map((row, rowIdx) => (
                      <tr key={rowIdx}>
                        {jsonResponse.dataframe.columns.map((col, colIdx) => (
                          <td key={colIdx}>{row[col]}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {selectedOperation === 'original' && activeTabs.original === 'rawData' && !(jsonResponse?.dataframe?.data?.length > 0) && (
          <div className="summary-plot-section">
            <div className="table-wrapper">
              <p>Keine Rohdaten geladen. Bitte den Datensatz erneut auswählen oder den Tab erneut öffnen.</p>
            </div>
          </div>
        )}

        {selectedOperation === 'original' && activeTabs.original === 'seeStatistics' && jsonResponse && (
          <div className="summary-plot-section">
            {jsonResponse.description?.data?.length > 0 && (
              <div className="table-wrapper">

                <div className="table-container">
                  <table className="csv-table">
                    <thead>
                      <tr>
                        {jsonResponse.description.columns.map((col, idx) => (
                          <th key={idx}>{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {jsonResponse.description.data.map((row, rowIdx) => (
                        <tr key={rowIdx}>
                          {jsonResponse.description.columns.map((col, colIdx) => (
                            <td key={colIdx}>{row[col]}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {selectedOperation === 'original' && activeTabs.original === 'seeGraphs' && jsonResponse && (
          <div className="summary-plot-section">
            {/* Boxplot (Plotly) */}
            {jsonResponse.boxplot && (
              <div className="plot-wrapper">
                <div className="plot-container">
                  <Plot
                    data={jsonResponse.boxplot.data.map((trace) => {
                      const parseStringToArray = (str) => {
                        try {
                          return str
                            .replace(/[\[\]']+/g, '')
                            .split(/[\s,]+/)
                            .map((item) => isNaN(item) ? item : Number(item));
                        } catch {
                          return [];
                        }
                      };

                      return {
                        ...trace,
                        x: typeof trace.x === 'string' ? parseStringToArray(trace.x) : trace.x,
                        y: typeof trace.y === 'string' ? parseStringToArray(trace.y) : trace.y,
                        marker: {
                          color: '#1f77b4',
                          line: {
                            color: '#08519c',
                            width: 1.5
                          }
                        },
                        line: {
                          color: '#bf91e5ff',
                        },
                        boxpoints: 'outliers'
                      };
                    })}
                    layout={{
                      ...jsonResponse.boxplot.layout,
                      title: {
                        text: 'Boxplot of Numerical Features',
                        font: {
                          size: 20,
                          color: '#333'
                        },
                        x: 0.5,
                        xanchor: 'center'
                      },
                      boxmode: 'group',
                      plot_bgcolor: '#ffffff',
                      paper_bgcolor: '#ffffff',
                      font: { color: '#333' }
                    }}
                    style={{ width: '100%', height: '500px', border: '2px solid #ccc9c9ff' }}
                    config={{ responsive: true }}
                  />
                </div>
              </div>
            )}

          </div>
        )}

        {selectedOperation === 'inference' && activeTabs.inference === 'seeFeatureResults' && jsonResponse?.dataframe?.data?.length > 0 && (
          <div className="summary-plot-section">
            <div className="table-wrapper">
              <div className="table-container">
                <table className="csv-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      {jsonResponse.dataframe.columns.map((key, idx) => (
                        <th
                          key={idx}
                          className={key === 'prediction' ? 'highlight-prediction-header' : ''}
                        >
                          {key}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {jsonResponse.dataframe.data.map((row, rowIdx) => (
                      <tr key={rowIdx}>
                        <td>{rowIdx + 1}</td>
                        {jsonResponse.dataframe.columns.map((key, idx) => (
                          <td
                            key={idx}
                            className={key === 'prediction' ? 'highlight-prediction' : ''}
                          >
                            {row[key]?.toString() ?? '-'}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {selectedOperation === 'inference' && activeTabs.inference === 'seeExplanation' && jsonResponse?.explanation_plot?.data && Array.isArray(jsonResponse.explanation_plot.data) && (
          <div className="summary-plot-section">
            <div className="plot-wrapper">
              {jsonResponse.explanation_plot.data.map((trace, index) => {
                const parseArray = (str) => {
                  if (typeof str !== 'string') return str;
                  try {
                    return str
                      .replace(/[\[\]'\n]+/g, '')
                      .split(/[\s,]+/)
                      .filter(Boolean)
                      .map((item) => isNaN(item) ? item : Number(item));
                  } catch {
                    return [];
                  }
                };

                const parsedTrace = {
                  ...trace,
                  x: parseArray(trace.x),
                  y: parseArray(trace.y).map(label => '\u00A0\u00A0' + label),
                  type: 'bar',
                  marker: {
                    color: '#dfc7ebff',
                    line: {
                      color: '#bf91e5ff',
                      width: 1.5
                    }
                  }
                };

                const layout = {
                  ...jsonResponse.explanation_plot.layout,
                  margin: {
                    ...jsonResponse.explanation_plot.layout?.margin,
                    l: 150
                  },
                  yaxis: {
                    ...jsonResponse.explanation_plot.layout?.yaxis,
                    autorange: 'reversed',
                  }
                };

                return (
                  <div className="plot-container" key={`explanation-${index}`}>
                    <Plot
                      data={[parsedTrace]}
                      layout={layout}
                      style={{ width: '100%', height: '500px', border: '2px solid #ccc9c9ff' }}
                      config={{ responsive: true }}
                    />
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {selectedOperation === 'imputation' && activeTabs.imputation === 'imputationResults' && Array.isArray(jsonResponse?.imputed?.data) && jsonResponse.imputed.data.length > 0 && (
          <div className="summary-plot-section">
            <div className="table-wrapper">
              <div style={{ marginBottom: '1rem' }}>
                <label htmlFor="row-select" style={{ marginRight: '0.5rem' }}>
                  {t('toShow')}
                </label>
                <select
                  id="row-select"
                  value={rowsToShowImputation}
                  onChange={(e) =>
                    setRowsToShowImputation(
                      e.target.value === 'all' ? 'all' : parseInt(e.target.value)
                    )
                  }
                >
                  {[10, 25, 50].map((count) =>
                    jsonResponse.imputed.data.length >= count ? (
                      <option key={count} value={count}>{count}</option>
                    ) : null
                  )}
                  <option value="all">{t('all')}</option>
                </select>
              </div>

              <div className="table-container">
                <table className="csv-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      {jsonResponse.imputed.columns.map((key, idx) => (
                        <th key={idx}>{key}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(rowsToShowImputation === 'all'
                      ? jsonResponse.imputed.data
                      : jsonResponse.imputed.data.slice(0, rowsToShowImputation)
                    ).map((row, rowIdx) => (
                      <tr key={rowIdx}>
                        <td>{rowIdx + 1}</td>
                        {jsonResponse.imputed.columns.map((key, idx) => (
                          <td
                            key={idx}
                            className={wasCellImputed(rowIdx, key) ? 'highlight-imputed-cell' : undefined}
                          >
                            {row[key]?.toString() ?? '-'}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {selectedOperation === 'imputation' && activeTabs.imputation === 'seeExplanationImp' && typeof jsonResponse?.explanation_plot === 'string' &&
          jsonResponse.explanation_plot.trim().length > 0 && (
            <div className="summary-plot-section">
              <div className="table-wrapper">
                <div className="table-container">
                  <table className="csv-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        {jsonResponse.imputed.columns.map((key, idx) => (
                          <th key={idx}>{key}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(rowsToShowImputation === 'all'
                        ? jsonResponse.imputed.data
                        : jsonResponse.imputed.data.slice(0, rowsToShowImputation)
                      ).map((row, rowIdx) => (
                        <tr key={rowIdx}>
                          <td>{rowIdx + 1}</td>
                          {jsonResponse.imputed.columns.map((key, idx) => (
                            <td
                              key={idx}
                              className={wasCellImputed(rowIdx, key) ? 'highlight-imputed-cell' : undefined}
                            >
                              {row[key]?.toString() ?? '-'}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="plot-wrapper" style={{ marginTop: '2rem' }}>
                <div className="table-container">
                  <div className="explanation-text">
                    {jsonResponse.explanation_plot}
                  </div>
                </div>
              </div>
            </div>
          )}

        {selectedOperation === 'anomaly' && activeTabs.anomaly === 'anomalyResults' && Array.isArray(jsonResponse?.anomalies?.data) && jsonResponse.anomalies.data.length > 0 && (
          <div className="summary-plot-section">
            <div className="table-wrapper">
              <div style={{ marginBottom: '1rem' }}>
                <label htmlFor="row-select" style={{ marginRight: '0.5rem' }}>
                  {t('toShow')}
                </label>
                <select
                  id="row-select"
                  value={rowsToShowAnomaly}
                  onChange={(e) =>
                    setRowsToShowAnomaly(
                      e.target.value === 'all' ? 'all' : parseInt(e.target.value)
                    )
                  }
                >
                  {[10, 25, 50].map((count) =>
                    jsonResponse.anomalies.data.length >= count ? (
                      <option key={count} value={count}>{count}</option>
                    ) : null
                  )}
                  <option value="all">{t('all')}</option>
                </select>
              </div>

              <div className="table-container">
                <table className="csv-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      {jsonResponse.anomalies.columns.map((key, idx) => (
                        <th key={idx}>{key}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(rowsToShowAnomaly === 'all'
                      ? jsonResponse.anomalies.data
                      : jsonResponse.anomalies.data.slice(0, rowsToShowAnomaly)
                    ).map((row, rowIdx) => (
                      <tr key={rowIdx}>
                        <td>{rowIdx + 1}</td>
                        {jsonResponse.anomalies.columns.map((key, idx) => {
                          let cellStyle = {};
                          if (key === 'Anomaly') {
                            if (row[key] === 0 || row[key] === '0') {
                              cellStyle = { backgroundColor: '#f3e6f9' };
                            } else {
                              cellStyle = { backgroundColor: '#bf91e5ff' };
                            }
                          }
                          return (
                            <td key={idx} style={cellStyle}>
                              {row[key]?.toString() ?? '-'}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {selectedOperation === 'anomaly' && activeTabs.anomaly === 'seeExplanationAno' && jsonResponse?.explanation_plot && (
          <div className="summary-plot-section">
            <div className="plot-wrapper">
              <div className="table-container">
                <div className="explanation-text">
                  {jsonResponse.explanation_plot}
                </div>
              </div>
            </div>
          </div>
        )}

        {selectedOperation === 'personal' && jsonResponse && (
          <div className="summary-plot-section">
            <div className="table-wrapper">
              <div className="table-container">
                {/* Tabelle für personal_0 */}
                {Array.isArray(jsonResponse?.personal_0?.data) &&
                  jsonResponse.personal_0.data.length > 0 && (
                    <div>
                      <table className="csv-table">
                        <thead>
                          <tr>
                            {jsonResponse.personal_0.columns.map((key, idx) => (
                              <th key={idx}>{key}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {jsonResponse.personal_0.data.map((row, rowIdx) => (
                            <tr key={rowIdx}>
                              {jsonResponse.personal_0.columns.map((key, colIdx) => (
                                <td key={colIdx}>{String(row[key])}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}

                {/* Tabelle für personal_1 */}
                {Array.isArray(jsonResponse?.personal_1?.data) && jsonResponse.personal_1.data.length > 0 && (
                  <div style={{ marginTop: '2rem' }}>
                    <table className="csv-table">
                      <thead>
                        <tr>
                          <th>Column</th>
                          <th style={{ backgroundColor: '#bf91e5ff' }}>Prediction</th>
                        </tr>
                      </thead>
                      <tbody>
                        {jsonResponse.personal_1.data.map((item, index) => (
                          <tr key={index}>
                            <td>{item.Column}</td>
                            <td style={{ backgroundColor: '#f3e6f9' }}>{item.Prediction}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
        {selectedOperation === 'summary' && jsonResponse && jsonResponse.inference && Array.isArray(jsonResponse.inference.data) && (
          <div className="summary-plot-section">
            <div className="plots-grid">
              {/* Inference Plot */}
              <div className="plot-wrapper">
                <h3 className="plot-title">{t('summaryInferenceTitle')}</h3>
                <div
                  style={{
                    width: '100%',
                    height: '400px',
                    border: '2px solid #ccc9c9ff',
                    boxSizing: 'border-box',
                    padding: '0',
                    position: 'relative'
                  }}
                >
                  <Plot
                    data={jsonResponse.inference.data.map((trace, index) => {
                      const parseArray = (str) => {
                        try {
                          return str.replace(/[\[\]'\n]+/g, '')
                            .split(/[\s,]+/)
                            .filter(Boolean)
                            .map((item) => isNaN(item) ? item : Number(item));
                        } catch {
                          return [];
                        }
                      };

                      const colors = ['#dfc7ebff', '#bf91e5ff'];

                      return {
                        ...trace,
                        x: typeof trace.x === 'string' ? parseArray(trace.x) : trace.x,
                        y: typeof trace.y === 'string' ? parseArray(trace.y) : trace.y,
                        marker: {
                          ...trace.marker,
                          color: colors[index % colors.length]
                        }
                      };
                    })}
                    layout={{
                      ...jsonResponse.inference.layout,
                      title: '',
                      xaxis: {
                        ...(jsonResponse.inference.layout?.xaxis || {}),
                        title: { text: t('featureType') },
                      },
                      yaxis: {
                        ...(jsonResponse.inference.layout?.yaxis || {}),
                        title: { text: t('count') },
                      },
                      margin: {
                        l: 100,
                        r: 100,
                        t: 100,
                        b: 100
                      }
                    }}
                    style={{
                      width: '100%',
                      height: '100%'
                    }}
                    config={{ responsive: true }}
                  />
                </div>
              </div>

              {/* Anomaly Pie Chart */}
              {jsonResponse.anomaly?.data && (
                <div className="plot-wrapper">
                  <h3 className="plot-title">{t('summaryAnomalyTitle')}</h3>
                  <div
                    style={{
                      width: '100%',
                      height: '400px',
                      border: '2px solid #ccc9c9ff',
                      boxSizing: 'border-box',
                      padding: '0', // optional
                      position: 'relative'
                    }}
                  >
                    <Plot
                      data={jsonResponse.anomaly.data.map((trace) => {
                        const parseArray = (str) => {
                          try {
                            return str.replace(/[\[\]'\n]+/g, '')
                              .split(/[\s,]+/)
                              .filter(Boolean)
                              .map((item) => isNaN(item) ? item : Number(item));
                          } catch {
                            return [];
                          }
                        };

                        const pieColors = ['#dfc7ebff', '#bf91e5ff', '#1f77b4', '#aec7e8'];

                        return {
                          ...trace,
                          labels: translatePlotLabels(
                            typeof trace.labels === 'string' ? parseArray(trace.labels) : trace.labels,
                            {
                              Anomaly: t('anomalyLabel'),
                              'No Anomaly': t('noAnomalyLabel'),
                            }
                          ),
                          values: typeof trace.values === 'string' ? parseArray(trace.values) : trace.values,
                          marker: {
                            ...trace.marker,
                            colors: pieColors
                          }
                        };
                      })}
                      layout={{
                        ...jsonResponse.anomaly.layout,
                        title: '',
                        margin: {
                          l: 100,
                          r: 100,
                          t: 100,
                          b: 100
                        }
                      }}
                      style={{
                        width: '100%',
                        height: '100%'
                      }}
                      config={{ responsive: true }}
                    />
                  </div>
                </div>
              )}

              {/* Personal Features Plot */}
              {jsonResponse.personal?.data && (
                <div className="plot-wrapper">
                  <h3 className="plot-title">{t('summaryPersonalTitle')}</h3>
                  <div
                    style={{
                      width: '100%',
                      height: '400px',
                      border: '2px solid #ccc9c9ff',
                      boxSizing: 'border-box',
                      padding: '0', // optional
                      position: 'relative'
                    }}
                  >
                    <Plot
                      data={jsonResponse.personal.data.map((trace) => {
                        const parseArray = (str) => {
                          try {
                            return str.replace(/[\[\]'\n]+/g, '')
                              .split(/[\s,]+/)
                              .filter(Boolean)
                              .map((item) => isNaN(item) ? item : Number(item));
                          } catch {
                            return [];
                          }
                        };

                        const pieColors = ['#dfc7ebff', '#bf91e5ff', '#1f77b4', '#aec7e8'];

                        return {
                          ...trace,
                          labels: translatePlotLabels(
                            typeof trace.labels === 'string' ? parseArray(trace.labels) : trace.labels,
                            {
                              'Personal Features': t('personalFeaturesLabel'),
                              'Non-Personal Features': t('nonPersonalFeaturesLabel'),
                            }
                          ),
                          values: typeof trace.values === 'string' ? parseArray(trace.values) : trace.values,
                          marker: {
                            ...trace.marker,
                            colors: pieColors
                          }
                        };
                      })}
                      layout={{
                        ...jsonResponse.personal.layout,
                        title: '',
                        margin: {
                          l: 100,
                          r: 100,
                          t: 100,
                          b: 100
                        }
                      }}
                      style={{
                        width: '100%',
                        height: '100%'
                      }}
                      config={{ responsive: true }}
                    />
                  </div>
                </div>
              )}

              {/* Imputation Pie Chart */}
              {jsonResponse.imputation?.data && (
                <div className="plot-wrapper">
                  <h3 className="plot-title">{t('summaryImputationTitle')}</h3>
                  <div
                    style={{
                      width: '100%',
                      height: '400px',
                      border: '2px solid #ccc9c9ff',
                      boxSizing: 'border-box',
                      padding: '0', // optional
                      position: 'relative'
                    }}
                  >
                    <Plot
                      data={jsonResponse.imputation.data.map((trace, index) => {
                        const parseArray = (str) => {
                          try {
                            return str.replace(/[\[\]'\n]+/g, '')
                              .split(/[\s,]+/)
                              .filter(Boolean)
                              .map((item) => isNaN(item) ? item : Number(item));
                          } catch {
                            return [];
                          }
                        };

                        const histColors = ['#dfc7ebff', '#bf91e5ff'];


                        return {
                          ...trace,
                          labels: translatePlotLabels(
                            typeof trace.labels === 'string' ? parseArray(trace.labels) : trace.labels,
                            {
                              'Imputed Values': t('imputedValuesLabel'),
                              'Not Imputed Values': t('notImputedValuesLabel'),
                            }
                          ),
                          values: typeof trace.values === 'string' ? parseArray(trace.values) : trace.values,
                          marker: {
                            ...trace.marker,
                            colors: histColors
                          }
                        };
                      })}
                      layout={{
                        ...jsonResponse.imputation.layout,
                        title: '',
                        margin: {
                          l: 100,
                          r: 100,
                          t: 100,
                          b: 100
                        }
                      }}
                      style={{
                        width: '100%',
                        height: '100%'
                      }}
                      config={{ responsive: true }}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {selectedOperation === 'cleaned' && jsonResponse && (
          <div className="summary-plot-section">
            {Array.isArray(cleanedPreviewTable?.data) && cleanedPreviewTable.data.length > 0 && (
              <div className="table-wrapper2">
                <h3 className="plot-title">Processed Table</h3>
                <div className="table-container2">
                  <table className="csv-table2">
                    <thead>
                      <tr>
                        {cleanedPreviewTable.columns.map((col, idx) => (
                          <th key={idx}>{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {cleanedPreviewTable.data.slice(0, 10).map((row, rowIdx) => (
                        <tr key={rowIdx}>
                          {cleanedPreviewTable.columns.map((col, colIdx) => (
                            <td key={colIdx}>{row[col]}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {selectedOperation === 'metadata' && jsonResponse && (
          <div className="summary-plot-section">
            <div className="table-wrapper">
              {/* General Metadata */}
              {jsonResponse?.general_metadata?.data && (
                <div className="table-container">
                  <h3>General Metadata</h3>
                  <table className="csv-table2">
                    <thead>
                      <tr>
                        <th>Key</th>
                        <th>Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {jsonResponse.general_metadata.data.map((row, idx) => {
                        const value = row["0"];
                        const labels = [
                          "Number of Rows",
                          "Number of Columns",
                          "Column Names",
                          "Missing Values",
                          "Duplicate Rows",
                          "Unique Values"
                        ];

                        return (
                          <tr key={idx}>
                            <td>{labels[idx] || `Row ${idx + 1}`}</td>
                            <td>
                              {typeof value === "object" ? (
                                <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>
                                  {JSON.stringify(value, null, 2)}
                                </pre>
                              ) : (
                                value?.toString()
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Featurewise Metadata */}
            {jsonResponse?.featurewise_metadata?.data && (
              <div className="table-wrapper" style={{ marginTop: '2rem' }}>
                <div className="table-container2">
                  <h3>Featurewise Metadata</h3>
                  <div style={{ overflowX: "auto" }}>
                    <table className="csv-table">
                      <thead>
                        <tr>
                          {jsonResponse.featurewise_metadata.columns.map((colName, idx) => (
                            <th key={idx}>{colName}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {jsonResponse.featurewise_metadata.data.map((row, rowIndex) => (
                          <tr key={rowIndex}>
                            {jsonResponse.featurewise_metadata.columns.map((colName, colIndex) => {
                              const cellValue = row[colName];
                              return (
                                <td key={colIndex}>
                                  {typeof cellValue === "object" ? (
                                    <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>
                                      {JSON.stringify(cellValue, null, 2)}
                                    </pre>
                                  ) : (
                                    cellValue?.toString()
                                  )}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>


      <div className="footer">
        <img src="/Hochschule-aalen.svg.png" alt="Hochschule Aalen" className="footer-logo" />
        <img src="/logo-m-wm_de.svg" alt="Ministerium für Wissenschaft, Forschung und Kunst Baden-Württemberg" className="footer-logo" />
        <div className="content">
          <p>{t('title')}</p>
        </div>
      </div>
    </div>

  );
}

export default App;
