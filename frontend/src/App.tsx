import { FormEvent, useEffect, useState } from "react";

type PredictionResponse = {
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
  important_keywords: string[];
};

type ModelInfoResponse = {
  model_name: string;
  version: string;
  framework: string;
  dataset: string;
  num_classes: number;
  training_accuracy: number;
};

type ApiError = {
  detail?: string | string[];
  message?: string;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL || (import.meta.env.PROD ? "" : "http://127.0.0.1:8000");

function App() {
  const [text, setText] = useState("" );
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/model-info`)
      .then(async (resp) => {
        if (!resp.ok) {
          throw new Error("Unable to fetch model info");
        }
        return resp.json();
      })
      .then((data) => setModelInfo(data))
      .catch(() => {
        setModelInfo(null);
      });
  }, []);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setResult(null);

    const trimmed = text.trim();
    if (!trimmed) {
      setError("Please enter a news headline or article.");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: trimmed }),
      });

      if (!response.ok) {
        const body: ApiError = await response.json();
        throw new Error(
          body.detail ? Array.isArray(body.detail) ? body.detail.join(", ") : body.detail : "Prediction failed"
        );
      }

      const prediction: PredictionResponse = await response.json();
      setResult(prediction);
    } catch (caught) {
      if (caught instanceof Error) {
        setError(caught.message);
      } else {
        setError("Unexpected error while submitting prediction.");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Fake News Detector</p>
          <h1>Classify news as real or fake with BERT</h1>
          <p className="subtitle">
            Paste an article or headline and get a prediction with confidence,
            probability scores, and important keywords.
          </p>
        </div>
        <div className="meta-card">
          <h2>Model details</h2>
          {modelInfo ? (
            <dl>
              <div>
                <dt>Model</dt>
                <dd>{modelInfo.model_name}</dd>
              </div>
              <div>
                <dt>Version</dt>
                <dd>{modelInfo.version}</dd>
              </div>
              <div>
                <dt>Framework</dt>
                <dd>{modelInfo.framework}</dd>
              </div>
              <div>
                <dt>Dataset</dt>
                <dd>{modelInfo.dataset}</dd>
              </div>
              <div>
                <dt>Classes</dt>
                <dd>{modelInfo.num_classes}</dd>
              </div>
              <div>
                <dt>Accuracy</dt>
                <dd>{(modelInfo.training_accuracy * 100).toFixed(1)}%</dd>
              </div>
            </dl>
          ) : (
            <p className="meta-loading">Unable to fetch model metadata.</p>
          )}
        </div>
      </header>

      <main>
        <section className="form-panel">
          <form onSubmit={handleSubmit}>
            <label htmlFor="newsText">Enter a news headline or article</label>
            <textarea
              id="newsText"
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="Type or paste text here..."
              rows={10}
            />

            <button type="submit" disabled={loading}>
              {loading ? "Analyzing..." : "Detect Fake News"}
            </button>

            {error ? <div className="alert">{error}</div> : null}
          </form>
        </section>

        {result ? (
          <section className="result-panel">
            <div className="result-card">
              <h2>Prediction</h2>
              <p className={`prediction-badge ${result.prediction.toLowerCase()}`}>
                {result.prediction}
              </p>
              <p className="confidence">
                Confidence: {(result.confidence * 100).toFixed(1)}%
              </p>
              <div className="probabilities">
                <h3>Probability scores</h3>
                <ul>
                  {Object.entries(result.probabilities).map(([label, score]) => (
                    <li key={label}>
                      <span>{label}</span>
                      <strong>{(score * 100).toFixed(1)}%</strong>
                    </li>
                  ))}
                </ul>
              </div>
              <div className="keywords">
                <h3>Important keywords</h3>
                {result.important_keywords.length > 0 ? (
                  <ul>
                    {result.important_keywords.map((keyword) => (
                      <li key={keyword}>{keyword}</li>
                    ))}
                  </ul>
                ) : (
                  <p>No keywords were extracted.</p>
                )}
              </div>
            </div>
          </section>
        ) : null}
      </main>

      <footer>
        <p>Powered by FastAPI and BERT. Keep your backend running on port 8000.</p>
      </footer>
    </div>
  );
}

export default App;
