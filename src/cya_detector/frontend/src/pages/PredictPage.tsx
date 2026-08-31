import { useCallback, useState } from 'react'
import { mockPredict, type PredictionResult } from '../data/evalResults'

interface PredictedImage {
  id: string
  file: File
  previewUrl: string
  result: PredictionResult
}

function PredictPage() {
  const [images, setImages] = useState<PredictedImage[]>([])
  const [isDragging, setIsDragging] = useState(false)

  const addFiles = useCallback((files: FileList | null) => {
    if (!files) return
    const next: PredictedImage[] = Array.from(files)
      .filter((file) => file.type.startsWith('image/'))
      .map((file) => ({
        id: `${file.name}-${file.lastModified}-${Math.random().toString(36).slice(2)}`,
        file,
        previewUrl: URL.createObjectURL(file),
        result: mockPredict(file.name),
      }))
    setImages((prev) => [...next, ...prev])
  }, [])

  return (
    <section className="predict-page">
      <div className="predict-intro">
        <h1>Detect AI-generated images</h1>
        <p>
          Drop an image below to see a prediction. This demo uses a{' '}
          <strong>mock predictor</strong> — it is not calling the real
          controlled-RINE model yet. Wire this page up to a real inference
          endpoint once Task 10 packaging exposes one.
        </p>
      </div>

      <label
        className={`dropzone ${isDragging ? 'dragging' : ''}`}
        onDragOver={(event) => {
          event.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(event) => {
          event.preventDefault()
          setIsDragging(false)
          addFiles(event.dataTransfer.files)
        }}
      >
        <input
          type="file"
          accept="image/*"
          multiple
          onChange={(event) => addFiles(event.target.files)}
          hidden
        />
        <span>Drag and drop images here, or click to choose files</span>
      </label>

      {images.length > 0 && (
        <div className="results-grid">
          {images.map(({ id, file, previewUrl, result }) => (
            <article key={id} className="result-card">
              <img src={previewUrl} alt={file.name} />
              <div className="result-body">
                <div className="result-filename" title={file.name}>
                  {file.name}
                </div>
                <span className={`verdict-badge ${result.label}`}>
                  {result.label === 'ai_generated' ? 'AI-generated' : 'Authentic'}
                </span>
                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{ width: `${Math.round(result.confidence * 100)}%` }}
                  />
                </div>
                <span className="confidence-label">
                  {Math.round(result.confidence * 100)}% confidence
                </span>
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  )
}

export default PredictPage
