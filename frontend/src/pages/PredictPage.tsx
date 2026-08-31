import { useCallback, useState } from 'react'
import { predict, type PredictionResult } from '../data/evalResults'

interface PredictedImage {
  id: string
  file: File
  previewUrl: string
  result: PredictionResult | null
  error: string | null
}

function PredictPage() {
  const [images, setImages] = useState<PredictedImage[]>([])
  const [isDragging, setIsDragging] = useState(false)

  const addFiles = useCallback((files: FileList | null) => {
    if (!files) return
    const picked = Array.from(files).filter((file) => file.type.startsWith('image/'))
    if (picked.length === 0) return

    const pending: PredictedImage[] = picked.map((file) => ({
      id: `${file.name}-${file.lastModified}-${Math.random().toString(36).slice(2)}`,
      file,
      previewUrl: URL.createObjectURL(file),
      result: null,
      error: null,
    }))
    setImages((prev) => [...pending, ...prev])

    predict(picked).then(({ results, errors }) => {
      setImages((prev) =>
        prev.map((image) => {
          const index = pending.findIndex((entry) => entry.id === image.id)
          if (index === -1) return image
          const result = results.get(pending[index].file.name)
          const error = errors.get(pending[index].file.name)
          return { ...image, result: result ?? null, error: error ?? (result ? null : 'Prediction failed') }
        }),
      )
    })
  }, [])

  return (
    <section className="predict-page">
      <div className="predict-intro">
        <h1>Detect AI-generated images</h1>
        <p>
          Drop an image below to score it with the controlled-RINE model,
          served by the local backend API.
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
          {images.map(({ id, file, previewUrl, result, error }) => (
            <article key={id} className="result-card">
              <img src={previewUrl} alt={file.name} />
              <div className="result-body">
                <div className="result-filename" title={file.name}>
                  {file.name}
                </div>
                {error && <span className="verdict-badge error">{error}</span>}
                {!error && !result && <span className="verdict-badge pending">Scoring…</span>}
                {result && (
                  <>
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
                  </>
                )}
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  )
}

export default PredictPage
