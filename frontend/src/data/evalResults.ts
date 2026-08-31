// Dashboard summary data for the submission build.
// Values are transcribed from docs/planning/nextSteps.md (as of the 2026-08-31 pull)
// rather than read from artifacts/, which is gitignored and empty in a fresh checkout.

export interface ModelResult {
  component: string
  clean: number | null
  robustness: number | null
  locked: number | null
  status: 'retained' | 'rejected' | 'diagnostic' | 'not-run'
  note: string
}

export const modelResults: ModelResult[] = [
  {
    component: 'Controlled RINE Stage B',
    clean: 100.0,
    robustness: 99.62,
    locked: 99.81,
    status: 'retained',
    note: 'Retained as the pre-Task-9 parent. Seeds 42/43/44: 99.85% / 99.81% / 99.78%.',
  },
  {
    component: 'Existing clean-trained RINE',
    clean: null,
    robustness: null,
    locked: 96.2,
    status: 'rejected',
    note: 'Superseded by controlled RINE.',
  },
  {
    component: 'Frozen CLIP Stage A',
    clean: 97.58,
    robustness: null,
    locked: 94.71,
    status: 'diagnostic',
    note: 'Retained only as the mandatory baseline comparator.',
  },
  {
    component: 'RINE + frequency',
    clean: null,
    robustness: null,
    locked: 52.15,
    status: 'rejected',
    note: 'Mean delta -47.66 pts vs. parent; seeds 43/44 unstable. Early exit stays disabled.',
  },
  {
    component: 'RINE + Lab',
    clean: null,
    robustness: null,
    locked: 98.95,
    status: 'rejected',
    note: 'Mean delta -0.87 pts; AI-generated accuracy regressed 1.82 pts beyond the 1-pt gate.',
  },
  {
    component: 'Frequency magnitude/residual',
    clean: 83.03,
    robustness: null,
    locked: null,
    status: 'diagnostic',
    note: 'Best standalone frequency representation; fusion rejected, preserved for diagnostics only.',
  },
  {
    component: 'Lab correlation',
    clean: 82.42,
    robustness: null,
    locked: null,
    status: 'diagnostic',
    note: '~99-100% extractor confidence and validity; fusion was rejected.',
  },
  {
    component: 'Reference-free PRNU v2 only',
    clean: 85.25,
    robustness: 70.92,
    locked: 78.09,
    status: 'diagnostic',
    note: 'Substantially weaker than controlled RINE; preserved for diagnostics only.',
  },
  {
    component: 'RINE + PRNU v2',
    clean: null,
    robustness: null,
    locked: 33.43,
    status: 'rejected',
    note: 'Seeds 42/43 collapsed to 0.32%/0.43%; seed 44 reached 99.55% but still missed parent.',
  },
]

export interface TaskStatus {
  task: string
  state: 'complete' | 'in-progress' | 'not-started'
  label: string
  decision: string
}

export const taskProgress: TaskStatus[] = [
  { task: '1. Project skeleton', state: 'complete', label: 'Complete', decision: 'Frozen Colab configuration and reproducibility helpers' },
  { task: '2. Data contract', state: 'complete', label: 'Complete', decision: '19,882 eligible primary images; fixed-Q96 selected later by Stage A' },
  { task: '3. Independent transforms', state: 'complete', label: 'Complete', decision: '19,460 variants from 1,390 clean parents across 14 independent cells' },
  { task: '4. Frozen CLIP Stage A', state: 'complete', label: 'Complete', decision: 'Three-seed locked 50/50 mean: 94.71%' },
  { task: '5. Evaluation harness', state: 'complete', label: 'Complete', decision: 'Evaluated 20,850 development rows; final_test sealed' },
  { task: '6. RINE Stage B', state: 'complete', label: 'Complete and retained', decision: '100.00% clean / 99.62% robustness / 99.81% locked' },
  { task: '7. Frequency Stage 1', state: 'complete', label: 'Complete; fusion rejected', decision: 'RINE+frequency 52.15% vs. 99.81% parent' },
  { task: '8. Color/physical auxiliaries', state: 'complete', label: 'Complete; Lab fusion rejected', decision: 'RINE+Lab 98.95%; AI accuracy regressed 1.82 pts' },
  { task: '8B. Native physical pilot', state: 'complete', label: 'Complete; no feature retained', decision: 'PRNU AUC 0.538/0.543 missed the 0.60 gate' },
  { task: '8B-v2. Improved PRNU estimator', state: 'complete', label: 'Complete; fusion rejected', decision: 'PRNU-only 78.09% locked; RINE+PRNU collapsed to 33.43%' },
  { task: '9. Texture path', state: 'complete', label: 'Complete; rejected at Stage-1 robustness', decision: 'reject_texture_robustness_stage1 — 93.13% robustness mean vs. 99.80% for controlled RINE; controlled RINE retained' },
  { task: '10. Packaging', state: 'complete', label: 'Complete', decision: 'final_test: 141 samples, 99.29% accuracy, 1.39% FPR, 0.00% FNR, controlled RINE seed 42' },
]

export interface PredictionResult {
  label: 'authentic' | 'ai_generated'
  confidence: number
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

interface PredictApiResponse {
  predictions: { filename: string; label: 'authentic' | 'ai_generated'; confidence: number }[]
  errors: { filename: string; code: string; message: string }[]
}

// Calls the backend's POST /predict (backend/app.py), which wraps the same
// run_inference() the CLI uses.
export async function predict(
  files: File[],
): Promise<{ results: Map<string, PredictionResult>; errors: Map<string, string> }> {
  const body = new FormData()
  for (const file of files) body.append('files', file)

  const response = await fetch(`${API_BASE_URL}/predict`, { method: 'POST', body })
  if (!response.ok) {
    throw new Error(`Prediction request failed with status ${response.status}`)
  }
  const data: PredictApiResponse = await response.json()

  const results = new Map<string, PredictionResult>()
  for (const row of data.predictions) {
    results.set(row.filename, { label: row.label, confidence: row.confidence })
  }
  const errors = new Map<string, string>()
  for (const row of data.errors) {
    errors.set(row.filename, row.message)
  }
  return { results, errors }
}
