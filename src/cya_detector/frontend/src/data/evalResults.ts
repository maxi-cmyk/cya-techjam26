// Mock evaluation data for the dashboard demo.
// Values are transcribed from docs/planning/nextSteps.md (as of the 2026-08-31 pull)
// rather than read from artifacts/, which is gitignored and empty in a fresh checkout.
// Swap this module for a real API/artifact reader once Task 10 packaging lands.

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
  { task: '9. Texture path', state: 'in-progress', label: 'Clean gate passed; Stage-1 robustness planned', decision: 'global_local 100% clean vs. 99.394% global-only mean' },
  { task: '10. Packaging', state: 'not-started', label: 'Not started', decision: 'Waiting for Task 9 robustness results' },
]

export interface PredictionResult {
  label: 'authentic' | 'ai_generated'
  confidence: number
}

// Deterministic mock predictor: stands in for the real inference API until
// Task 10's directory-inference contract is wired up.
export function mockPredict(fileName: string): PredictionResult {
  let hash = 0
  for (let i = 0; i < fileName.length; i += 1) {
    hash = (hash * 31 + fileName.charCodeAt(i)) >>> 0
  }
  const isAi = hash % 2 === 0
  const confidence = 0.7 + (hash % 30) / 100
  return {
    label: isAi ? 'ai_generated' : 'authentic',
    confidence: Math.min(confidence, 0.99),
  }
}
