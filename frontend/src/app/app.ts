import { Component, computed, signal } from '@angular/core';

/* ---------------------------------------------------------------------
   DEMO EDIT POINT (TypeScript)
   Change this text, commit and push. It shows in the page header and in
   the "How a change reaches this page" section.
--------------------------------------------------------------------- */
const VERSION = 'v1.0';

// Address of the Flask backend. It only answers while backend/app.py is
// running on this computer. The page works without it.
const API_URL = 'http://localhost:5000';

type MetricKey = 'accuracy' | 'precision' | 'recall' | 'f1';

interface ModelResult {
  name: string;
  short: string;
  source: 'ours' | 'paper';
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}

interface SampleRow {
  protocol: string;
  service: string;
  flag: string;
  src: number;
  dst: number;
  label: string;
  note: string;
}

interface GapBar {
  label: string;
  value: number;
  text: string;
  danger: boolean;
}

interface GapCard {
  model: string;
  bars: GapBar[];
}

/* ---------- Real numbers from our runs (also served by the backend) ---------- */

const BUILT_IN_RESULTS: ModelResult[] = [
  { name: 'Decision Tree', short: 'DT', source: 'ours', accuracy: 99.19, precision: 99.12, recall: 99.19, f1: 99.15 },
  { name: 'Random Forest', short: 'RF', source: 'ours', accuracy: 99.38, precision: 99.42, recall: 99.3, f1: 99.36 },
  { name: 'XGBoost', short: 'XGB', source: 'ours', accuracy: 99.61, precision: 99.66, recall: 99.53, f1: 99.59 },
  { name: 'XGBoost + feature engineering', short: 'XGB+FE', source: 'ours', accuracy: 99.62, precision: 99.66, recall: 99.55, f1: 99.61 },
  { name: 'Decision Tree (paper)', short: 'DT', source: 'paper', accuracy: 99.51, precision: 99.49, recall: 99.46, f1: 99.47 },
  { name: 'Random Forest (paper)', short: 'RF', source: 'paper', accuracy: 99.72, precision: 99.84, recall: 99.56, f1: 99.7 },
];

// The first five rows of KDDTrain+ (the labels are the dataset's own labels)
const SAMPLE_ROWS: SampleRow[] = [
  {
    protocol: 'tcp', service: 'ftp_data', flag: 'SF', src: 491, dst: 0, label: 'normal',
    note: 'A file-transfer connection that opened and closed cleanly. 491 bytes went out and none came back.',
  },
  {
    protocol: 'udp', service: 'other', flag: 'SF', src: 146, dst: 0, label: 'normal',
    note: 'A short UDP message. UDP has no handshake, so there is nothing to complete.',
  },
  {
    protocol: 'tcp', service: 'private', flag: 'S0', src: 0, dst: 0, label: 'neptune',
    note: 'A SYN flood, called neptune in the dataset. The sender starts a connection and never finishes it (S0: no reply seen, no data moved). The attack repeats this until the server is overwhelmed.',
  },
  {
    protocol: 'tcp', service: 'http', flag: 'SF', src: 232, dst: 8153, label: 'normal',
    note: 'A web page request. The browser sends a little (232 bytes), the server sends back a lot (8,153 bytes), and the connection ends cleanly.',
  },
  {
    protocol: 'tcp', service: 'http', flag: 'SF', src: 199, dst: 420, label: 'normal',
    note: 'Another ordinary web request: small question, slightly larger answer, clean finish.',
  },
];

/* ---------- Random split vs. cross-validation vs. official test file ---------- */

const GAP_ROWS = [
  { model: 'Decision Tree', paper: 99.51, single: 99.19, cvMean: 99.4, cvStd: 0.05, official: 79.49 },
  { model: 'Random Forest', paper: 99.72, single: 99.38, cvMean: 99.59, cvStd: 0.03, official: 77.87 },
];

const GAP_CARDS: GapCard[] = GAP_ROWS.map((g) => ({
  model: g.model,
  bars: [
    { label: 'Paper, one random split', value: g.paper, text: `${g.paper.toFixed(2)}%`, danger: false },
    { label: 'Ours, one random split', value: g.single, text: `${g.single.toFixed(2)}%`, danger: false },
    { label: 'Ours, 5-fold cross-validation', value: g.cvMean, text: `${g.cvMean.toFixed(2)} ± ${g.cvStd.toFixed(2)}`, danger: false },
    { label: 'Ours, official test file', value: g.official, text: `${g.official.toFixed(2)}%`, danger: true },
  ],
}));

const DROPS = GAP_ROWS.map((g) => g.cvMean - g.official);
const DROP_TEXT = `${Math.round(Math.min(...DROPS))} to ${Math.round(Math.max(...DROPS))}`;

/* ---------- Class balance of the 148,517 rows ---------- */

const NORMAL_COUNT = 77054;
const ATTACK_COUNT = 71463;
const TOTAL = NORMAL_COUNT + ATTACK_COUNT;

/* ---------- Checks that data coming from the API has the right shape ---------- */

function isModelResult(x: unknown): x is ModelResult {
  if (typeof x !== 'object' || x === null) {
    return false;
  }
  const r = x as Record<string, unknown>;
  return (
    typeof r['name'] === 'string' &&
    typeof r['short'] === 'string' &&
    (r['source'] === 'ours' || r['source'] === 'paper') &&
    ['accuracy', 'precision', 'recall', 'f1'].every((k) => typeof r[k] === 'number')
  );
}

@Component({
  selector: 'app-root',
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  readonly version = VERSION;

  readonly sections = [
    { id: 'about', label: 'Data' },
    { id: 'method', label: 'Method' },
    { id: 'results', label: 'Results' },
    { id: 'gap', label: 'Gap' },
    { id: 'ship', label: 'Shipping' },
  ];

  // Hero table
  readonly samples = SAMPLE_ROWS;
  readonly selectedRow = signal(2);
  readonly currentSample = computed(() => this.samples[this.selectedRow()]);

  // Data section
  readonly normalCount = NORMAL_COUNT;
  readonly attackCount = ATTACK_COUNT;
  readonly total = TOTAL;
  readonly normalPct = (NORMAL_COUNT / TOTAL) * 100;
  readonly attackPct = (ATTACK_COUNT / TOTAL) * 100;

  // Method section
  readonly steps = [
    { title: 'Combine the two files', text: 'KDDTrain+ and KDDTest+ are merged into one table of 148,517 connections, the same way the paper did.' },
    { title: 'Label each connection', text: 'The dataset names the exact attack (neptune, smurf and others). We reduce that to two labels: normal or attack.' },
    { title: 'Keep 25 of the 41 measurements', text: 'The paper ranked the measurements by importance and kept the 25 that mattered. We kept the same 25.' },
    { title: 'Turn words into numbers', text: 'Protocol, service and flag hold words such as tcp or http. They become numbers, and every column is then scaled to the range 0 to 1.' },
    { title: 'Train the models', text: 'Decision Tree and Random Forest, the paper’s best two, plus XGBoost, which the paper did not use. 80% of the rows train the model and 20% are held back for testing.' },
    { title: 'Measure', text: 'Accuracy, precision, recall and F1 on the held-back 20%, compared with the paper’s numbers.' },
  ];

  // Results section
  readonly results = signal<ModelResult[]>(BUILT_IN_RESULTS);
  readonly selectedModel = signal(3);
  readonly metrics: { key: MetricKey; label: string }[] = [
    { key: 'accuracy', label: 'Accuracy' },
    { key: 'precision', label: 'Precision' },
    { key: 'recall', label: 'Recall' },
    { key: 'f1', label: 'F1 score' },
  ];

  readonly ours = computed(() => this.results().filter((r) => r.source === 'ours'));

  readonly model = computed(() => {
    const list = this.ours();
    return list[Math.min(this.selectedModel(), list.length - 1)];
  });

  // The paper's row we compare against: the same model if the paper used it,
  // otherwise the paper's best model (Random Forest)
  readonly reference = computed(() => {
    const paper = this.results().filter((r) => r.source === 'paper');
    const current = this.model();
    return paper.find((p) => p.short === current.short) ?? paper.find((p) => p.short === 'RF') ?? paper[0];
  });

  readonly referenceNote = computed(() =>
    this.reference().short === this.model().short
      ? 'Compared with the same model in the paper.'
      : `The paper did not use this model, so it is compared with the paper’s best model, ${this.reference().name.replace(' (paper)', '')}.`,
  );

  // Highest value in each column, used to underline the best score in the table
  readonly best = computed(() => {
    const rows = this.results();
    return {
      accuracy: Math.max(...rows.map((r) => r.accuracy)),
      precision: Math.max(...rows.map((r) => r.precision)),
      recall: Math.max(...rows.map((r) => r.recall)),
      f1: Math.max(...rows.map((r) => r.f1)),
    };
  });

  // Gap section
  readonly gapCards = GAP_CARDS;
  readonly dropText = DROP_TEXT;

  // Shipping section
  readonly flow = [
    { title: 'Push', text: 'You commit and push to the main branch.' },
    { title: 'CI', text: 'Backend: syntax check and unit tests. Frontend: install and production build.' },
    { title: 'CD', text: 'Runs only if CI passed. Builds with the Pages address and publishes.' },
    { title: 'Live', text: 'GitHub Pages serves the new version of this page.' },
  ];

  // Backend status
  readonly apiState = signal<'checking' | 'online' | 'offline'>('checking');
  readonly dataSource = signal<'built-in' | 'api'>('built-in');

  readonly apiLabel = computed(() => {
    switch (this.apiState()) {
      case 'online':
        return 'Backend online';
      case 'offline':
        return 'Backend offline';
      default:
        return 'Checking backend';
    }
  });

  readonly apiNote = computed(() => {
    if (this.apiState() === 'checking') {
      return 'Looking for the Flask backend.';
    }
    if (this.apiState() === 'offline') {
      return 'The backend is not running. This page works without it and uses results built into the page.';
    }
    return this.dataSource() === 'api'
      ? 'The Flask backend answered, and the results on this page were loaded from it.'
      : 'The Flask backend answered, but its results were not usable, so built-in results are shown.';
  });

  constructor() {
    this.loadFonts();
    void this.loadFromApi();
  }

  /* ---------- Helpers used by the template ---------- */

  fmt(value: number): string {
    return value.toFixed(2);
  }

  fmtInt(value: number): string {
    return value.toLocaleString('en-US');
  }

  pct(value: number): string {
    return `${value.toFixed(1)}%`;
  }

  isAttack(row: SampleRow): boolean {
    return row.label !== 'normal';
  }

  // Bars are zoomed to the 98-100% range, otherwise every bar looks the same
  bar(value: number): number {
    return Math.max(0, Math.min(100, ((value - 98) / 2) * 100));
  }

  compare(ours: number, paper: number): string {
    const diff = ours - paper;
    if (Math.abs(diff) < 0.005) {
      return `Paper ${this.fmt(paper)}%, the same as ours`;
    }
    return `Paper ${this.fmt(paper)}%, ours is ${Math.abs(diff).toFixed(2)} points ${diff > 0 ? 'above' : 'below'}`;
  }

  // Fonts are added from here, not from CSS, so the build never needs the internet.
  // If they cannot load, the page quietly uses the system fonts.
  private loadFonts(): void {
    const urls = [
      'https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;700;800&display=swap',
      'https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&display=swap',
    ];
    for (const href of urls) {
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = href;
      document.head.appendChild(link);
    }
  }

  private motion(): ScrollBehavior {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth';
  }

  go(id: string): void {
    document.getElementById(id)?.scrollIntoView({ behavior: this.motion(), block: 'start' });
  }

  top(): void {
    window.scrollTo({ top: 0, behavior: this.motion() });
  }

  /* ---------- Ask the Flask backend for the results table ---------- */

  private async loadFromApi(): Promise<void> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 2500);
    try {
      const res = await fetch(`${API_URL}/api/results`, { signal: controller.signal });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      this.apiState.set('online');

      const body: unknown = await res.json();
      const models = (body as { models?: unknown } | null)?.models;
      if (Array.isArray(models) && models.length > 0 && models.every((m) => isModelResult(m))) {
        const rows = models as ModelResult[];
        if (rows.some((r) => r.source === 'ours') && rows.some((r) => r.source === 'paper')) {
          this.results.set(rows);
          this.selectedModel.set(0);
          this.dataSource.set('api');
        }
      }
    } catch {
      if (this.apiState() !== 'online') {
        this.apiState.set('offline');
      }
    } finally {
      clearTimeout(timer);
    }
  }
}