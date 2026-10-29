import { AfterViewInit, Component, ElementRef, HostListener, ViewChild, computed, signal } from '@angular/core';
import { CountUp } from './count-up';
import { Cursor } from './cursor';
import { Globe } from './globe';
import { Reveal } from './reveal';

const VERSION = 'v1.1';
const API_URL = (globalThis as typeof globalThis & { NSLKDD_API_URL?: string }).NSLKDD_API_URL ?? 'http://localhost:5000';

type Theme = 'light' | 'dark';
type Mode = 'single' | 'batch';
type View = 'single' | 'batch' | 'comparison';
type Prediction = {
  row?: number;
  predicted_class: number;
  prediction: 'Normal' | 'Attack';
  actual_label?: string;
  probabilities?: { Normal: number; Attack: number };
};
type Metrics = { accuracy: number; precision: number; recall: number; f1: number };
type CsvResult = {
  model: string;
  total_records: number;
  normal_count: number;
  attack_count: number;
  records: Prediction[];
  metrics?: Metrics;
  evaluation_note?: string;
};
type ModelOption = { id: string; name: string; available: boolean; error?: string };
type ComparisonRow = {
  id: string;
  name: string;
  total_records?: number;
  normal_count?: number;
  attack_count?: number;
  metrics?: Metrics;
  error?: string;
};
type Feature = {
  key: string;
  label: string;
  hint: string;
  kind: 'number' | 'select';
  options?: { label: string; value: string | number }[];
  rate?: boolean;
};
type FeatureGroup = { title: string; description: string; fields: Feature[] };

const protocolOptions = [
  { label: 'TCP', value: 'tcp' }, { label: 'UDP', value: 'udp' }, { label: 'ICMP', value: 'icmp' },
];
const serviceNames = [
  'aol','auth','bgp','courier','csnet_ns','ctf','daytime','discard','domain','domain_u','echo','eco_i','ecr_i','efs','exec','finger','ftp','ftp_data','gopher','harvest','hostnames','http','http_2784','http_443','http_8001','imap4','IRC','iso_tsap','klogin','kshell','ldap','link','login','mtp','name','netbios_dgm','netbios_ns','netbios_ssn','netstat','nnsp','nntp','ntp_u','other','pm_dump','pop_2','pop_3','printer','private','red_i','remote_job','rje','shell','smtp','sql_net','ssh','sunrpc','supdup','systat','telnet','tftp_u','tim_i','time','urh_i','urp_i','uucp','uucp_path','vmnet','whois','X11','Z39_50',
].map((value) => ({ label: value, value }));
const flagOptions = ['OTH','REJ','RSTO','RSTOS0','RSTR','S0','S1','S2','S3','SF','SH'].map((value) => ({ label: value, value }));
const yesNoOptions = [{ label: 'No', value: 0 }, { label: 'Yes', value: 1 }];

const FEATURE_GROUPS: FeatureGroup[] = [
  {
    title: 'Connection details', description: 'Protocol, service, connection state and duration.', fields: [
      { key: 'protocol_type', label: 'Protocol type', hint: 'Transport protocol used by the connection.', kind: 'select', options: protocolOptions },
      { key: 'service', label: 'Service', hint: 'Destination network service recorded by NSL-KDD.', kind: 'select', options: serviceNames },
      { key: 'flag', label: 'Connection flag', hint: 'Status of the connection as recorded in the dataset.', kind: 'select', options: flagOptions },
      { key: 'duration', label: 'Duration (seconds)', hint: 'Length of the connection.', kind: 'number' },
    ],
  },
  {
    title: 'Traffic volume', description: 'Bytes and unusual fragments observed in the connection.', fields: [
      { key: 'src_bytes', label: 'Source bytes', hint: 'Bytes sent from source to destination.', kind: 'number' },
      { key: 'dst_bytes', label: 'Destination bytes', hint: 'Bytes sent from destination back to source.', kind: 'number' },
      { key: 'wrong_fragment', label: 'Wrong fragments', hint: 'Count of malformed or unusual fragments.', kind: 'number' },
      { key: 'hot', label: 'Hot indicators', hint: 'Dataset count of certain suspicious operations.', kind: 'number' },
      { key: 'num_failed_logins', label: 'Failed logins', hint: 'Number of failed login attempts.', kind: 'number' },
      { key: 'logged_in', label: 'Successfully logged in', hint: 'Whether the connection achieved a successful login.', kind: 'select', options: yesNoOptions },
    ],
  },
  {
    title: 'Connection patterns', description: 'Counts and error rates across recent connections.', fields: [
      { key: 'count', label: 'Connection count', hint: 'Connections to the same host in the observation window.', kind: 'number' },
      { key: 'srv_count', label: 'Service count', hint: 'Connections to the same service in the observation window.', kind: 'number' },
      { key: 'serror_rate', label: 'SYN error rate', hint: 'Rate of connections with SYN errors (0–1).', kind: 'number', rate: true },
      { key: 'same_srv_rate', label: 'Same service rate', hint: 'Rate of connections to the same service (0–1).', kind: 'number', rate: true },
      { key: 'diff_srv_rate', label: 'Different service rate', hint: 'Rate of connections to different services (0–1).', kind: 'number', rate: true },
    ],
  },
  {
    title: 'Destination host patterns', description: 'Aggregated traffic statistics for the destination host.', fields: [
      { key: 'dst_host_count', label: 'Destination host count', hint: 'Recent connections to the destination host.', kind: 'number' },
      { key: 'dst_host_srv_count', label: 'Destination service count', hint: 'Recent connections to the same service at that host.', kind: 'number' },
      { key: 'dst_host_same_srv_rate', label: 'Same service rate', hint: 'Destination host same-service rate (0–1).', kind: 'number', rate: true },
      { key: 'dst_host_diff_srv_rate', label: 'Different service rate', hint: 'Destination host different-service rate (0–1).', kind: 'number', rate: true },
      { key: 'dst_host_same_src_port_rate', label: 'Same source-port rate', hint: 'Destination host same-source-port rate (0–1).', kind: 'number', rate: true },
      { key: 'dst_host_srv_diff_host_rate', label: 'Service different-host rate', hint: 'Rate of same-service connections to different hosts (0–1).', kind: 'number', rate: true },
      { key: 'dst_host_serror_rate', label: 'Host SYN error rate', hint: 'Destination host SYN error rate (0–1).', kind: 'number', rate: true },
      { key: 'dst_host_srv_serror_rate', label: 'Host service SYN error rate', hint: 'Destination host same-service SYN error rate (0–1).', kind: 'number', rate: true },
      { key: 'dst_host_rerror_rate', label: 'Host REJ error rate', hint: 'Destination host REJ error rate (0–1).', kind: 'number', rate: true },
      { key: 'dst_host_srv_rerror_rate', label: 'Host service REJ error rate', hint: 'Destination host same-service REJ error rate (0–1).', kind: 'number', rate: true },
    ],
  },
];

const DEFAULT_FEATURES: Record<string, string | number> = {
  duration: 0, protocol_type: 'tcp', service: 'http', flag: 'SF', src_bytes: 232, dst_bytes: 8153,
  wrong_fragment: 0, hot: 0, num_failed_logins: 0, logged_in: 1, count: 1, srv_count: 1,
  serror_rate: 0, same_srv_rate: 1, diff_srv_rate: 0, dst_host_count: 1, dst_host_srv_count: 1,
  dst_host_same_srv_rate: 1, dst_host_diff_srv_rate: 0, dst_host_same_src_port_rate: 0,
  dst_host_srv_diff_host_rate: 0, dst_host_serror_rate: 0, dst_host_srv_serror_rate: 0,
  dst_host_rerror_rate: 0, dst_host_srv_rerror_rate: 0,
};

const SCENARIOS: Record<string, { title: string; description: string; features: Record<string, string | number> }> = {
  normal: {
    title: 'Normal web connection',
    description: 'A clean HTTP connection with a small request and a larger response. Example features are illustrative.',
    features: { ...DEFAULT_FEATURES, protocol_type: 'tcp', service: 'http', flag: 'SF', src_bytes: 232, dst_bytes: 8153, logged_in: 1, count: 4, srv_count: 4, dst_host_count: 155, dst_host_srv_count: 255, dst_host_same_srv_rate: 1, dst_host_same_src_port_rate: 0.01 },
  },
  dos: {
    title: 'DoS-like connection pattern',
    description: 'A sample pattern with repeated rejected connections. It is a demonstration scenario, not a guarantee of an attack prediction.',
    features: { ...DEFAULT_FEATURES, protocol_type: 'tcp', service: 'private', flag: 'REJ', src_bytes: 0, dst_bytes: 0, logged_in: 0, count: 229, srv_count: 10, same_srv_rate: 0.04, diff_srv_rate: 0.06, dst_host_count: 255, dst_host_srv_count: 10, dst_host_same_srv_rate: 0.04, dst_host_diff_srv_rate: 0.06, dst_host_rerror_rate: 1, dst_host_srv_rerror_rate: 1 },
  },
  udp: {
    title: 'Short UDP message',
    description: 'A small UDP record from the dataset style. It is included to explore how the model responds to different inputs.',
    features: { ...DEFAULT_FEATURES, protocol_type: 'udp', service: 'other', flag: 'SF', src_bytes: 146, dst_bytes: 0, logged_in: 0, count: 1, srv_count: 1, dst_host_count: 1, dst_host_srv_count: 1 },
  },
};

@Component({
  selector: 'app-root',
  imports: [Globe, Reveal, CountUp, Cursor],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App implements AfterViewInit {
  @ViewChild('navLinks', { static: true }) private navLinks!: ElementRef<HTMLElement>;
  readonly navPill = signal<{ x: number; w: number; show: boolean }>({ x: 0, w: 0, show: false });
  readonly version = VERSION;
  readonly theme = signal<Theme>('light');
  readonly mode = signal<Mode>('single');
  readonly activeView = signal<View>('single');
  readonly scrolled = signal(false);
  readonly featureGroups = FEATURE_GROUPS;
  readonly scenarios = SCENARIOS;
  readonly scenarioKeys = Object.keys(SCENARIOS);
  readonly selectedScenario = signal('normal');
  readonly features = signal<Record<string, string | number>>({ ...DEFAULT_FEATURES });
  readonly inferenceModels = signal<ModelOption[]>([]);
  readonly selectedModel = signal('random_forest');
  readonly apiState = signal<'checking' | 'online' | 'offline'>('checking');
  readonly singleBusy = signal(false);
  readonly singleError = signal('');
  readonly singleResult = signal<Prediction | null>(null);
  readonly selectedCsv = signal<File | null>(null);
  readonly csvBusy = signal(false);
  readonly csvError = signal('');
  readonly csvResult = signal<CsvResult | null>(null);
  readonly comparisonBusy = signal(false);
  readonly comparisonError = signal('');
  readonly comparisonResults = signal<ComparisonRow[] | null>(null);
  readonly selectedComparisonModels = signal<string[]>(['random_forest', 'xgboost']);
  readonly visibleRecords = computed(() => (this.csvResult()?.records ?? []).slice(0, 100));
  readonly totalFeatures = FEATURE_GROUPS.reduce((sum, group) => sum + group.fields.length, 0);
  readonly apiLabel = computed(() => this.apiState() === 'online' ? 'Inference API connected' : this.apiState() === 'checking' ? 'Checking API' : 'API unavailable');

  constructor() {
    try {
      const saved = localStorage.getItem('nslkdd-theme');
      if (saved === 'light' || saved === 'dark') this.theme.set(saved);
    } catch { /* Continue with the default theme. */ }
    void this.loadInferenceModels();
  }

  ngAfterViewInit(): void {
    this.moveNavPill();
    document.fonts?.ready.then(() => this.moveNavPill());
  }

  navigate(view: View): void {
    this.activeView.set(view);
    this.mode.set(view === 'single' ? 'single' : 'batch');
    this.moveNavPill();
    document.getElementById('workspace')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  setView(view: View): void {
    this.activeView.set(view);
    this.mode.set(view === 'single' ? 'single' : 'batch');
    this.moveNavPill();
  }

  /** Slides the nav highlight under a link (hovered link, or the active one when nothing is hovered). */
  moveNavPill(hovered?: HTMLElement): void {
    queueMicrotask(() => {
      const box = this.navLinks?.nativeElement;
      const el = hovered ?? box?.querySelector<HTMLElement>('button.active') ?? null;
      if (!box || !el) { this.navPill.set({ x: 0, w: 0, show: false }); return; }
      this.navPill.set({ x: el.offsetLeft, w: el.offsetWidth, show: true });
    });
  }

  @HostListener('window:resize')
  onResize(): void { this.moveNavPill(); }

  private readonly spot = { x: 0, y: 0, tx: 0, ty: 0, raf: 0, el: null as HTMLElement | null };

  onHeroMove(event: PointerEvent): void {
    const el = event.currentTarget as HTMLElement;
    const rect = el.getBoundingClientRect();
    const s = this.spot;
    s.tx = event.clientX - rect.left;
    s.ty = event.clientY - rect.top;
    if (!s.el) { s.x = s.tx; s.y = s.ty; }
    s.el = el;
    el.style.setProperty('--spot', '1');
    if (!s.raf) s.raf = requestAnimationFrame(() => this.spotTick());
  }

  onHeroLeave(event: PointerEvent): void {
    (event.currentTarget as HTMLElement).style.setProperty('--spot', '0');
  }

  private spotTick(): void {
    const s = this.spot;
    s.raf = 0;
    if (!s.el) return;
    s.x += (s.tx - s.x) * 0.12;
    s.y += (s.ty - s.y) * 0.12;
    s.el.style.setProperty('--mx', `${s.x.toFixed(1)}px`);
    s.el.style.setProperty('--my', `${s.y.toFixed(1)}px`);
    if (Math.abs(s.tx - s.x) > 0.3 || Math.abs(s.ty - s.y) > 0.3) s.raf = requestAnimationFrame(() => this.spotTick());
  }

  pct(part: number, total: number): string {
    return total ? ((part / total) * 100).toFixed(1) : '0.0';
  }

  @HostListener('window:scroll')
  onWindowScroll(): void {
    this.scrolled.set(window.scrollY > 18);
  }

  toggleTheme(): void {
    const next: Theme = this.theme() === 'dark' ? 'light' : 'dark';
    this.theme.set(next);
    try { localStorage.setItem('nslkdd-theme', next); } catch { /* Theme still works for this session. */ }
  }

  modelName(id: string): string {
    return this.inferenceModels().find((model) => model.id === id)?.name ?? id;
  }

  async loadInferenceModels(): Promise<void> {
    this.apiState.set('checking');
    try {
      const response = await fetch(`${API_URL}/api/models`);
      if (!response.ok) throw new Error(`API returned ${response.status}`);
      const body = await response.json();
      const models = (body.models ?? []) as ModelOption[];
      this.inferenceModels.set(models.filter((model) => model.available));
      if (!this.inferenceModels().some((model) => model.id === this.selectedModel())) {
        this.selectedModel.set(this.inferenceModels()[0]?.id ?? 'random_forest');
      }
      this.apiState.set('online');
    } catch {
      this.inferenceModels.set([]);
      this.apiState.set('offline');
    }
  }

  setFeature(key: string, event: Event): void {
    const target = event.target as HTMLInputElement | HTMLSelectElement;
    const field = this.featureGroups.flatMap((group) => group.fields).find((item) => item.key === key);
    const value: string | number = field?.kind === 'number' ? Number(target.value || 0) : target.value;
    this.features.update((current) => ({ ...current, [key]: value }));
    this.singleResult.set(null);
    this.singleError.set('');
  }

  featureValue(key: string): string | number {
    return this.features()[key] ?? 0;
  }

  useScenario(key: string): void {
    const scenario = this.scenarios[key];
    if (!scenario) return;
    this.selectedScenario.set(key);
    this.features.set({ ...scenario.features });
    this.singleResult.set(null);
    this.singleError.set('');
  }

  async runSinglePrediction(): Promise<void> {
    if (!this.inferenceModels().length) {
      this.singleError.set('The inference API is unavailable. Start the Flask backend and check that model artifacts are loaded.');
      return;
    }
    this.singleBusy.set(true);
    this.singleError.set('');
    this.singleResult.set(null);
    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: this.selectedModel(), input_format: 'raw', features: this.features() }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error ?? `Prediction failed (${response.status})`);
      this.singleResult.set(body.result as Prediction);
    } catch (error) {
      this.singleError.set(error instanceof Error ? error.message : 'Prediction request failed.');
    } finally {
      this.singleBusy.set(false);
    }
  }

  chooseCsv(event: Event): void {
    this.acceptCsv((event.target as HTMLInputElement).files?.[0] ?? null);
  }

  dropCsv(event: DragEvent): void {
    event.preventDefault();
    this.acceptCsv(event.dataTransfer?.files?.[0] ?? null);
  }

  private acceptCsv(file: File | null): void {
    this.csvError.set('');
    this.csvResult.set(null);
    this.comparisonResults.set(null);
    this.comparisonError.set('');
    if (!file) { this.selectedCsv.set(null); return; }
    if (!file.name.toLowerCase().endsWith('.csv')) {
      this.selectedCsv.set(null);
      this.csvError.set('Please choose a .csv file.');
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      this.selectedCsv.set(null);
      this.csvError.set('This file exceeds the 5 MB upload limit.');
      return;
    }
    this.selectedCsv.set(file);
  }

  async loadSampleCsv(filename: string): Promise<void> {
    this.csvError.set('');
    this.csvResult.set(null);
    this.comparisonResults.set(null);
    try {
      const url = new URL(`samples/${filename}`, document.baseURI);
      const response = await fetch(url);
      if (!response.ok) throw new Error('The sample file could not be loaded.');
      const blob = await response.blob();
      this.acceptCsv(new File([blob], filename, { type: 'text/csv' }));
      this.mode.set('batch');
    } catch (error) {
      this.csvError.set(error instanceof Error ? error.message : 'Could not load sample data.');
    }
  }

  async runCsvPrediction(): Promise<void> {
    const file = this.selectedCsv();
    if (!file) { this.csvError.set('Choose a CSV file or load a sample first.'); return; }
    this.csvBusy.set(true);
    this.csvError.set('');
    this.csvResult.set(null);
    try {
      const form = new FormData();
      form.append('file', file);
      form.append('model', this.selectedModel());
      form.append('input_format', 'raw');
      const response = await fetch(`${API_URL}/api/predict/csv`, { method: 'POST', body: form });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error ?? `CSV analysis failed (${response.status})`);
      this.csvResult.set(body as CsvResult);
    } catch (error) {
      this.csvError.set(error instanceof Error ? error.message : 'CSV analysis failed.');
    } finally {
      this.csvBusy.set(false);
    }
  }

  toggleComparisonModel(id: string, checked: boolean): void {
    const current = this.selectedComparisonModels();
    if (checked && !current.includes(id)) {
      if (current.length >= 3) return;
      this.selectedComparisonModels.set([...current, id]);
    } else if (!checked) {
      this.selectedComparisonModels.set(current.filter((item) => item !== id));
    }
    this.comparisonResults.set(null);
    this.comparisonError.set('');
  }

  async compareModels(): Promise<void> {
    const file = this.selectedCsv();
    if (!file) { this.comparisonError.set('Choose a CSV file or load a sample first.'); return; }
    const selected = this.selectedComparisonModels();
    const models = this.inferenceModels().filter((model) => selected.includes(model.id));
    if (models.length < 2) { this.comparisonError.set('Select at least two models to compare.'); return; }
    if (models.length > 3) { this.comparisonError.set('Select no more than three models.'); return; }
    this.comparisonBusy.set(true);
    this.comparisonError.set('');
    this.comparisonResults.set(null);
    try {
      const rows = await Promise.all(models.map(async (model): Promise<ComparisonRow> => {
        try {
          const form = new FormData();
          form.append('file', file, file.name);
          form.append('model', model.id);
          form.append('input_format', 'raw');
          const response = await fetch(`${API_URL}/api/predict/csv`, { method: 'POST', body: form });
          const body = await response.json();
          if (!response.ok) throw new Error(body.error ?? `Request failed (${response.status})`);
          return {
            id: model.id,
            name: model.name,
            total_records: body.total_records,
            normal_count: body.normal_count,
            attack_count: body.attack_count,
            metrics: body.metrics,
          };
        } catch (error) {
          return { id: model.id, name: model.name, error: error instanceof Error ? error.message : 'Model comparison failed.' };
        }
      }));
      this.comparisonResults.set(rows);
      if (rows.every((row) => row.error)) this.comparisonError.set('All model requests failed. Check the CSV columns and backend logs.');
    } catch (error) {
      this.comparisonError.set(error instanceof Error ? error.message : 'Could not compare models.');
    } finally {
      this.comparisonBusy.set(false);
    }
  }

  downloadCsvResults(): void {
    const result = this.csvResult();
    if (!result?.records?.length) return;
    const columns = Object.keys(result.records[0]);
    const escape = (value: unknown) => {
      const raw = value === null || value === undefined ? '' : typeof value === 'object' ? JSON.stringify(value) : String(value);
      return /[",\r\n]/.test(raw) ? `"${raw.replace(/"/g, '""')}"` : raw;
    };
    const csv = [columns.join(','), ...result.records.map((row) => columns.map((column) => escape((row as unknown as Record<string, unknown>)[column])).join(','))].join('\r\n');
    this.downloadBlob(csv, 'nsl-kdd-predictions.csv', 'text/csv;charset=utf-8');
  }

  private downloadBlob(content: BlobPart, filename: string, type: string): void {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  formatPercent(value: number | undefined): string {
    return typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : '—';
  }

  maxProbability(probabilities: { Normal: number; Attack: number }): number {
    return Math.max(probabilities.Normal, probabilities.Attack);
  }

  displayedLabel(value: string | undefined): string {
    return value ?? 'Not provided';
  }

  hasComparisonMetrics(rows: ComparisonRow[]): boolean {
    return rows.some((row) => !!row.metrics);
  }
}
