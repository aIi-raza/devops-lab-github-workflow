import { Component, computed, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';

// Where the Flask backend runs. Change this if you host the backend elsewhere.
const API_URL = 'http://localhost:5000';

type Operation = 'add' | 'subtract' | 'multiply' | 'divide';

const SYMBOLS: Record<Operation, string> = {
  add: '+',
  subtract: '−',
  multiply: '×',
  divide: '÷',
};

@Component({
  selector: 'app-root',
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  // Change these for the demo: they show up on the page.
  readonly appTitle = 'CI/CD Calculator';
  readonly version = 'v1';

  display = signal('0');
  expression = signal('');
  error = signal('');
  loading = signal(false);
  pendingOp = signal<Operation | null>(null);
  waitingForSecond = signal(false);

  private firstValue = 0;

  formatted = computed(() => this.format(this.display()));

  displaySize = computed(() => {
    const digits = this.display().replace(/[-.,]/g, '').length;
    if (digits > 9) return 'small';
    if (digits > 7) return 'medium';
    return 'large';
  });

  constructor(private http: HttpClient) {}

  // ---------- Button handlers ----------

  pressDigit(d: string): void {
    if (this.loading()) return;
    this.error.set('');

    if (this.waitingForSecond() || this.display() === '0' || this.display() === 'Error') {
      this.display.set(d);
      this.waitingForSecond.set(false);
    } else if (this.display().replace(/[-.]/g, '').length < 12) {
      this.display.set(this.display() + d);
    }
  }

  pressDecimal(): void {
    if (this.loading()) return;
    this.error.set('');

    if (this.waitingForSecond() || this.display() === 'Error') {
      this.display.set('0.');
      this.waitingForSecond.set(false);
    } else if (!this.display().includes('.')) {
      this.display.set(this.display() + '.');
    }
  }

  pressOperator(op: Operation): void {
    if (this.loading() || this.display() === 'Error') return;
    const current = parseFloat(this.display());
    const previousOp = this.pendingOp();

    if (previousOp && !this.waitingForSecond()) {
      // Chained operation (example: 2 + 3 +): calculate the first part now.
      this.compute(this.firstValue, current, previousOp, (result) => {
        this.firstValue = result;
        this.display.set(this.toText(result));
        this.pendingOp.set(op);
        this.waitingForSecond.set(true);
        this.expression.set(`${this.format(this.toText(result))} ${SYMBOLS[op]}`);
      });
    } else {
      this.firstValue = current;
      this.pendingOp.set(op);
      this.waitingForSecond.set(true);
      this.expression.set(`${this.format(this.display())} ${SYMBOLS[op]}`);
    }
  }

  pressEquals(): void {
    const op = this.pendingOp();
    if (!op || this.loading()) return;

    const a = this.firstValue;
    const b = this.waitingForSecond() ? a : parseFloat(this.display());

    this.compute(a, b, op, (result) => {
      this.expression.set(
        `${this.format(this.toText(a))} ${SYMBOLS[op]} ${this.format(this.toText(b))} =`
      );
      this.display.set(this.toText(result));
      this.pendingOp.set(null);
      this.waitingForSecond.set(true);
    });
  }

  clear(): void {
    this.display.set('0');
    this.expression.set('');
    this.error.set('');
    this.pendingOp.set(null);
    this.waitingForSecond.set(false);
    this.firstValue = 0;
  }

  backspace(): void {
    if (this.loading() || this.waitingForSecond() || this.display() === 'Error') return;
    const next = this.display().slice(0, -1);
    this.display.set(next === '' || next === '-' ? '0' : next);
  }

  toggleSign(): void {
    const value = this.display();
    if (value === 'Error' || value === '0') return;
    this.display.set(value.startsWith('-') ? value.slice(1) : '-' + value);
  }

  percent(): void {
    if (this.display() === 'Error') return;
    this.display.set(this.toText(parseFloat(this.display()) / 100));
  }

  isActive(op: Operation): boolean {
    return this.pendingOp() === op && this.waitingForSecond();
  }

  // ---------- Backend call ----------

  private compute(a: number, b: number, op: Operation, onSuccess: (result: number) => void): void {
    this.loading.set(true);
    this.error.set('');

    this.http.post<{ result: number }>(`${API_URL}/calculate`, { a, b, operation: op }).subscribe({
      next: (res) => {
        onSuccess(res.result);
        this.loading.set(false);
      },
      error: (err) => {
        const message =
          err?.error?.error ?? 'Cannot reach the Flask backend. Start it first (see README).';
        this.display.set('Error');
        this.expression.set('');
        this.pendingOp.set(null);
        this.waitingForSecond.set(true);
        this.error.set(message);
        this.loading.set(false);
      },
    });
  }

  // ---------- Helpers ----------

  private toText(n: number): string {
    return String(Number(n.toPrecision(10)));
  }

  private format(raw: string): string {
    if (raw === 'Error') return raw;
    const negative = raw.startsWith('-');
    const body = negative ? raw.slice(1) : raw;
    const [whole, decimals] = body.split('.');
    const withCommas = whole.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    return (negative ? '-' : '') + withCommas + (decimals !== undefined ? '.' + decimals : '');
  }
}