import { Component, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';

// Where the Flask backend runs. Change this if you host the backend elsewhere.
const API_URL = 'http://localhost:5000';

@Component({
  selector: 'app-root',
  imports: [FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  // Change these for the demo: they show up on the page.
  readonly appTitle = 'CI/CD Calculator';
  readonly version = 'v1';

  a: number | null = null;
  b: number | null = null;
  operation = 'add';

  result = signal<number | null>(null);
  error = signal('');
  loading = signal(false);

  constructor(private http: HttpClient) {}

  calculate(): void {
    this.error.set('');
    this.result.set(null);

    if (this.a === null || this.b === null) {
      this.error.set('Enter both numbers.');
      return;
    }

    this.loading.set(true);
    this.http
      .post<{ result: number }>(`${API_URL}/calculate`, {
        a: this.a,
        b: this.b,
        operation: this.operation,
      })
      .subscribe({
        next: (res) => {
          this.result.set(res.result);
          this.loading.set(false);
        },
        error: (err) => {
          this.error.set(
            err?.error?.error ?? 'Cannot reach the Flask backend. Start it first (see README).'
          );
          this.loading.set(false);
        },
      });
  }
}
