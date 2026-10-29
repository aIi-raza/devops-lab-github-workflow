import { AfterViewInit, Component, ElementRef, NgZone, OnDestroy, ViewChild, inject } from '@angular/core';
import createGlobe, { type Globe as CobeGlobe } from 'cobe';

type RGB = [number, number, number];
type Vec = [number, number, number];

// Cities [lat, lon]: Islamabad, London, New York, Singapore, Seoul, São Paulo, Nairobi, Sydney, Frankfurt.
// They are only used as start/end points for the random traffic lines (no markers are drawn).
const CITIES: [number, number][] = [[33.7, 73.1], [51.5, -0.1], [40.7, -74], [1.35, 103.8], [37.5, 127], [-23.5, -46.6], [-1.3, 36.8], [-33.9, 151.2], [50.1, 8.7]];

// Same lat/lon -> unit vector conversion cobe uses internally, so lines sit exactly on the dots.
const toVec = (lat: number, lon: number): Vec => {
  const r = (lat * Math.PI) / 180, a = (lon * Math.PI) / 180 - Math.PI, c = Math.cos(r);
  return [-c * Math.cos(a), Math.sin(r), c * Math.sin(a)];
};
const NODES: Vec[] = CITIES.map(([a, b]) => toVec(a, b));
const SPHERE = 0.8;   // cobe draws the sphere at 80% of the canvas half-size

// Per-theme colors, tuned so the sphere matches the page background (light #F5F6F5, dark #242628).
// cobe math: dark mode paints the sphere at baseColor x 0.1, so baseColor is deliberately above 1 there.
// If the dots look too strong or too faint, change `mapBrightness` below.
const THEMES: Record<'light' | 'dark', {
  dark: number; diffuse: number; mapBrightness: number; base: RGB; glow: RGB; marker: RGB; line: string;
}> = {
  light: { dark: 0, diffuse: 1.2, mapBrightness: 0.9, base: [0.87, 0.88, 0.87], glow: [0.96, 0.965, 0.96], marker: [0.18, 0.42, 0.31], line: '47,107,80' },
  dark: { dark: 1, diffuse: 1.2, mapBrightness: 0.55, base: [1.41, 1.49, 1.57], glow: [0.1, 0.105, 0.11], marker: [0.6, 0.82, 0.69], line: '154,208,177' },
};

type Comet = { a: Vec; b: Vec; start: number; dur: number; lift: number; om: number };

/**
 * Dotted WebGL globe (cobe) plus a thin overlay canvas that draws fast, random traffic lines.
 * Rotates slowly; the pointer adds a small tilt/speed change. Colors follow the
 * `data-theme` attribute of the closest `.app-shell`.
 */
@Component({
  selector: 'app-globe',
  template: '<div class="wrap"><canvas #cv aria-hidden="true"></canvas><canvas #ov class="ov" aria-hidden="true"></canvas></div>',
  // Note: no `position` on :host, so the page's `.hero-globe { position: absolute }` rule stays in control.
  styles: [':host{display:block;pointer-events:none}.wrap{position:relative;width:100%;height:100%}canvas{display:block;width:100%;height:100%;opacity:0;transition:opacity 1.2s ease}canvas.ready{opacity:1}.ov{position:absolute;inset:0}'],
})
export class Globe implements AfterViewInit, OnDestroy {
  @ViewChild('cv', { static: true }) private cv!: ElementRef<HTMLCanvasElement>;
  @ViewChild('ov', { static: true }) private ov!: ElementRef<HTMLCanvasElement>;
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly zone = inject(NgZone);
  private globe?: CobeGlobe;
  private raf = 0;
  private visible = true;
  private size = 0; private dpr = 1;
  private phi = 3.6;           // starts with Asia/Pakistan roughly facing the viewer
  private theta = 0.25;
  private px = 0; private py = 0;
  private theme: 'light' | 'dark' | '' = '';
  private frame = 0;
  private reduce = false;
  private comets: Comet[] = [];
  private nextSpawn = 0;
  private ro?: ResizeObserver; private io?: IntersectionObserver;
  private readonly onMove = (e: PointerEvent) => {
    this.px = e.clientX / window.innerWidth - 0.5;
    this.py = e.clientY / window.innerHeight - 0.5;
  };

  ngAfterViewInit(): void {
    this.reduce = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.zone.runOutsideAngular(() => {
      this.ro = new ResizeObserver(() => this.resize());
      this.ro.observe(this.host.nativeElement);
      this.io = new IntersectionObserver(([e]) => { this.visible = e.isIntersecting; });
      this.io.observe(this.host.nativeElement);
      this.resize();
      if (!this.reduce) window.addEventListener('pointermove', this.onMove, { passive: true });
      const loop = () => {
        if (this.globe && this.visible && !document.hidden) this.tick();
        this.raf = requestAnimationFrame(loop);
      };
      this.raf = requestAnimationFrame(loop);
    });
  }

  ngOnDestroy(): void {
    cancelAnimationFrame(this.raf);
    this.ro?.disconnect(); this.io?.disconnect();
    window.removeEventListener('pointermove', this.onMove);
    this.globe?.destroy();
  }

  private currentTheme(): 'light' | 'dark' {
    const shell = this.host.nativeElement.closest('[data-theme]');
    return shell?.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
  }

  private themeOptions(t: 'light' | 'dark') {
    const c = THEMES[t];
    return { dark: c.dark, diffuse: c.diffuse, mapBrightness: c.mapBrightness, baseColor: c.base, glowColor: c.glow, markerColor: c.marker };
  }

  private resize(): void {
    const r = this.host.nativeElement.getBoundingClientRect();
    const size = Math.round(Math.min(r.width, r.height));
    if (size < 10 || size === this.size) return;
    this.size = size;
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    const ov = this.ov.nativeElement;
    ov.width = ov.height = Math.round(size * this.dpr);
    // cobe multiplies width/height by devicePixelRatio itself, so pass CSS pixels.
    if (this.globe) { this.globe.update({ width: size, height: size }); return; }
    this.theme = this.currentTheme();
    this.globe = createGlobe(this.cv.nativeElement, {
      devicePixelRatio: this.dpr, width: size, height: size,
      phi: this.phi, theta: this.theta,
      mapSamples: 16000,
      ...this.themeOptions(this.theme),
      markers: [],
    });
    this.cv.nativeElement.classList.add('ready');
    this.ov.nativeElement.classList.add('ready');
  }

  private tick(): void {
    if (!this.reduce) {
      this.phi += 0.0028 + this.px * 0.0016;
      this.theta += (0.25 - this.py * 0.35 - this.theta) * 0.05;
    }
    const state: Record<string, unknown> = { phi: this.phi, theta: this.theta };
    if (this.frame++ % 20 === 0) {
      const t = this.currentTheme();
      if (t !== this.theme) { this.theme = t; Object.assign(state, this.themeOptions(t)); }
    }
    this.globe!.update(state);
    if (!this.reduce) this.drawLines(performance.now());
  }

  /** Projects a unit-sphere vector (scaled by `radius`) to overlay pixels, using cobe's own rotation math. */
  private project(v: Vec, radius: number, cp: number, sp: number, ct: number, st: number, half: number): { x: number; y: number; z: number } {
    const x = v[0] * radius, y = v[1] * radius, z = v[2] * radius;
    const c = cp * x + sp * z;
    const s = sp * st * x + ct * y - cp * st * z;
    const depth = -sp * ct * x + st * y + cp * ct * z;
    return { x: (c + 1) * half, y: (1 - s) * half, z: depth };
  }

  private drawLines(now: number): void {
    const cv = this.ov.nativeElement, ctx = cv.getContext('2d');
    if (!ctx || !this.size) return;
    const d = this.dpr, half = (this.size * d) / 2;
    const cp = Math.cos(this.phi), sp = Math.sin(this.phi), ct = Math.cos(this.theta), st = Math.sin(this.theta);

    // Spawn random, short-lived lines between random cities (prefer ones whose start is facing us).
    if (now > this.nextSpawn && this.comets.length < 5) {
      const front = NODES.map((n, i) => ({ i, z: this.project(n, SPHERE, cp, sp, ct, st, half).z })).filter((n) => n.z > 0.05);
      if (front.length) {
        const ia = front[Math.floor(Math.random() * front.length)].i;
        let ib = Math.floor(Math.random() * NODES.length);
        if (ib === ia) ib = (ib + 1) % NODES.length;
        const A = NODES[ia], B = NODES[ib];
        const om = Math.acos(Math.max(-1, Math.min(1, A[0] * B[0] + A[1] * B[1] + A[2] * B[2])));
        this.comets.push({ a: A, b: B, start: now, dur: 900 + Math.random() * 900, om, lift: 0.04 + 0.1 * (om / Math.PI) });
      }
      this.nextSpawn = now + 150 + Math.random() * 450;
    }

    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.lineCap = 'round';
    const rgb = THEMES[this.theme || 'light'].line;
    const STEPS = 44, TRAIL = 16;
    this.comets = this.comets.filter((c) => now - c.start < c.dur);

    for (const c of this.comets) {
      const p = (now - c.start) / c.dur;                 // 0..1
      const head = p * (STEPS + TRAIL);                   // head runs past the end so the tail clears out
      const so = Math.sin(c.om) || 1;
      const fade = Math.min(1, (1 - p) * 4);
      const pts: { x: number; y: number; z: number }[] = [];
      for (let k = 0; k <= STEPS; k++) {
        const t = k / STEPS, f1 = Math.sin((1 - t) * c.om) / so, f2 = Math.sin(t * c.om) / so;
        const e = 1 + Math.sin(Math.PI * t) * c.lift;
        const v: Vec = [(c.a[0] * f1 + c.b[0] * f2) * e, (c.a[1] * f1 + c.b[1] * f2) * e, (c.a[2] * f1 + c.b[2] * f2) * e];
        pts.push(this.project(v, SPHERE, cp, sp, ct, st, half));
      }
      for (let k = Math.max(0, Math.floor(head - TRAIL)); k < Math.min(STEPS, Math.ceil(head)); k++) {
        const a = pts[k], b = pts[k + 1];
        if (a.z <= 0.02 || b.z <= 0.02) continue;
        const f = 1 - (head - k) / TRAIL;                 // 0 at tail, 1 at head
        ctx.strokeStyle = `rgba(${rgb},${(0.9 * f * f * fade).toFixed(3)})`;
        ctx.lineWidth = 1.1 * d;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
    }
  }
}
