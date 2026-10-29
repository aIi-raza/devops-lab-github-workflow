import { AfterViewInit, Component, ElementRef, NgZone, OnDestroy, ViewChild, inject } from '@angular/core';

const MAGNETIC = '[data-magnetic], button:not(:disabled), a[href], select, summary, label.choice, label.dropzone';
const NATIVE = 'input, textarea';           // keep the real cursor for text entry

/**
 * Magnetic cursor: a blended circle that follows the pointer on a spring, snaps toward
 * interactive elements and morphs to their shape, stretches with speed, and bursts on click.
 * Disabled on touch devices and for reduced motion. No dependencies.
 */
@Component({
  selector: 'app-cursor',
  template: '<div #c class="mc" aria-hidden="true"></div>',
  styles: [`
    .mc { position: fixed; left: 0; top: 0; z-index: 9999; width: 40px; height: 40px; border-radius: 50%; background: #fff;
      mix-blend-mode: exclusion; pointer-events: none; opacity: 0; will-change: transform, width, height;
      transition: opacity .2s ease; }
    .mc.on { opacity: 1; }
    :host { display: contents; }
  `],
})
export class Cursor implements AfterViewInit, OnDestroy {
  @ViewChild('c', { static: true }) private c!: ElementRef<HTMLDivElement>;
  private readonly zone = inject(NgZone);

  // Tunables (same idea as the reference: magneticFactor 0.55, size 40).
  private readonly SIZE = 40;
  private readonly PULL = 0.4;
  private readonly STIFF = 0.3;
  private readonly DAMP = 0.62;

  private enabled = false;
  private raf = 0;
  private mx = -100; private my = -100;         // pointer
  private x = -100; private y = -100;           // cursor center
  private vx = 0; private vy = 0;
  private w = this.SIZE; private h = this.SIZE; // current size
  private radius = this.SIZE / 2;
  private target: HTMLElement | null = null;
  private pressed = false;
  private seen = false;
  private pspeed = 0;                          // smoothed pointer speed (px per event)
  private engage = 0;                          // 0 = free cursor, 1 = locked to element

  private readonly onMove = (e: PointerEvent) => {
    if (e.pointerType === 'touch') return;
    if (this.seen) this.pspeed = this.pspeed * 0.7 + Math.hypot(e.clientX - this.mx, e.clientY - this.my) * 0.3;
    this.mx = e.clientX; this.my = e.clientY;
    const el = e.target instanceof Element ? e.target : null;
    const native = !!el?.closest(NATIVE);
    let t = native ? null : (el?.closest(MAGNETIC) as HTMLElement | null) ?? null;
    if (t) { const r = t.getBoundingClientRect(); if (r.width > 200 || r.height > 64) t = null; }   // big cards/panels are not magnetic
    this.target = t;
    const node = this.c.nativeElement;
    if (!this.seen) { this.x = this.mx; this.y = this.my; this.seen = true; }
    node.classList.toggle('on', !native);
  };
  private readonly onDown = (e: PointerEvent) => {
    if (e.pointerType === 'touch') return;
    this.pressed = true;
    this.burst(e.clientX, e.clientY);
  };
  private readonly onUp = () => { this.pressed = false; };
  private readonly onLeave = () => { this.c.nativeElement.classList.remove('on'); };

  ngAfterViewInit(): void {
    const fine = matchMedia('(hover: hover) and (pointer: fine)').matches;
    const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!fine || reduce) return;
    this.enabled = true;
    document.documentElement.classList.add('cursor-on');
    this.zone.runOutsideAngular(() => {
      document.addEventListener('pointermove', this.onMove, { passive: true });
      document.addEventListener('pointerdown', this.onDown, { passive: true });
      document.addEventListener('pointerup', this.onUp, { passive: true });
      document.documentElement.addEventListener('pointerleave', this.onLeave);
      this.raf = requestAnimationFrame(() => this.tick());
    });
  }

  ngOnDestroy(): void {
    if (!this.enabled) return;
    cancelAnimationFrame(this.raf);
    document.removeEventListener('pointermove', this.onMove);
    document.removeEventListener('pointerdown', this.onDown);
    document.removeEventListener('pointerup', this.onUp);
    document.documentElement.removeEventListener('pointerleave', this.onLeave);
    document.documentElement.classList.remove('cursor-on');
  }

  private tick(): void {
    // 1. Magnetism only engages when the pointer slows down over a small element,
    //    so moving across the page never drags the cursor around.
    this.pspeed *= 0.9;
    const el = this.target;
    const want = el && el.isConnected && this.pspeed < 4 ? 1 : 0;
    this.engage += (want - this.engage) * 0.14;
    const k = this.engage;

    let tx = this.mx, ty = this.my, tw = this.SIZE, th = this.SIZE, tr = this.SIZE / 2;
    if (el && el.isConnected && k > 0.01) {
      const r = el.getBoundingClientRect();
      const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
      tx = this.mx + (cx - this.mx) * this.PULL * k;
      ty = this.my + (cy - this.my) * this.PULL * k;
      tw = this.SIZE + (Math.max(r.width + 10, this.SIZE) - this.SIZE) * k;
      th = this.SIZE + (Math.max(r.height + 10, this.SIZE) - this.SIZE) * k;
      const br = parseFloat(getComputedStyle(el).borderTopLeftRadius) || 0;
      tr = Math.min(this.SIZE / 2 + (Math.min(br + 5, th / 2) - this.SIZE / 2) * k, th / 2);
    }

    // 2. Spring toward target (stiff, so it tracks the real pointer closely).
    this.vx = (this.vx + (tx - this.x) * this.STIFF) * this.DAMP;
    this.vy = (this.vy + (ty - this.y) * this.STIFF) * this.DAMP;
    this.x += this.vx; this.y += this.vy;

    // 3. Ease size and corner radius (the "blob" morph).
    const ease = 0.25;
    this.w += (tw - this.w) * ease;
    this.h += (th - this.h) * ease;
    this.radius += (tr - this.radius) * ease;

    // 4. Stretch along velocity when free, flatten to normal when locked on an element.
    const speed = Math.hypot(this.vx, this.vy);
    const stretch = Math.min(speed / 60, 0.25) * (1 - k);
    const angle = Math.atan2(this.vy, this.vx);
    const press = this.pressed ? 0.82 : 1;

    const n = this.c.nativeElement;
    n.style.width = `${this.w.toFixed(1)}px`;
    n.style.height = `${this.h.toFixed(1)}px`;
    n.style.borderRadius = `${this.radius.toFixed(1)}px`;
    n.style.transform =
      `translate3d(${(this.x - this.w / 2).toFixed(1)}px, ${(this.y - this.h / 2).toFixed(1)}px, 0) ` +
      `rotate(${angle.toFixed(3)}rad) scale(${((1 + stretch) * press).toFixed(3)}, ${((1 - stretch * 0.5) * press).toFixed(3)}) ` +
      `rotate(${(-angle).toFixed(3)}rad)`;

    this.raf = requestAnimationFrame(() => this.tick());
  }

  /** Small particle burst on click. */
  private burst(x: number, y: number): void {
    const count = 10;
    for (let i = 0; i < count; i++) {
      const p = document.createElement('i');
      const a = (Math.PI * 2 * i) / count + Math.random() * 0.5, d = 28 + Math.random() * 34, s = 3 + Math.random() * 3;
      p.style.cssText = `position:fixed;left:${x}px;top:${y}px;width:${s}px;height:${s}px;margin:${-s / 2}px 0 0 ${-s / 2}px;border-radius:50%;background:#fff;mix-blend-mode:exclusion;pointer-events:none;z-index:9998`;
      document.body.appendChild(p);
      p.animate(
        [{ transform: 'translate(0,0) scale(1)', opacity: 1 }, { transform: `translate(${Math.cos(a) * d}px,${Math.sin(a) * d}px) scale(0)`, opacity: 0 }],
        { duration: 520 + Math.random() * 180, easing: 'cubic-bezier(.2,.7,.2,1)' },
      ).onfinish = () => p.remove();
    }
  }
}
