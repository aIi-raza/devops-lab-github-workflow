import { Directive, ElementRef, Input, OnChanges, OnDestroy, inject } from '@angular/core';

/** Animates a number from 0 to the bound value. */
@Directive({ selector: '[appCountUp]' })
export class CountUp implements OnChanges, OnDestroy {
  @Input({ required: true }) appCountUp = 0;
  private readonly el = inject<ElementRef<HTMLElement>>(ElementRef);
  private raf = 0;

  ngOnChanges(): void {
    cancelAnimationFrame(this.raf);
    const target = Number(this.appCountUp) || 0;
    const node = this.el.nativeElement;
    const reduce = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce || target === 0) { node.textContent = target.toLocaleString(); return; }
    const start = performance.now();
    const duration = 900;
    const tick = (now: number) => {
      const p = Math.min(1, (now - start) / duration);
      node.textContent = Math.round(target * (1 - Math.pow(1 - p, 3))).toLocaleString();
      if (p < 1) this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  ngOnDestroy(): void { cancelAnimationFrame(this.raf); }
}
