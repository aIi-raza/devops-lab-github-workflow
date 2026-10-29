import { Directive, ElementRef, OnDestroy, OnInit, inject } from '@angular/core';

/** Fades and lifts an element in the first time it scrolls into view. */
@Directive({ selector: '[appReveal]' })
export class Reveal implements OnInit, OnDestroy {
  private readonly el = inject<ElementRef<HTMLElement>>(ElementRef);
  private io?: IntersectionObserver;

  ngOnInit(): void {
    const node = this.el.nativeElement;
    node.classList.add('reveal');
    const reduce = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce || typeof IntersectionObserver === 'undefined') { node.classList.add('in'); return; }
    this.io = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) { node.classList.add('in'); this.io?.disconnect(); }
    }, { threshold: 0.1, rootMargin: '0px 0px -5% 0px' });
    this.io.observe(node);
  }

  ngOnDestroy(): void { this.io?.disconnect(); }
}
