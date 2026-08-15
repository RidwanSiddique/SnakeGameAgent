'use client';

import { useEffect, useState } from 'react';

/**
 * Whether the visitor has asked for reduced motion.
 *
 * The home page otherwise animates five boards at once, which is a lot of
 * movement to put in front of someone who has said they do not want it. When
 * this is true the boards render their layout and stand still, so the page
 * still shows what each level looks like.
 */
export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    const query = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReduced(query.matches);

    const onChange = (event: MediaQueryListEvent) => setReduced(event.matches);
    query.addEventListener('change', onChange);
    return () => query.removeEventListener('change', onChange);
  }, []);

  return reduced;
}
