/**
 * Binary min-heap over cell indices keyed by an f64 priority.
 *
 * Flat typed arrays rather than objects: the priority-flood in lem.ts pushes
 * on the order of the whole grid, and an object per entry turns a two-second
 * bake into a garbage-collection benchmark.
 */
export class MinHeap {
  private key: Float64Array;
  private val: Int32Array;
  private n = 0;

  constructor(capacity: number) {
    this.key = new Float64Array(capacity);
    this.val = new Int32Array(capacity);
  }

  get size(): number {
    return this.n;
  }

  private grow(): void {
    const k = new Float64Array(this.key.length * 2);
    const v = new Int32Array(this.val.length * 2);
    k.set(this.key);
    v.set(this.val);
    this.key = k;
    this.val = v;
  }

  push(key: number, val: number): void {
    if (this.n === this.key.length) this.grow();
    let i = this.n++;
    this.key[i] = key;
    this.val[i] = val;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.key[p] <= this.key[i]) break;
      this.swap(p, i);
      i = p;
    }
  }

  /** Pops the minimum and returns its value; -1 when empty. */
  pop(): number {
    if (this.n === 0) return -1;
    const top = this.val[0];
    this.n--;
    if (this.n > 0) {
      this.key[0] = this.key[this.n];
      this.val[0] = this.val[this.n];
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < this.n && this.key[l] < this.key[m]) m = l;
        if (r < this.n && this.key[r] < this.key[m]) m = r;
        if (m === i) break;
        this.swap(m, i);
        i = m;
      }
    }
    return top;
  }

  /** Priority of the minimum. Only valid when size > 0. */
  peekKey(): number {
    return this.key[0];
  }

  private swap(a: number, b: number): void {
    const k = this.key[a];
    this.key[a] = this.key[b];
    this.key[b] = k;
    const v = this.val[a];
    this.val[a] = this.val[b];
    this.val[b] = v;
  }
}
