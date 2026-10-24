//! Tiny deterministic PRNG (SplitMix64) so the crate needs no dependencies.

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Poisson sample (Knuth). Fine for the small rates used for football goals.
    pub fn poisson(&mut self, lambda: f64) -> u32 {
        let limit = (-lambda.min(30.0)).exp();
        let mut k = 0u32;
        let mut p = 1.0;
        loop {
            p *= self.next_f64();
            if p <= limit {
                return k;
            }
            k += 1;
        }
    }

    /// Pick index 0..3 with the given (already normalised) weights.
    pub fn pick3(&mut self, w: [f64; 3]) -> usize {
        let r = self.next_f64();
        if r < w[0] {
            0
        } else if r < w[0] + w[1] {
            1
        } else {
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_is_in_range_and_deterministic() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..1000 {
            let x = a.next_f64();
            assert!((0.0..1.0).contains(&x));
            assert_eq!(x, b.next_f64());
        }
    }

    #[test]
    fn poisson_mean_is_close() {
        let mut r = Rng::new(1);
        let n = 20_000;
        let mean: f64 = (0..n).map(|_| r.poisson(1.5) as f64).sum::<f64>() / n as f64;
        assert!((mean - 1.5).abs() < 0.05, "mean was {mean}");
    }
}
