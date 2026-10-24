//! Match-outcome model: optional pairwise probabilities exported from the notebook,
//! with an Elo-based fallback, plus the Elo "tilt" used by team events.

use crate::data::{DEFAULT_ELO, ELO};
use std::collections::HashMap;
use std::fs;

/// `[P(team A loses), P(draw), P(team A wins)]`
pub type Probs = [f64; 3];

pub struct Model {
    elo: HashMap<String, f64>,
    matrix: HashMap<String, HashMap<String, Probs>>,
}

impl Model {
    pub fn new() -> Self {
        let elo = ELO.iter().map(|(t, e)| (t.to_string(), *e)).collect();
        Model { elo, matrix: HashMap::new() }
    }

    pub fn elo(&self, team: &str) -> f64 {
        self.elo.get(team).copied().unwrap_or(DEFAULT_ELO)
    }

    pub fn has_elo(&self, team: &str) -> bool {
        self.elo.contains_key(team)
    }

    pub fn set_elo(&mut self, team: &str, rating: f64) {
        self.elo.insert(team.to_string(), rating);
    }

    /// Case-insensitive lookup of a team name known to the model.
    pub fn canonical_name(&self, name: &str) -> Option<String> {
        let wanted = name.trim().to_lowercase();
        self.elo.keys().find(|k| k.to_lowercase() == wanted).cloned()
    }

    /// Load `team_a,team_b,p_loss,p_draw,p_win` rows (written by the notebook's export cell).
    /// Returns the number of rows loaded.
    pub fn load_matrix(&mut self, path: &str) -> Result<usize, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        let mut n = 0;
        for (i, line) in text.lines().enumerate().skip(1) {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split(',').collect();
            if f.len() != 5 {
                return Err(format!("{path}:{}: expected 5 columns, got {}", i + 1, f.len()));
            }
            let num = |s: &str| -> Result<f64, String> {
                s.trim().parse::<f64>().map_err(|_| format!("{path}:{}: bad number '{s}'", i + 1))
            };
            let p = normalize([num(f[2])?, num(f[3])?, num(f[4])?]);
            self.matrix
                .entry(f[0].trim().to_string())
                .or_default()
                .insert(f[1].trim().to_string(), p);
            n += 1;
        }
        Ok(n)
    }

    /// Base probabilities for `a` vs `b` (before any event adjustments).
    pub fn base_probs(&self, a: &str, b: &str) -> Probs {
        if let Some(p) = self.matrix.get(a).and_then(|m| m.get(b)) {
            return *p;
        }
        elo_probs(self.elo(a) - self.elo(b))
    }
}

fn normalize(p: Probs) -> Probs {
    let s: f64 = p.iter().sum();
    if s <= 0.0 {
        return [1.0 / 3.0; 3];
    }
    [p[0] / s, p[1] / s, p[2] / s]
}

/// Build `[loss, draw, win]` from an expected score `e` (= P(win) + P(draw)/2) and a draw probability.
fn from_expected(e: f64, p_draw: f64) -> Probs {
    let p_win = (e - 0.5 * p_draw).max(0.0);
    let p_loss = ((1.0 - e) - 0.5 * p_draw).max(0.0);
    normalize([p_loss, p_draw, p_win])
}

/// Elo-only fallback: logistic expected score, with fewer draws between mismatched teams.
pub fn elo_probs(diff: f64) -> Probs {
    let e = 1.0 / (1.0 + 10f64.powf(-diff / 400.0));
    let p_draw = 0.10 + 0.14 * (1.0 - (2.0 * e - 1.0).powi(2));
    from_expected(e, p_draw)
}

/// Shift team A's strength by `delta` Elo points, keeping P(draw) fixed.
/// Positive delta helps A, negative hurts A.
pub fn tilt(p: Probs, delta: f64) -> Probs {
    if delta == 0.0 {
        return p;
    }
    let e = (p[2] + 0.5 * p[1]).clamp(1e-6, 1.0 - 1e-6);
    let x = 400.0 * (e / (1.0 - e)).log10() + delta;
    let e2 = 1.0 / (1.0 + 10f64.powf(-x / 400.0));
    from_expected(e2, p[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sums_to_one(p: Probs) -> bool {
        (p.iter().sum::<f64>() - 1.0).abs() < 1e-9 && p.iter().all(|x| *x >= 0.0)
    }

    #[test]
    fn elo_probs_are_valid_and_monotone() {
        let mut last = 0.0;
        for d in (-800..=800).step_by(100) {
            let p = elo_probs(d as f64);
            assert!(sums_to_one(p));
            assert!(p[2] >= last);
            last = p[2];
        }
        let even = elo_probs(0.0);
        assert!((even[0] - even[2]).abs() < 1e-9);
    }

    #[test]
    fn tilt_moves_win_probability_the_right_way() {
        let p = [0.2, 0.25, 0.55];
        let worse = tilt(p, -100.0);
        let better = tilt(p, 100.0);
        assert!(sums_to_one(worse) && sums_to_one(better));
        assert!(worse[2] < p[2] && better[2] > p[2]);
        assert!((worse[1] - p[1]).abs() < 1e-9);
        assert_eq!(tilt(p, 0.0), p);
    }
}
