//! Tournament simulator: 12 groups -> best 8 thirds -> Round of 32 -> Final, with team events.
//! Mirrors `run_simulations` in the notebook (section 19).

use crate::data::GROUPS;
use crate::events::{Kind, TeamEvent};
use crate::model::{tilt, Model};
use crate::rng::Rng;
use std::collections::{HashMap, HashSet};

pub const STAGES: [&str; 6] = ["R32", "R16", "QF", "SF", "Final", "Champion"];

const R32_PAIRS: [(usize, usize); 12] = [
    (0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4),
    (6, 7), (7, 6), (8, 9), (9, 8), (10, 11), (11, 10),
];

pub struct Scenario<'a> {
    model: &'a Model,
    groups: Vec<Vec<String>>,
    pub teams: Vec<String>,
    index: HashMap<String, usize>,
    shock_group: HashMap<String, f64>,
    shock_ko: HashMap<String, f64>,
    out_group: HashSet<String>,
    out_ko: HashSet<String>,
}

#[derive(Clone)]
struct Row {
    team: String,
    pts: u32,
    gd: i32,
    gf: i32,
    tie: f64,
}

fn rank_cmp(a: &Row, b: &Row) -> std::cmp::Ordering {
    b.pts
        .cmp(&a.pts)
        .then(b.gd.cmp(&a.gd))
        .then(b.gf.cmp(&a.gf))
        .then(a.tie.partial_cmp(&b.tie).unwrap_or(std::cmp::Ordering::Equal))
}

fn find(groups: &[Vec<String>], name: &str) -> Option<(usize, usize)> {
    for (g, ts) in groups.iter().enumerate() {
        if let Some(i) = ts.iter().position(|t| t == name) {
            return Some((g, i));
        }
    }
    None
}

impl<'a> Scenario<'a> {
    pub fn new(model: &'a Model, events: &[TeamEvent]) -> Result<Self, String> {
        let mut groups: Vec<Vec<String>> = GROUPS
            .iter()
            .map(|(_, ts)| ts.iter().map(|t| t.to_string()).collect())
            .collect();
        let mut shock_group: HashMap<String, f64> = HashMap::new();
        let mut shock_ko: HashMap<String, f64> = HashMap::new();
        let mut out_group = HashSet::new();
        let mut out_ko = HashSet::new();

        for ev in events {
            let (gi, ti) = find(&groups, &ev.team)
                .ok_or_else(|| format!("'{}' is not in the tournament", ev.team))?;
            match &ev.kind {
                Kind::EloShock(d) => {
                    if !ev.knockout_only {
                        *shock_group.entry(ev.team.clone()).or_insert(0.0) += *d;
                    }
                    *shock_ko.entry(ev.team.clone()).or_insert(0.0) += *d;
                }
                Kind::Withdraw { replacement: Some(r) } => {
                    if ev.knockout_only {
                        return Err("a replacement only makes sense before the group stage (drop @ko)".into());
                    }
                    if find(&groups, r).is_some() {
                        return Err(format!("replacement '{r}' is already in the tournament"));
                    }
                    if !model.has_elo(r) {
                        return Err(format!("no Elo for '{r}'; pass --elo \"{r}=<rating>\""));
                    }
                    groups[gi][ti] = r.clone();
                }
                Kind::Withdraw { replacement: None } => {
                    if !ev.knockout_only {
                        out_group.insert(ev.team.clone());
                    }
                    out_ko.insert(ev.team.clone());
                }
            }
        }

        let teams: Vec<String> = groups.iter().flatten().cloned().collect();
        let index = teams.iter().enumerate().map(|(i, t)| (t.clone(), i)).collect();
        Ok(Scenario { model, groups, teams, index, shock_group, shock_ko, out_group, out_ko })
    }

    fn shock(map: &HashMap<String, f64>, t: &str) -> f64 {
        map.get(t).copied().unwrap_or(0.0)
    }

    fn group_match(&self, a: &str, b: &str, rng: &mut Rng) -> (u32, u32) {
        match (self.out_group.contains(a), self.out_group.contains(b)) {
            (true, true) => return (0, 0),
            (true, false) => return (0, 3), // walkover for b
            (false, true) => return (3, 0), // walkover for a
            _ => {}
        }
        let delta = Self::shock(&self.shock_group, a) - Self::shock(&self.shock_group, b);
        let p = tilt(self.model.base_probs(a, b), delta);
        let outcome = rng.pick3([p[2], p[1], p[0]]); // 0 = a wins, 1 = draw, 2 = b wins
        let diff = self.model.elo(a) - self.model.elo(b);
        let mut ga = rng.poisson((1.25 + diff / 900.0).max(0.3));
        let mut gb = rng.poisson((1.25 - diff / 900.0).max(0.3));
        match outcome {
            0 => ga = ga.max(gb + 1),
            2 => gb = gb.max(ga + 1),
            _ => {
                let m = ga.min(gb);
                ga = m;
                gb = m;
            }
        }
        (ga, gb)
    }

    fn knockout(&self, a: &str, b: &str, rng: &mut Rng) -> String {
        match (self.out_ko.contains(a), self.out_ko.contains(b)) {
            (true, false) => return b.to_string(),
            (false, true) => return a.to_string(),
            (true, true) => return if rng.next_f64() < 0.5 { a.to_string() } else { b.to_string() },
            _ => {}
        }
        let delta = Self::shock(&self.shock_ko, a) - Self::shock(&self.shock_ko, b);
        let p = tilt(self.model.base_probs(a, b), delta);
        let pw = p[2] + p[1] * p[2] / (p[0] + p[2] + 1e-9); // draws -> extra time / pens
        if rng.next_f64() < pw { a.to_string() } else { b.to_string() }
    }

    fn simulate_once(&self, rng: &mut Rng, counts: &mut [[u32; 6]]) {
        let mut winners: Vec<String> = Vec::new();
        let mut runners: Vec<String> = Vec::new();
        let mut thirds: Vec<Row> = Vec::new();

        for grp in &self.groups {
            let mut rows: Vec<Row> = grp
                .iter()
                .map(|t| Row { team: t.clone(), pts: 0, gd: 0, gf: 0, tie: rng.next_f64() })
                .collect();
            for i in 0..grp.len() {
                for j in (i + 1)..grp.len() {
                    let (ga, gb) = self.group_match(&grp[i], &grp[j], rng);
                    rows[i].gf += ga as i32;
                    rows[j].gf += gb as i32;
                    rows[i].gd += ga as i32 - gb as i32;
                    rows[j].gd += gb as i32 - ga as i32;
                    if ga > gb {
                        rows[i].pts += 3;
                    } else if gb > ga {
                        rows[j].pts += 3;
                    } else {
                        rows[i].pts += 1;
                        rows[j].pts += 1;
                    }
                }
            }
            rows.sort_by(rank_cmp);
            winners.push(rows[0].team.clone());
            runners.push(rows[1].team.clone());
            thirds.push(rows[2].clone());
        }
        thirds.sort_by(rank_cmp);

        let mut r32: Vec<(String, String)> = R32_PAIRS
            .iter()
            .map(|&(w, r)| (winners[w].clone(), runners[r].clone()))
            .collect();
        for i in (0..8).step_by(2) {
            r32.push((thirds[i].team.clone(), thirds[i + 1].team.clone()));
        }
        for (a, b) in &r32 {
            counts[self.index[a]][0] += 1;
            counts[self.index[b]][0] += 1;
        }

        let mut round: Vec<String> = r32.iter().map(|(a, b)| self.knockout(a, b, rng)).collect();
        for stage in 1..=5 {
            for t in &round {
                counts[self.index[t]][stage] += 1;
            }
            if round.len() == 1 {
                break;
            }
            round = round.chunks(2).map(|c| self.knockout(&c[0], &c[1], rng)).collect();
        }
    }

    /// Run `sims` tournaments. Row i of the result belongs to `self.teams[i]`;
    /// columns follow `STAGES` (count of simulations in which the team reached that stage).
    pub fn run(&self, sims: u32, seed: u64) -> Vec<[u32; 6]> {
        let mut rng = Rng::new(seed);
        let mut counts = vec![[0u32; 6]; self.teams.len()];
        for _ in 0..sims {
            self.simulate_once(&mut rng, &mut counts);
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::parse_event;

    fn column_sum(c: &[[u32; 6]], col: usize) -> u32 {
        c.iter().map(|r| r[col]).sum()
    }

    #[test]
    fn stage_totals_are_exact() {
        let m = Model::new();
        let s = Scenario::new(&m, &[]).unwrap();
        let sims = 200;
        let c = s.run(sims, 1);
        assert_eq!(s.teams.len(), 48);
        for (col, per_sim) in [32u32, 16, 8, 4, 2, 1].iter().enumerate() {
            assert_eq!(column_sum(&c, col), per_sim * sims, "stage {}", STAGES[col]);
        }
    }

    #[test]
    fn withdrawn_team_never_advances() {
        let m = Model::new();
        let ev = parse_event("withdraw:Brazil").unwrap();
        let s = Scenario::new(&m, &[ev]).unwrap();
        let c = s.run(300, 2);
        let i = s.teams.iter().position(|t| t == "Brazil").unwrap();
        assert_eq!(c[i], [0; 6]);
    }

    #[test]
    fn knockout_withdrawal_stops_at_the_round_of_32() {
        let m = Model::new();
        let ev = parse_event("withdraw:Argentina@ko").unwrap();
        let s = Scenario::new(&m, &[ev]).unwrap();
        let c = s.run(300, 3);
        let i = s.teams.iter().position(|t| t == "Argentina").unwrap();
        assert_eq!(c[i][1], 0);
        assert!(c[i][0] > 250);
    }

    #[test]
    fn replacement_takes_the_slot() {
        let mut m = Model::new();
        m.set_elo("Italy", 1900.0);
        let ev = parse_event("withdraw:Iraq=Italy").unwrap();
        let s = Scenario::new(&m, &[ev]).unwrap();
        assert!(s.teams.iter().any(|t| t == "Italy"));
        assert!(!s.teams.iter().any(|t| t == "Iraq"));
    }

    #[test]
    fn bad_events_are_rejected() {
        let m = Model::new();
        for spec in ["shock:Narnia:-50", "withdraw:Iraq=Spain", "withdraw:Iraq=Italy"] {
            let ev = parse_event(spec).unwrap();
            assert!(Scenario::new(&m, &[ev]).is_err(), "{spec}");
        }
    }

    #[test]
    fn negative_shock_hurts_the_favourite() {
        let m = Model::new();
        let base = Scenario::new(&m, &[]).unwrap();
        let hit = Scenario::new(&m, &[parse_event("shock:Argentina:-300").unwrap()]).unwrap();
        let (cb, ch) = (base.run(3000, 5), hit.run(3000, 5));
        let i = base.teams.iter().position(|t| t == "Argentina").unwrap();
        assert!(ch[i][5] < cb[i][5]);
    }
}
