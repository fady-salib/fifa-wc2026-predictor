mod data;
mod events;
mod model;
mod rng;
mod sim;

use events::{parse_event, Kind, TeamEvent};
use model::Model;
use sim::{Scenario, STAGES};
use std::process::exit;

const HELP: &str = "\
wc2026 — FIFA World Cup 2026 Monte Carlo simulator with team-event what-ifs

USAGE:
    wc2026 [OPTIONS]

OPTIONS:
    --sims N            number of simulated tournaments        [default: 10000]
    --seed N            RNG seed                               [default: 42]
    --top N             rows to print                          [default: 15]
    --counts FILE       write unrounded per-team stage counts as CSV
    --probs FILE        wc2026_prob_matrix.csv exported by the notebook
                        (default: Elo-only probabilities)
    --event SPEC        team event, repeatable (see below)
    --elo \"TEAM=RATING\" Elo for a replacement team, repeatable
    -h, --help          show this help

EVENTS:
    shock:TEAM:DELTA       Elo shock, e.g. shock:France:-100  (injury, turmoil)
    withdraw:TEAM          team forfeits all its matches
    withdraw:TEAM=OTHER    OTHER takes its place (needs --elo \"OTHER=1850\")
    append @ko to apply an event only from the Round of 32, e.g. shock:Brazil:-60@ko

EXAMPLE:
    wc2026 --sims 20000 --event shock:France:-100 --event withdraw:Iraq=Italy --elo \"Italy=1880\"
";

struct Args {
    sims: u32,
    seed: u64,
    top: usize,
    probs: Option<String>,
    events: Vec<String>,
    elos: Vec<String>,
    counts: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        sims: 10_000,
        seed: 42,
        top: 15,
        probs: None,
        events: vec![],
        elos: vec![],
        counts: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag == "-h" || flag == "--help" {
            print!("{HELP}");
            exit(0);
        }
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--sims" => {
                a.sims = value()?
                    .parse()
                    .map_err(|_| "--sims must be a whole number")?
            }
            "--seed" => {
                a.seed = value()?
                    .parse()
                    .map_err(|_| "--seed must be a whole number")?
            }
            "--top" => {
                a.top = value()?
                    .parse()
                    .map_err(|_| "--top must be a whole number")?
            }
            "--probs" => a.probs = Some(value()?),
            "--counts" => a.counts = Some(value()?),
            "--event" => a.events.push(value()?),
            "--elo" => a.elos.push(value()?),
            other => return Err(format!("unknown option '{other}' (try --help)")),
        }
    }
    if a.sims == 0 {
        return Err("--sims must be at least 1".into());
    }
    Ok(a)
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    eprintln!("Toy tournament: simplified group tiebreaks and Round-of-32 mapping, not official FIFA bracket odds.");
    let mut model = Model::new();

    if let Some(path) = &args.probs {
        let n = model.load_matrix(path)?;
        println!("Loaded {n} pairwise probabilities from {path}");
    } else {
        println!("No --probs file given: using Elo-only probabilities (export the notebook's ML matrix for the full model).");
    }
    for spec in &args.elos {
        let (team, rating) = spec
            .split_once('=')
            .ok_or_else(|| format!("bad --elo '{spec}', expected TEAM=RATING"))?;
        let rating: f64 = rating
            .trim()
            .parse()
            .map_err(|_| format!("bad Elo rating in '{spec}'"))?;
        model.set_elo(team.trim(), rating)?;
    }

    // Resolve team names case-insensitively so `shock:france:-100` works.
    let mut events: Vec<TeamEvent> = Vec::new();
    for spec in &args.events {
        let mut ev = parse_event(spec)?;
        if let Some(c) = model.canonical_name(&ev.team) {
            ev.team = c;
        }
        if let Kind::Withdraw {
            replacement: Some(r),
        } = &mut ev.kind
        {
            if let Some(c) = model.canonical_name(r) {
                *r = c;
            }
        }
        events.push(ev);
    }

    let baseline = Scenario::new(&model, &[])?;
    let scenario = Scenario::new(&model, &events)?;
    let base_counts = baseline.run(args.sims, args.seed);
    let counts = scenario.run(args.sims, args.seed);
    if let Some(path) = &args.counts {
        let mut text = format!("team,{}\n", STAGES.join(","));
        for (team, row) in scenario.teams.iter().zip(&counts) {
            text.push_str(&format!(
                "{},{}\n",
                team,
                row.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
            ));
        }
        std::fs::write(path, text).map_err(|e| format!("cannot write counts: {e}"))?;
    }

    let base_champ: std::collections::HashMap<&str, f64> = baseline
        .teams
        .iter()
        .zip(&base_counts)
        .map(|(t, c)| (t.as_str(), pct(c[5], args.sims)))
        .collect();

    let mut order: Vec<usize> = (0..scenario.teams.len()).collect();
    order.sort_by(|&a, &b| {
        counts[b][5]
            .cmp(&counts[a][5])
            .then(counts[b][4].cmp(&counts[a][4]))
            .then(counts[b][3].cmp(&counts[a][3]))
    });

    println!(
        "\n{} simulations, seed {}{}",
        args.sims,
        args.seed,
        if events.is_empty() {
            String::new()
        } else {
            format!(", {} event(s) applied", events.len())
        }
    );
    print!("{:<22}", "Team");
    for s in STAGES.iter().rev() {
        print!("{:>10}", format!("{s}%"));
    }
    if !events.is_empty() {
        print!("{:>14}", "Δ Champ (pp)");
    }
    println!();
    for &i in order.iter().take(args.top) {
        let team = &scenario.teams[i];
        print!("{:<22}", team);
        for col in (0..6).rev() {
            print!("{:>10.1}", pct(counts[i][col], args.sims));
        }
        if !events.is_empty() {
            let b = base_champ.get(team.as_str()).copied().unwrap_or(0.0);
            print!("{:>+14.1}", pct(counts[i][5], args.sims) - b);
        }
        println!();
    }
    Ok(())
}

fn pct(n: u32, sims: u32) -> f64 {
    n as f64 / sims as f64 * 100.0
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        exit(1);
    }
}
