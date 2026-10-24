//! "Something happens to a team" — parsed from the command line.
//!
//! Spec grammar (append `@ko` to apply the event only from the Round of 32):
//!   shock:TEAM:DELTA        Elo shock, e.g.  shock:France:-100
//!   withdraw:TEAM           team forfeits every match it would play
//!   withdraw:TEAM=OTHER     OTHER takes the team's place before the group stage

#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    Withdraw { replacement: Option<String> },
    EloShock(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct TeamEvent {
    pub team: String,
    pub kind: Kind,
    pub knockout_only: bool,
}

pub fn parse_event(spec: &str) -> Result<TeamEvent, String> {
    let (body, knockout_only) = match spec.trim().strip_suffix("@ko") {
        Some(b) => (b, true),
        None => (spec.trim(), false),
    };
    let parts: Vec<&str> = body.splitn(3, ':').collect();
    match parts.as_slice() {
        ["withdraw", rest] => {
            let (team, replacement) = match rest.split_once('=') {
                Some((t, r)) => (t.trim(), Some(r.trim().to_string())),
                None => (rest.trim(), None),
            };
            if team.is_empty() || replacement.as_deref() == Some("") {
                return Err(format!("bad event '{spec}': empty team name"));
            }
            Ok(TeamEvent {
                team: team.to_string(),
                kind: Kind::Withdraw { replacement },
                knockout_only,
            })
        }
        ["shock", team, delta] => {
            let d: f64 = delta
                .trim()
                .parse()
                .map_err(|_| format!("bad event '{spec}': '{delta}' is not a number"))?;
            Ok(TeamEvent {
                team: team.trim().to_string(),
                kind: Kind::EloShock(d),
                knockout_only,
            })
        }
        _ => Err(format!(
            "bad event '{spec}'. Use shock:TEAM:DELTA or withdraw:TEAM[=REPLACEMENT], optionally ending in @ko"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_forms() {
        let e = parse_event("shock:France:-100").unwrap();
        assert_eq!(e.kind, Kind::EloShock(-100.0));
        assert!(!e.knockout_only);

        let e = parse_event("withdraw:Brazil@ko").unwrap();
        assert_eq!(e.kind, Kind::Withdraw { replacement: None });
        assert!(e.knockout_only);

        let e = parse_event("withdraw:Iraq=Italy").unwrap();
        assert_eq!(e.team, "Iraq");
        assert_eq!(e.kind, Kind::Withdraw { replacement: Some("Italy".into()) });
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_event("explode:France").is_err());
        assert!(parse_event("shock:France:lots").is_err());
        assert!(parse_event("withdraw:").is_err());
    }
}
