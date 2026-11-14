//! Which IO builtins reach the Program AST through the mutable handle
//! (`shared_state::get_shared_program_mut`). This table is the ONE classification:
//!
//! * the evaluator installs a private Program wrapper on demand for [`Edits`] builtins;
//! * a native `run` block (#129) cannot change the program: a direct call to a builtin that
//!   [`changes_program`] is refused at admission, and the block's frozen coroutine handle
//!   refuses the same builtins at dispatch when a called function or spawned task reaches
//!   them (`coroutine/program_freeze.rs`).
//!
//! A new builtin that takes the mutable handle must be added here; the source-scan test
//! below fails until it is, and fails when a writer is classified as a reader.
//!
//! [`Edits`]: ProgramAccess::Edits

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgramAccess {
    /// Adds, replaces, moves or removes declarations in place.
    Edits,
    /// Replaces module declarations from the library on disk. Unlike [`Self::Edits`] it
    /// requires an already installed Program handle (the evaluator creates none for it).
    Reloads,
    /// Takes the mutable handle but only reads it.
    Reads,
}

const PROGRAM_ACCESS: &[(&str, ProgramAccess)] = &[
    ("mutate", ProgramAccess::Edits),
    ("fn_remove", ProgramAccess::Edits),
    ("type_remove", ProgramAccess::Edits),
    ("module_remove", ProgramAccess::Edits),
    ("module_create", ProgramAccess::Edits),
    ("fn_replace", ProgramAccess::Edits),
    ("move_symbols", ProgramAccess::Edits),
    ("library_reload", ProgramAccess::Reloads),
    ("run_module_startups", ProgramAccess::Reads),
];

/// Classification of a builtin spelling (name or registered alias).
pub fn program_access(name: &str) -> Option<ProgramAccess> {
    let canonical = super::IO_BUILTINS
        .iter()
        .find(|builtin| builtin.name == name || builtin.aliases.contains(&name))?
        .name;
    PROGRAM_ACCESS
        .iter()
        .find(|(entry, _)| *entry == canonical)
        .map(|(_, access)| *access)
}

/// True when awaiting this builtin can change declarations of the Program it runs on.
pub fn changes_program(name: &str) -> bool {
    matches!(
        program_access(name),
        Some(ProgramAccess::Edits | ProgramAccess::Reloads)
    )
}

/// The builtins [`changes_program`] is true for, in table order (documentation and
/// refusal texts).
pub fn program_changing_builtins() -> Vec<&'static str> {
    PROGRAM_ACCESS
        .iter()
        .filter(|(_, access)| !matches!(access, ProgramAccess::Reads))
        .map(|(name, _)| *name)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (builtin spellings of one dispatch arm, whether it takes a write lock)
    fn arms_taking_the_mutable_program(source: &str) -> Vec<(Vec<String>, bool)> {
        let mut arms: Vec<(Vec<String>, String)> = Vec::new();
        for line in source.lines() {
            let trimmed = line.trim_start();
            let indent = line.len() - trimmed.len();
            if indent == 12 && trimmed.starts_with('"') && trimmed.ends_with("=> {") {
                let names = trimmed
                    .trim_end_matches("=> {")
                    .split('|')
                    .map(|name| name.trim().trim_matches('"').to_string())
                    .collect();
                arms.push((names, String::new()));
            } else if let Some((_, body)) = arms.last_mut() {
                body.push_str(line);
                body.push('\n');
            }
        }
        arms.into_iter()
            .filter(|(_, body)| body.contains("get_shared_program_mut()"))
            .map(|(names, body)| (names, body.contains("program_lock.write()")))
            .collect()
    }

    #[test]
    fn every_builtin_taking_the_mutable_program_is_classified() {
        let source = include_str!("../coroutine/mod.rs");
        let arms = arms_taking_the_mutable_program(source);
        assert_eq!(
            arms.len(),
            source.matches("get_shared_program_mut()").count(),
            "a get_shared_program_mut() call sits outside a recognizable builtin dispatch arm; \
             extend this scan and classify the builtin in PROGRAM_ACCESS"
        );
        assert!(arms.len() >= 9, "the scan found too few arms: {arms:?}");
        for (names, writes) in &arms {
            for name in names {
                let access = program_access(name).unwrap_or_else(|| {
                    panic!("builtin `{name}` takes the mutable Program but is not classified in PROGRAM_ACCESS")
                });
                assert_eq!(
                    *writes,
                    changes_program(name),
                    "builtin `{name}` is classified {access:?} but its dispatch arm {} a write lock",
                    if *writes { "takes" } else { "does not take" }
                );
            }
        }
        let scanned: Vec<&str> = arms.iter().flat_map(|(names, _)| names.iter().map(String::as_str)).collect();
        for (name, _) in PROGRAM_ACCESS {
            assert!(scanned.contains(name), "`{name}` is classified but no dispatch arm takes the mutable Program");
        }
    }

    /// The handle must not be taken anywhere the scan above cannot see.
    #[test]
    fn the_mutable_program_handle_is_taken_only_in_the_scanned_dispatch() {
        fn visit(dir: &std::path::Path, found: &mut Vec<String>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    visit(&path, found);
                } else if path.extension().is_some_and(|ext| ext == "rs")
                    && std::fs::read_to_string(&path).unwrap().contains("get_shared_program_mut()")
                {
                    found.push(path.strip_prefix(env!("CARGO_MANIFEST_DIR")).unwrap().display().to_string());
                }
            }
        }
        let mut found = Vec::new();
        visit(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut found);
        found.sort();
        let allowed = [
            "src/builtins/program_effects.rs", // this scan
            "src/coroutine/mod.rs",            // the scanned builtin dispatch
            "src/eval/mod.rs",                 // installs the wrapper for `Edits` builtins
            "src/eval/tests.rs",               // thread-local roundtrip tests
            "src/shared_state.rs",             // the accessor itself
        ];
        for file in &found {
            assert!(allowed.contains(&file.as_str()), "{file} takes the mutable Program handle; classify the new access in builtins/program_effects.rs");
        }
    }

    #[test]
    fn classified_names_are_registered_io_builtins_and_aliases_resolve() {
        for (name, access) in PROGRAM_ACCESS {
            assert!(crate::builtins::is_io_builtin(name), "{name}");
            assert_eq!(program_access(name), Some(*access));
        }
        for alias_owner in crate::builtins::IO_BUILTINS {
            for alias in alias_owner.aliases {
                assert_eq!(program_access(alias), program_access(alias_owner.name), "{alias}");
            }
        }
        assert_eq!(program_access("file_read"), None);
        assert_eq!(program_access("concat"), None);
        assert!(!changes_program("run_module_startups"));
        assert_eq!(
            program_changing_builtins(),
            ["mutate", "fn_remove", "type_remove", "module_remove", "module_create", "fn_replace", "move_symbols", "library_reload"]
        );
    }
}
