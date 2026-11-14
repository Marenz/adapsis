//! A frozen handle refuses every classified program-changing builtin before the builtin
//! reads its arguments or reaches the Program wrapper; everything else is untouched.
use super::super::*;
use crate::builtins::program_effects::{program_changing_builtins, program_access, ProgramAccess};

const SOURCE: &str = "+module FreezeUnit\n+fn victim ()->Int\n  +return 1\n+end\n";

/// A handle with no mocks and no IO loop, plus the thread-local Program wrapper the
/// program-changing builtins write to.
fn setup(frozen: bool) -> (CoroutineHandle, std::sync::Arc<std::sync::RwLock<crate::ast::Program>>) {
    let mut program = crate::ast::Program::default();
    for op in crate::parser::parse(SOURCE).unwrap() {
        crate::validator::apply_and_validate(&mut program, &op).unwrap();
    }
    program.rebuild_function_index();
    crate::shared_state::set_shared_program(Some(std::sync::Arc::new(program.clone())));
    let wrapper = std::sync::Arc::new(std::sync::RwLock::new(program));
    crate::shared_state::set_shared_program_mut(Some(wrapper.clone()));
    let (sender, _receiver) = mpsc::channel(1);
    (CoroutineHandle::new(sender).with_program_frozen(frozen), wrapper)
}

fn spellings() -> Vec<&'static str> {
    crate::builtins::IO_BUILTINS
        .iter()
        .flat_map(|builtin| std::iter::once(builtin.name).chain(builtin.aliases.iter().copied()))
        .filter(|name| crate::builtins::program_effects::changes_program(name))
        .collect()
}

#[test]
fn a_frozen_handle_refuses_every_program_changing_builtin_before_it_touches_anything() {
    let (handle, wrapper) = setup(true);
    assert!(handle.program_frozen());
    let before = wrapper.read().unwrap().clone();
    let names = spellings();
    for builtin in program_changing_builtins() {
        assert!(names.contains(&builtin), "{builtin}");
    }
    for name in names {
        // Valid-looking arguments, no arguments and wrong types get the same refusal: the
        // check precedes argument handling.
        for args in [
            vec![Value::string("+module FreezeUnitNew"), Value::string("FreezeUnit")],
            vec![],
            vec![Value::Int(1)],
        ] {
            let error = handle.execute_await(name, &args).unwrap_err().to_string();
            assert!(error.starts_with(&format!("{name}: refused.")), "{name}: {error}");
            assert!(error.contains("a run block cannot change the program"), "{name}: {error}");
            assert!(error.contains("Use the mutate tool"), "{name}: {error}");
        }
    }
    let after = wrapper.read().unwrap();
    assert!(*after == before);
    assert!(std::sync::Arc::ptr_eq(&after.modules[0].functions[0], &before.modules[0].functions[0]));
    assert_eq!(after.modules.len(), 1);
}

#[test]
fn the_freeze_survives_cloning_and_leaves_other_builtins_alone() {
    let (handle, wrapper) = setup(true);
    let cloned = handle.clone();
    assert!(cloned.program_frozen());
    assert!(cloned.execute_await("mutate", &[Value::string("+module FreezeUnitClone")]).is_err());
    assert!(matches!(Value::CoroutineHandle(cloned.clone()), Value::CoroutineHandle(h) if h.program_frozen()));
    // Reads of the Program and unrelated in-process builtins still work.
    assert_eq!(program_access("run_module_startups"), Some(ProgramAccess::Reads));
    let startups = cloned.execute_await("run_module_startups", &[]).unwrap();
    assert!(startups.to_string().contains("no modules have startup blocks"), "{startups}");
    let symbols = cloned.execute_await("query_symbols", &[]).unwrap();
    assert!(symbols.to_string().contains("victim"), "{symbols}");
    assert_eq!(wrapper.read().unwrap().modules.len(), 1);
}

#[test]
fn an_unfrozen_handle_is_unchanged() {
    let (handle, wrapper) = setup(false);
    assert!(!handle.program_frozen());
    let applied = handle.execute_await("mutate", &[Value::string("+module FreezeUnitNew")]).unwrap();
    assert!(applied.to_string().contains("Applied 1 mutation"), "{applied}");
    assert_eq!(wrapper.read().unwrap().modules.len(), 2);
    for constructed in [
        CoroutineHandle::new(handle.io_sender()),
        CoroutineHandle::new_mock(vec![]),
        CoroutineHandle::new(handle.io_sender()).with_turn(None),
    ] {
        assert!(!constructed.program_frozen());
    }
}

#[test]
fn a_spawned_task_handle_inherits_the_freeze_it_is_given() {
    let (sender, _receiver) = mpsc::channel(1);
    let registries = || (TaskRegistry::default(), TaskSnapshotRegistry::default());
    for frozen in [true, false] {
        let (tasks, snapshots) = registries();
        let handle = CoroutineHandle::for_spawned_task(sender.clone(), 7, tasks, snapshots, frozen);
        assert_eq!(handle.task_id, Some(7));
        assert_eq!(handle.program_frozen(), frozen);
        assert_eq!(handle.execute_await("fn_remove", &[Value::string("Missing.f")]).unwrap_err().to_string().contains("refused"), frozen);
    }
}
