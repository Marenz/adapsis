//! Shared call-site admission rules for source lowering and dispatch validation.
//! Modes apply only to the root call; arguments are always ordinary expressions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Site {
    Ordinary,
    AwaitBinding,
    DirectBuiltinMatch,
    SpawnTarget,
}

#[derive(Clone, Copy, Default)]
pub(super) struct Effects {
    pub asynchronous: bool,
    pub io: bool,
}

impl Effects {
    pub fn from_names(names: &[String]) -> Self {
        Self {
            asynchronous: names.iter().any(|e| e == "async"),
            io: names.iter().any(|e| e == "io"),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct UserSignature {
    pub effects: Effects,
    pub returns_result: bool,
}

#[derive(Clone, Copy)]
pub(super) enum Target {
    // Non-IO bare constructors still undergo the existing type/authored gates.
    BuiltinOrConstructor { io: bool },
    // None is allowed only during unresolved standalone source lowering. Native
    // dispatch supplies a signature before executing any submitted function.
    User(Option<UserSignature>),
}

pub(super) fn check(site: Site, target: Target, caller: Effects) -> Result<(), &'static str> {
    match site {
        Site::DirectBuiltinMatch if matches!(target, Target::BuiltinOrConstructor { io: true }) => {
            if !caller.io {
                return Err("IO match subject requires @effects('io')");
            }
            if !caller.asynchronous {
                return Err("IO match subject requires async def");
            }
            Ok(())
        }
        Site::DirectBuiltinMatch | Site::Ordinary => match target {
            Target::BuiltinOrConstructor { io: true } => Err(
                "IO/async call requires an annotated await binding; unawaited async values are unsupported",
            ),
            Target::User(Some(s)) if s.effects.asynchronous || s.effects.io => Err(
                "IO/async call requires an annotated await binding; unawaited async values are unsupported",
            ),
            _ => Ok(()),
        },
        Site::AwaitBinding => {
            if !caller.asynchronous {
                return Err("await requires async def");
            }
            let io = match target {
                Target::BuiltinOrConstructor { io: false } => {
                    return Err(
                        "await requires an async function or runtime IO builtin; synchronous calls are not awaitable",
                    );
                }
                Target::BuiltinOrConstructor { io: true } => true,
                Target::User(Some(s)) => {
                    if !s.effects.asynchronous {
                        return Err(
                            "await requires an async function or runtime IO builtin; synchronous calls are not awaitable",
                        );
                    }
                    if s.returns_result {
                        return Err(
                            "awaiting Result-returning user functions is unsupported until runtime Result unwrapping is reconciled (#82)",
                        );
                    }
                    s.effects.io
                }
                Target::User(None) => false,
            };
            if io && !caller.io {
                return Err("IO await requires @effects('io')");
            }
            Ok(())
        }
        Site::SpawnTarget => {
            let Target::User(signature) = target else {
                return Err(
                    "spawn requires a qualified user-function target, not a builtin or constructor",
                );
            };
            if !caller.asynchronous {
                return Err("spawn requires async def for its coroutine context");
            }
            if signature.is_some_and(|s| s.effects.io) && !caller.io {
                return Err("spawning an IO function requires @effects('io')");
            }
            Ok(())
        }
    }
}

/// Metadata-only predicate shared by #83 literal safety, inspection and call
/// admission. Keep current and same-submission collision checks additive.
pub(crate) fn runtime_shadowed(program: &crate::ast::Program, name: &str) -> bool {
    fn types(types: &[crate::ast::TypeDecl], name: &str) -> bool {
        types.iter().any(|ty| matches!(ty, crate::ast::TypeDecl::TaggedUnion(union) if union.variants.iter().any(|variant| variant.name == name)))
    }
    fn module_shadowed(node: &crate::ast::Module, name: &str) -> bool {
        node.functions.iter().any(|function| function.name == name)
            || types(&node.types, name)
            || node
                .modules
                .iter()
                .any(|child| module_shadowed(child, name))
    }
    program
        .functions
        .iter()
        .any(|function| function.name == name)
        || types(&program.types, name)
        || program
            .modules
            .iter()
            .any(|value| module_shadowed(value, name))
}
