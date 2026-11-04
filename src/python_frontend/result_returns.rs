//! Return-context-only admission for the accepted runtime Result contract (#82/#86 G7).
//! A function declaring Result[T] can return a plain, EXACT known T: invocation wraps it
//! once. This is Adapsis runtime semantics, not CPython inference or a general coercion.
//! The lowering emits the same Return AST, never inserts Ok. Existing explicit-constructor
//! validation is unchanged; malformed payload validation is not broadened here.
use super::expressions::Ty;

pub(super) fn implicit_inner_matches(expected: &Ty, actual: &Ty) -> bool {
    let Ty::Result(inner) = expected else {
        return false;
    };
    // Result[Result[T]] needs an explicit outer constructor. The caller's existing type
    // checking handles already-matching Result returns; don't silently lift an inner one.
    // Some shares Value::Ok with Result success, so an outer Option is ambiguous too.
    // A List[Option[T]] has a distinct outer Value::List and can still be wrapped once.
    !matches!(inner.as_ref(), Ty::Result(_) | Ty::Option(_))
        && inner.as_ref() == actual
        && known(actual)
}

fn known(ty: &Ty) -> bool {
    let mut pending = vec![ty];
    while let Some(ty) = pending.pop() {
        match ty {
            Ty::Unknown => return false,
            Ty::List(t) | Ty::Set(t) | Ty::Option(t) | Ty::Result(t) => pending.push(t),
            Ty::Map(k, v) => pending.extend([k.as_ref(), v.as_ref()]),
            _ => {}
        }
    }
    true
}

#[cfg(test)]
mod tests;
