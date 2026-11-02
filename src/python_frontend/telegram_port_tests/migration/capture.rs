//! Optional private runtime snapshot overlay; no live getter/mutation is performed here.
use super::{Checked,digest,parser};
use std::{fs,io::Read,path::Path};
use std::os::unix::fs::{MetadataExt,OpenOptionsExt,PermissionsExt};

pub(super) fn read(path:&Path)->Checked<Vec<u8>> {
    let mut file=fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW)
        .open(path).map_err(|_| "private capture file open failed")?;
    let m=file.metadata().map_err(|_| "private capture stat failed")?;
    if !m.is_file() || m.permissions().mode() & 0o777 != 0o600 || m.nlink()!=1 || m.uid()!=unsafe{libc::geteuid()} {
        return Err("capture file not owner-only regular file");
    }
    let mut bytes=Vec::new();file.read_to_end(&mut bytes).map_err(|_| "capture read failed")?;Ok(bytes)
}

pub(super) fn overlay(m:&mut parser::ModuleDecl,path:&Path)->Checked<()> {
    let meta=fs::symlink_metadata(path).map_err(|_| "capture directory missing")?;
    if path.parent()!=Some(Path::new("/tmp/opencode")) || !meta.is_dir() ||
        meta.permissions().mode() & 0o777 != 0o700 || meta.uid()!=unsafe{libc::geteuid()} {
        return Err("capture directory not private");
    }
    let manifest:serde_json::Value=serde_json::from_slice(&read(&path.join("capture-complete.json"))?)
        .map_err(|_| "capture manifest invalid")?;
    if manifest["schema_version"]!=3 || manifest["runtime_state"]!=true || manifest["poll_terminal"]!=true {
        return Err("runtime snapshot not complete/drained");
    }
    let mut count=0;
    for op in &mut m.body {
        let parser::Operation::SharedVar(v)=op else {continue;};
        let raw=read(&path.join(format!("{}.value",v.name)))?;
        if manifest["sha256"][&v.name]!=digest(&raw) {return Err("runtime snapshot hash mismatch");}
        let text=String::from_utf8(raw).map_err(|_| "runtime snapshot invalid UTF-8")?;
        v.default=match &v.ty {
            parser::TypeExpr::Named(t) if t=="String" => parser::Expr::String(text),
            parser::TypeExpr::Named(t) if t=="Int" && v.name=="poll_offset" => {
                let number=text.parse::<i64>().map_err(|_| "cursor not i64")?;
                if number<0 || number.to_string()!=text {return Err("cursor not canonical nonnegative integer");}
                parser::Expr::Int(number)
            }
            parser::TypeExpr::Named(t) if t=="Bool" && v.name=="poll_running" && text=="false" => parser::Expr::Bool(false),
            _=>return Err("runtime snapshot field/type mismatch or polling still active"),
        };
        count+=1;
    }
    if count!=9 || manifest["sha256"].as_object().map(|x|x.len())!=Some(9) {return Err("runtime snapshot field inventory mismatch");}
    Ok(())
}

/// Startup reads token/allowlist files again. Prove that these reads will not replace
/// the captured effective authorization with different values at the checked instant.
pub(super) fn startup_files(m:&parser::ModuleDecl,p:&crate::ast::Program)->Checked<()> {
    let value=|name:&str|->Checked<String> {
        m.body.iter().find_map(|op|match op {
            parser::Operation::SharedVar(v) if v.name==name => match &v.default {parser::Expr::String(s)=>Some(s.clone()),_=>None},
            _=>None,
        }).ok_or("startup string setting missing")
    };
    let token=fs::read_to_string(value("token_path")?).map_err(|_| "startup token source unreadable")?;
    if token.trim()!=value("telegram_token")? {return Err("startup token source differs from effective runtime capture");}
    let allowed=value("allowed_path")?;
    if Path::new(&allowed).exists() {
        let raw=fs::read_to_string(allowed).map_err(|_| "startup allowlist source unreadable")?;
        let case=parser::TestCase{input:parser::Expr::String(raw),expected:parser::Expr::String(value("allowed_user_ids")?),matcher:None,after_checks:vec![]};
        crate::eval::eval_test_case(p,"TelegramBot.parse_id_list",&case)
            .map_err(|_| "startup allowlist differs from effective runtime capture")?;
    }
    Ok(())
}

#[test]
fn telegram_migration_private_files_reject_permissions_and_symlinks() {
    use std::os::unix::fs::symlink;
    let dir=tempfile::tempdir().unwrap();let file=dir.path().join("value");
    fs::write(&file,b"false").unwrap();fs::set_permissions(&file,fs::Permissions::from_mode(0o600)).unwrap();
    assert_eq!(read(&file).unwrap(),b"false");
    let link=dir.path().join("link");symlink(&file,&link).unwrap();assert!(read(&link).is_err());
    fs::set_permissions(&file,fs::Permissions::from_mode(0o644)).unwrap();assert!(read(&file).is_err());
}
