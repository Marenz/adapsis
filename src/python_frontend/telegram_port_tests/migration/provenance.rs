//! Hash-bound v3 capture chain. This reads evidence only; it never queries the service.
use super::{capture::read,digest,Checked};
use serde_json::{json,Value};
use std::{fs,path::{Path,PathBuf}};
use std::os::unix::fs::{MetadataExt,PermissionsExt};

fn dir(path:&Path)->Checked<()> {
    let m=fs::symlink_metadata(path).map_err(|_| "provenance directory absent")?;
    if path.parent()!=Some(Path::new("/tmp/opencode")) || !m.is_dir() ||
        m.permissions().mode() & 0o777 != 0o700 || m.uid()!=unsafe{libc::geteuid()} {
        return Err("provenance directory not private");
    } Ok(())
}
fn load(path:&Path)->Checked<Value> {serde_json::from_slice(&read(path)?).map_err(|_| "provenance JSON invalid")}
fn hash(path:&Path)->Checked<String> {Ok(digest(&read(path)?))}
fn require(ok:bool)->Checked<()> {if ok {Ok(())} else {Err("capture provenance chain mismatch")}}
fn text(v:&Value)->Checked<&str> {v.as_str().ok_or("provenance string absent")}

pub(super) fn verify(path:&Path,inputs:&[String])->Checked<Value> {
    dir(path)?;
    let request=load(&path.join("request.json"))?;
    require(request["schema_version"]==3)?;
    let initial=PathBuf::from(text(&request["initial_candidate"])?);dir(&initial)?;
    require(request["initial_manifest_sha256"]==hash(&initial.join("manifest.json"))?)?;
    require(request["initial_library_sha256"]==hash(&initial.join("library-verified.json"))?)?;
    let initial_manifest=load(&initial.join("manifest.json"))?;
    let initial_attest=load(&initial.join("library-verified.json"))?;
    require(initial_manifest["schema_version"]==3 && initial_manifest["superseded"]==false)?;
    require(initial_attest["schema_version"]==3 && initial_attest["load_reload_passed"]==true)?;
    require(initial_attest["manifest_sha256"]==request["initial_manifest_sha256"] &&
        initial_attest["artifacts"]==initial_manifest["artifacts"])?;
    for name in ["TelegramBot.ax","TelegramBot.py","TgFormat.ax","TgFormat.py"] {
        require(initial_manifest["artifacts"][name]==hash(&initial.join(name))?)?;
    }
    require(initial_manifest["live_sha256"]==json!(inputs))?;
    let helper=load(&path.join("helpers-job.json"))?;
    let applied=load(&path.join("helpers-applied.json"))?;
    require(helper["job_id"]==request["job_id"] && helper["request_sha256"]==hash(&path.join("request.json"))?)?;
    require(helper["helpers_sha256"]==hash(&path.join("helpers.ax"))?)?;
    require(applied["job_id"]==request["job_id"] && applied["helpers_job_sha256"]==hash(&path.join("helpers-job.json"))?)?;
    let backup=PathBuf::from(text(&helper["backup"])?);dir(&backup)?;
    require(helper["backup_manifest_sha256"]==hash(&backup.join("backup.json"))?)?;
    let saved=load(&backup.join("backup.json"))?;
    require(saved["job_id"]==request["job_id"] && saved["request_sha256"]==helper["request_sha256"])?;
    require(saved["hashes"].as_object().map(|x|x.len())==Some(3))?;
    for name in ["TelegramBot.ax","TgFormat.ax","session-before-helpers.json"] {
        require(saved["hashes"][name]==hash(&backup.join(name))?)?;
    }
    require(json!([saved["hashes"]["TelegramBot.ax"],saved["hashes"]["TgFormat.ax"]])==json!(inputs))?;
    let chain=json!({"job_id":request["job_id"],"request_sha256":helper["request_sha256"],
        "helpers_job_sha256":hash(&path.join("helpers-job.json"))?,
        "helpers_applied_sha256":hash(&path.join("helpers-applied.json"))?,
        "backup_manifest_sha256":helper["backup_manifest_sha256"],
        "initial_manifest_sha256":request["initial_manifest_sha256"],
        "initial_library_sha256":request["initial_library_sha256"]});
    verify_stopped(path,chain)
}

fn verify_stopped(path:&Path,mut chain:Value)->Checked<Value> {
    let capture=load(&path.join("capture-complete.json"))?;
    let fresh=load(&path.join("recapture-complete.json"))?;
    // REQUIRED, not an optional marker: exporter must never accept a merely drained job.
    let stopped=load(&path.join("service-stopped.json"))?;
    let started=load(&path.join("drain-started.json"))?;
    require(capture["schema_version"]==3 && stopped["schema_version"]==3 && stopped["stopped"]==true)?;
    require(capture["chain"]==chain && fresh["chain"]==chain && stopped["chain"]==chain && started["chain"]==chain)?;
    require(capture["identity"]==fresh["identity"] && capture["identity"]==stopped["identity"] && capture["identity"]==started["identity"])?;
    require(capture["sha256"]==fresh["sha256"])?;
    require(capture["drain_started_sha256"]==hash(&path.join("drain-started.json"))?)?;
    let capture_hash=hash(&path.join("capture-complete.json"))?;
    require(fresh["capture_sha256"]==capture_hash && stopped["capture_sha256"]==capture_hash)?;
    require(stopped["recapture_sha256"]==hash(&path.join("recapture-complete.json"))?)?;
    for (name,expected) in capture["sha256"].as_object().ok_or("capture hashes absent")? {
        if !["telegram_token","token_path","allowed_path","admin_user_ids","allowed_user_ids",
            "sandbox_perm_model","intake_mode","poll_offset","poll_running"].contains(&name.as_str()) {
            return Err("unknown capture field");
        }
        require(*expected==hash(&path.join(format!("{name}.value")))?)?;
        require(*expected==hash(&path.join(format!("{name}_recheck.value")))?)?;
    }
    chain["identity"]=capture["identity"].clone();
    chain["capture_sha256"]=json!(capture_hash);
    chain["recapture_sha256"]=json!(hash(&path.join("recapture-complete.json"))?);
    chain["stopped_sha256"]=json!(hash(&path.join("service-stopped.json"))?);
    Ok(chain)
}

#[test]
fn telegram_migration_missing_stop_attestation_is_refused() {
    let dir=tempfile::tempdir().unwrap();
    let p=dir.path();let identity=json!({"pid":123,"start_ticks":"456","exe_sha256":"synthetic"});
    let write=|name:&str,value:Value| super::private_file(p,name,serde_json::to_string(&value).unwrap().as_bytes()).unwrap();
    write("drain-started.json",json!({"chain":{},"identity":identity}));
    write("capture-complete.json",json!({"schema_version":3,"chain":{},"identity":identity,
        "sha256":{},"drain_started_sha256":hash(&p.join("drain-started.json")).unwrap()}));
    write("recapture-complete.json",json!({"chain":{},"identity":identity,"sha256":{},
        "capture_sha256":hash(&p.join("capture-complete.json")).unwrap()}));
    write("service-stopped.json",json!({"schema_version":3,"stopped":true,"chain":{},"identity":identity,
        "capture_sha256":hash(&p.join("capture-complete.json")).unwrap(),
        "recapture_sha256":hash(&p.join("recapture-complete.json")).unwrap()}));
    // This is the stopped-attestation boundary only; the outer overlay separately
    // enforces nine typed fields. First prove a valid chain, then remove only S.
    assert!(verify_stopped(p,json!({})).is_ok());
    fs::remove_file(p.join("service-stopped.json")).unwrap();
    assert!(verify_stopped(dir.path(),json!({})).is_err());
}
