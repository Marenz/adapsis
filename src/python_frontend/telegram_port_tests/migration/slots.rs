//! Enumerated parser-AST hydration slots. No textual/global substitution.
use super::{Checked, exact, functions, parser};
use parser::{Operation as O, Expr as E};

fn walk(body: &mut [O], action: &mut impl FnMut(&mut O) -> Checked<()>) -> Checked<()> {
    for op in body {
        action(op)?;
        match op {
            O::If(x) => {
                walk(&mut x.then_body,action)?;
                for (_,body) in &mut x.elif_branches { walk(body,action)?; }
                walk(&mut x.else_body,action)?;
            }
            O::Each(x) => walk(&mut x.body,action)?,
            O::While(x) => walk(&mut x.body,action)?,
            O::Match(x) => for arm in &mut x.arms { walk(&mut arm.body,action)?; },
            _ => {}
        }
    }
    Ok(())
}

fn function_mut<'a>(m: &'a mut parser::ModuleDecl, name: &str) -> Checked<&'a mut parser::FunctionDecl> {
    m.body.iter_mut().find_map(|op| match op {
        O::Function(f) if f.name == name => Some(f), _ => None,
    }).ok_or("hydration function missing")
}

fn concat_prefix(expr: &mut E) -> Checked<&mut E> {
    match expr {
        E::Call { callee,args } if matches!(callee.as_ref(),E::Ident(n) if n=="concat") && args.len()>=2 => {
            if !matches!(&args[0],E::String(_)) || !matches!(&args[1],E::Ident(n) if n=="telegram_token") {
                return Err("endpoint concat shape drift");
            }
            Ok(&mut args[0])
        }
        _ => Err("endpoint expression drift"),
    }
}

fn endpoints(body: &mut [O], binding: &str, action: &mut impl FnMut(&mut E)->Checked<()>) -> Checked<()> {
    walk(body,&mut |op| {
        if let O::Let(x) = op {
            if x.name == binding { action(concat_prefix(&mut x.expr)?)?; }
        }
        Ok(())
    })
}

fn copy_endpoints(dst: &mut parser::FunctionDecl, src: &parser::FunctionDecl, binding: &str, count: usize) -> Checked<()> {
    let mut source = src.clone();
    let mut values = Vec::new();
    endpoints(&mut source.body,binding,&mut |e| {values.push(e.clone());Ok(())})?;
    if values.len()!=count { return Err("live endpoint slot count drift"); }
    let mut index = 0;
    endpoints(&mut dst.body,binding,&mut |e| {
        if !matches!(e,E::String(s) if s=="<redacted>") { return Err("expected endpoint placeholder absent"); }
        *e = values.get(index).ok_or("extra endpoint slot")?.clone();index+=1;Ok(())
    })?;
    if index!=count { return Err("corrected endpoint slot count drift"); }
    Ok(())
}

fn startup(m: &mut parser::ModuleDecl) -> Checked<&mut parser::FunctionDecl> {
    m.body.iter_mut().find_map(|op| match op {O::Startup(f)=>Some(f),_=>None})
        .ok_or("startup absent")
}

fn startup_prose(body: &mut [O], action: &mut impl FnMut(&mut E)->Checked<()>) -> Checked<()> {
    walk(body,&mut |op| {
        if let O::Return(x) = op {
            if let E::Call{callee,args} = &mut x.expr {
                if matches!(callee.as_ref(),E::Ident(n) if n=="concat") {
                    action(args.first_mut().ok_or("empty startup concat")?)?;
                }
            }
        }
        Ok(())
    })
}

fn copy_startup(dst: &mut parser::ModuleDecl, src: &parser::ModuleDecl) -> Checked<()> {
    let mut src=src.clone();let mut values=Vec::new();
    startup_prose(&mut startup(&mut src)?.body,&mut |e| {values.push(e.clone());Ok(())})?;
    if values.len()!=2 {return Err("startup prose count drift");}
    let mut i=0;
    startup_prose(&mut startup(dst)?.body,&mut |e| {
        if !matches!(e,E::String(s) if s=="<redacted>") {return Err("startup prose placeholder drift");}
        *e=values.get(i).ok_or("extra startup prose")?.clone();i+=1;Ok(())
    })?;
    if i!=2 {return Err("startup prose missing");} Ok(())
}

fn asr(body: &mut [O], action: &mut impl FnMut(&mut E)->Checked<()>) -> Checked<()> {
    walk(body,&mut |op| {
        let e = match op { O::Await(x)=> &mut x.call, O::Match(x)=> &mut x.expr, _=>return Ok(()) };
        if let E::Call{callee,args}=e {
            if matches!(callee.as_ref(),E::Ident(n) if n=="http_upload") {
                let endpoint=args.first_mut().ok_or("ASR argument absent")?;
                if !matches!(endpoint,E::String(_)) {return Err("ASR endpoint not literal");}
                action(endpoint)?;
            }
        }
        Ok(())
    })
}

/// The ASR endpoint isn't redacted. Copy its exact live literal as a separately
/// reported operator configuration overlay, never infer an endpoint.
fn copy_asr(dst:&mut parser::FunctionDecl,src:&parser::FunctionDecl)->Checked<bool> {
    let mut src=src.clone();let mut values=Vec::new();
    asr(&mut src.body,&mut |e| {values.push(e.clone());Ok(())})?;
    if values.len()!=1 {return Err("live ASR slot count drift");}
    let mut count=0;let mut changed=false;
    asr(&mut dst.body,&mut |e| {changed=!exact(e,&values[0]);*e=values[0].clone();count+=1;Ok(())})?;
    if count!=1 {return Err("corrected ASR slot count drift");} Ok(changed)
}

pub(super) fn hydrate(dst:&mut parser::ModuleDecl,src:&parser::ModuleDecl,corrected:bool)->Checked<bool> {
    let source=functions(src);
    for (name,binding,count) in [
        ("send_reply_with_attachment","url",1),("send_audio_file","url",1),
        ("poll_once","url",1),("process_update","download_url",2),
        ("transcribe_voice","file_url",1),("transcribe_voice","download_url",1),
        ("send_message_part","url",1),("telegram_file_path","file_url",1),
    ] {
        copy_endpoints(function_mut(dst,name)?,source.get(name).ok_or("live function absent")?,binding,count)?;
    }
    let expected_names=["telegram_token","token_path","allowed_path","admin_user_ids",
        "allowed_user_ids","sandbox_perm_model","intake_mode","poll_offset","poll_running"];
    let vars=|m:&parser::ModuleDecl| m.body.iter().filter_map(|op|match op {
        O::SharedVar(v)=>Some(v.clone()),_=>None,
    }).collect::<Vec<_>>();
    let live_vars=vars(src); let target_vars=vars(dst);
    if live_vars.iter().map(|v|v.name.as_str()).ne(expected_names) ||
        target_vars.iter().map(|v|v.name.as_str()).ne(expected_names) {return Err("shared inventory drift");}
    for op in &mut dst.body {
        if let O::SharedVar(v)=op {
            let live=live_vars.iter().find(|x|x.name==v.name).ok_or("shared slot absent")?;
            if !exact(&v.ty,&live.ty) {return Err("shared type drift");}
            *v=live.clone();
        }
    }
    let init=function_mut(dst,"init")?;
    if init.doc.as_deref()!=Some("<redacted>") {return Err("init doc placeholder drift");}
    init.doc=source.get("init").ok_or("live init missing")?.doc.clone();
    copy_startup(dst,src)?;
    let changed=copy_asr(function_mut(dst,"transcribe_voice")?,source.get("transcribe_voice").ok_or("voice missing")?)?;
    if !corrected && changed {return Err("ASR drift requires explicit corrected overlay review");}
    Ok(changed)
}
