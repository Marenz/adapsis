//! Explicit operator CLI. No default database and no conversational provisioning.
#[derive(clap::Args)]
pub(crate) struct Args {
    #[arg(long)]
    database: std::path::PathBuf,
    #[arg(long)]
    grants: std::path::PathBuf,
    #[arg(long)]
    dry_run: bool,
}

pub(crate) fn run(args: Args) -> anyhow::Result<()> {
    let config = crate::memory_graph::records::grants::GrantConfig::read(&args.grants)
        .map_err(anyhow::Error::msg)?;
    let report=crate::memory_graph::records::grants::provision::apply(&args.database,&config,args.dry_run)
        .map_err(|error| anyhow::anyhow!("record provisioning failed ({error}); use an existing stopped database and a valid monotonic grant configuration; reconcile unknown outcomes before retry"))?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
