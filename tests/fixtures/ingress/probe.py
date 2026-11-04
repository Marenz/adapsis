async def echo(payload: str) -> str:
    return payload

@effects("io")
async def denied(payload: str) -> str:
    result: Result[str] = await records_read('{"version":1,"keys":["source"]}')
    match result:
        case Err(reason):
            return reason
        case Ok(body):
            return "unexpected"
