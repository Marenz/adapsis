@effects("io")
async def entry(payload: str) -> str:
    spawn(Flow.worker(payload))
    if contains(payload, "duplicate"):
        spawn(Flow.worker(payload))
    return "queued"

@effects("io")
async def worker(payload: str) -> str:
    text: str = json_get(payload, "message.text")
    profile: str = ""
    if json_kind(payload, "permission_profile") == "string":
        profile = json_get(payload, "permission_profile")
    context: str = "telegram:7"
    if contains(text, "mismatch"):
        context = "telegram:8"
    answer: str = await llm_takeover(context, text, "Flow.reply", "7", profile, '{"speaker_id":"telegram:user:7"}')
    if contains(text, "descendant"):
        spawn(Flow.descendant(text))
    return answer

@effects("io")
async def misbound(payload: str) -> str:
    text: str = json_get(payload, "message.text")
    answer: str = await llm_takeover("telegram:8", text, "Flow.reply", "7", "", '{"speaker_id":"telegram:user:7"}')
    return answer

@effects("io")
async def descendant(text: str) -> str:
    answer: str = await llm_takeover("telegram:9", text, "Flow.reply", "7", "", '{"speaker_id":"telegram:user:7"}')
    return answer

def reply(chat: str, text: str) -> str:
    return text

@effects("io")
async def write(value: str) -> str:
    result: Result[str] = await records_commit("fixture-write", '{"version":1,"conditions":[{"key":"proof-row","expected":{"kind":"absent"}}],"puts":[{"key":"proof-row","fields":[{"name":"evidence","value":"' + json_escape(value) + '"}]}]}')
    match result:
        case Ok(body):
            return "stored"
        case Err(reason):
            return reason

@effects("io")
async def write_late(value: str) -> str:
    result: Result[str] = await records_commit("late-write", '{"version":1,"conditions":[{"key":"late-row","expected":{"kind":"absent"}}],"puts":[{"key":"late-row","fields":[{"name":"evidence","value":"' + json_escape(value) + '"}]}]}')
    match result:
        case Ok(body):
            return "stored"
        case Err(reason):
            return reason
