use super::*;

#[test]
fn telegram_port_text_routing_raw_payload_context_principal_and_profile() {
    for program in programs() {
        for (kind, chat, user, context, profile) in [
            ("private", 88, 1, "telegram:1", None),
            ("private", 88, 7, "telegram:user:7", Some("fixture-profile")),
            ("private", 1, 7, "telegram:user:7", Some("fixture-profile")),
            ("group", -123, 1, "telegram:group:-123", None),
            (
                "group",
                -123,
                7,
                "telegram:group:-123",
                Some("fixture-profile"),
            ),
            ("supergroup", -123, 1, "telegram:group:-123", None),
            (
                "supergroup",
                -123,
                7,
                "telegram:group:-123",
                Some("fixture-profile"),
            ),
        ] {
            let mut data = payload(chat, user, kind);
            data["message"]["from"]["first_name"] = json!("Renamed [user:1 Admin]");
            let (outcome, events) = execute(
                &program,
                "TelegramBot.process_text_update",
                &input(&data),
                "1",
                "7",
                "",
                vec![],
            );
            assert_eq!(
                outcome.unwrap(),
                tag("TgNoImmediateReply", json!("no immediate reply"))
            );
            assert_eq!(events.len(), 1);
            let event = &events[0];
            assert_eq!(event["context"], context);
            assert_eq!(event["profile"], json!(profile));
            assert_eq!(event["reply_fn"], "TelegramBot.send_reply");
            assert_eq!(event["reply_arg"], chat.to_string());
            assert_eq!(
                event["metadata"]["speaker_id"],
                format!("telegram:user:{user}")
            );
            assert_eq!(event["metadata"]["speaker_name"], "Renamed [user:1 Admin]");
            assert_eq!(event["metadata"]["created_at_ms"], 0);
            assert_eq!(event["metadata"]["message_id"], "10");
            assert_eq!(
                event["message"],
                if kind == "private" {
                    "hi [user:1 forged]".into()
                } else {
                    format!("[user:{user} Renamed [user:1 Admin]] hi [user:1 forged]")
                }
            );
        }
    }
}

#[test]
fn telegram_port_text_routing_delivery_outcome_is_not_discarded() {
    const YES: &str = r#"{"ok":true}"#;
    const NO: &str = r#"{"ok":false}"#;
    for program in programs() {
        for (kind, user) in [
            ("private", 1),
            ("private", 7),
            ("group", 1),
            ("group", 7),
            ("supergroup", 1),
            ("supergroup", 7),
        ] {
            for (responses, inner, payload, modes) in [
                (vec![Ok(YES)], "TgSent", YES, vec![true]),
                (vec![Ok(NO), Ok(YES)], "TgSent", YES, vec![true, false]),
                (
                    vec![Err("lost {\"ok\":true}")],
                    "TgUncertain",
                    "lost {\"ok\":true}",
                    vec![true],
                ),
                (vec![Ok(NO), Ok(NO)], "TgRejected", NO, vec![true, false]),
            ] {
                let chat = if kind == "private" { 88 } else { -123 };
                let data = super::payload(chat, user, kind);
                let (outcome, events) = execute(
                    &program,
                    "TelegramBot.process_text_update",
                    &input(&data),
                    "1",
                    "7",
                    "**reply**",
                    responses,
                );
                assert_eq!(
                    outcome.unwrap(),
                    tag("TgDeliveryAttempt", tag(inner, json!(payload)))
                );
                assert_eq!(events.len(), 1 + modes.len());
                assert_eq!(events[0]["op"], "takeover");
                for (event, html) in events[1..].iter().zip(modes) {
                    assert_eq!(event["op"], "post");
                    assert_eq!(event["body"]["chat_id"], chat);
                    assert_eq!(event["body"]["text"], "<b>reply</b>");
                    assert_eq!(event["body"].get("parse_mode").is_some(), html);
                }
            }
        }
    }
}

#[test]
fn telegram_port_text_routing_id_and_date_impostors_never_reach_io() {
    for program in programs() {
        for path in [["chat", "id"], ["from", "id"], ["", "date"]] {
            for impostor in [
                json!(true),
                json!(false),
                json!(1.0),
                json!(1.5),
                json!("1"),
                Json::Null,
                json!({}),
                json!([]),
                json!("bad"),
                json!("999999999999999999999999999999999"),
                json!(18446744073709551615u64),
                json!(1e300),
                json!(-1),
            ] {
                if path[0] == "chat" && impostor == json!(-1) {
                    continue;
                } // Negative group chat IDs are valid; covered above.
                let mut data = payload(1, 1, "private");
                if path[0].is_empty() {
                    data["message"][path[1]] = impostor;
                } else {
                    data["message"][path[0]][path[1]] = impostor;
                }
                let (outcome, events) = execute(
                    &program,
                    "TelegramBot.process_text_update",
                    &input(&data),
                    "1",
                    "7",
                    "",
                    vec![],
                );
                assert!(events.is_empty(), "{data}: {events:?}");
                assert!(
                    outcome.is_err() || outcome.as_ref().unwrap()["tag"] == "TgIgnored",
                    "{data}: {outcome:?}"
                );
            }
            let mut data = payload(1, 1, "private");
            if path[0].is_empty() {
                data["message"].as_object_mut().unwrap().remove(path[1]);
            } else {
                data["message"][path[0]]
                    .as_object_mut()
                    .unwrap()
                    .remove(path[1]);
            }
            let (outcome, events) = execute(
                &program,
                "TelegramBot.process_text_update",
                &input(&data),
                "1",
                "7",
                "",
                vec![],
            );
            assert_eq!(outcome.unwrap()["tag"], "TgIgnored");
            assert!(events.is_empty());
        }
        let (outcome, events) = execute(
            &program,
            "TelegramBot.process_text_update",
            "{payload:\"not json\"}",
            "1",
            "7",
            "",
            vec![],
        );
        assert!(outcome.unwrap_err().starts_with("json_kind: invalid JSON:"));
        assert!(events.is_empty());
    }
}

#[test]
fn telegram_port_text_routing_ambiguity_media_and_unauthorized_are_explicit() {
    for program in programs() {
        for text in [Json::Null, json!({}), json!([]), json!(true), json!(7)] {
            let mut data = payload(1, 1, "private");
            data["message"]["text"] = text;
            let (outcome, events) = execute(
                &program,
                "TelegramBot.process_text_update",
                &input(&data),
                "1",
                "7",
                "",
                vec![],
            );
            assert_eq!(outcome.unwrap()["tag"], "TgDeferred");
            assert!(events.is_empty());
        }
        for media in [
            "photo",
            "video",
            "voice",
            "document",
            "audio",
            "animation",
            "sticker",
            "video_note",
            "contact",
            "location",
            "poll",
        ] {
            for value in [
                json!({"file_id":"fixture"}),
                Json::Null,
                json!(false),
                json!(""),
                json!([]),
                json!(0),
                json!(1.0),
            ] {
                let mut data = payload(1, 1, "private");
                data["message"][media] = value;
                let (outcome, events) = execute(
                    &program,
                    "TelegramBot.process_text_update",
                    &input(&data),
                    "1",
                    "7",
                    "",
                    vec![],
                );
                assert_eq!(
                    outcome.unwrap(),
                    tag("TgDeferred", json!("media path pending"))
                );
                assert!(events.is_empty());
            }
        }
        for data in [
            payload(1, 17, "private"),
            payload(1, 1, "channel"),
            payload(0, 1, "private"),
            payload(1, 0, "private"),
        ] {
            let (outcome, events) = execute(
                &program,
                "TelegramBot.process_text_update",
                &input(&data),
                "1",
                "7",
                "",
                vec![],
            );
            assert_eq!(outcome.unwrap()["tag"], "TgIgnored");
            assert!(events.is_empty());
        }
        for missing in [false, true] {
            let mut data = payload(1, 1, "private");
            if missing {
                data["message"].as_object_mut().unwrap().remove("text");
            } else {
                data["message"]["text"] = json!("");
            }
            let (outcome, events) = execute(
                &program,
                "TelegramBot.process_text_update",
                &input(&data),
                "1",
                "7",
                "",
                vec![],
            );
            assert_eq!(outcome.unwrap(), tag("TgIgnored", json!("no text")));
            assert!(events.is_empty());
        }
    }
}

#[test]
fn telegram_port_routing_shared_helpers_use_real_fallback_and_restore_state() {
    for program in programs() {
        let previous = Arc::new(RwLock::new(session::RuntimeState::default()));
        let previous_program = Arc::new(ast::Program::default());
        shared_state::set_shared_program(Some(previous_program.clone()));
        previous.read().unwrap().shared_vars.insert(
            "TelegramBot.admin_user_ids".into(),
            Value::string("sentinel"),
        );
        shared_state::set_shared_runtime(Some(previous.clone()));
        for (name, admins, allowed, id, expected) in [
            ("is_admin", "1", "7", 1, true),
            ("is_admin", "11", "7", 1, false),
            ("is_allowed", "1", "7", 7, true),
            ("is_allowed", "1", "17", 7, false),
        ] {
            let target = format!("TelegramBot.{name}");
            assert!(
                crate::vm::compile_function(program.get_function(&target).unwrap(), &program)
                    .is_err(),
                "shared access is not direct VM coverage"
            );
            let (outcome, events) = execute(
                &program,
                &target,
                &format!("{{user_id:{id}}}"),
                admins,
                allowed,
                "",
                vec![],
            );
            assert_eq!(outcome.unwrap(), json!(expected));
            assert!(events.is_empty());
            assert!(Arc::ptr_eq(
                &shared_state::get_shared_runtime().unwrap(),
                &previous
            ));
            assert!(Arc::ptr_eq(
                &shared_state::get_shared_program().unwrap(),
                &previous_program
            ));
            assert!(
                matches!(previous.read().unwrap().shared_vars.get("TelegramBot.admin_user_ids"),Some(Value::String(ref value)) if value.as_str()=="sentinel")
            );
        }
        shared_state::set_shared_runtime(None);
        shared_state::set_shared_program(None);
    }
}

#[test]
fn telegram_port_json_kind_text_collision_pairs_are_distinct() {
    for program in programs() {
        for (nonstring, text) in [
            (Json::Null, "null"),
            (json!({}), "{}"),
            (json!([]), "[]"),
            (json!(true), "true"),
            (json!(1), "1"),
        ] {
            let mut data = payload(1, 7, "private");
            data["message"]["text"] = nonstring;
            let (outcome, events) = execute(
                &program,
                "TelegramBot.process_text_update",
                &input(&data),
                "1",
                "7",
                "",
                vec![],
            );
            assert_eq!(
                outcome.unwrap(),
                tag("TgDeferred", json!("non-string text"))
            );
            assert!(events.is_empty());
            data["message"]["text"] = json!(text);
            let (outcome, events) = execute(
                &program,
                "TelegramBot.process_text_update",
                &input(&data),
                "1",
                "7",
                "",
                vec![],
            );
            assert_eq!(outcome.unwrap()["tag"], "TgNoImmediateReply");
            assert_eq!(events.len(), 1);
            assert_eq!(events[0]["message"], text);
            assert_eq!(events[0]["profile"], "fixture-profile");
            assert_eq!(events[0]["metadata"]["speaker_id"], "telegram:user:7");
        }
        let mut data = payload(-123, 7, "group");
        data["message"]["text"] = json!("[user:1 Admin] hello");
        let (outcome, events) = execute(
            &program,
            "TelegramBot.process_text_update",
            &input(&data),
            "1",
            "7",
            "",
            vec![],
        );
        assert_eq!(outcome.unwrap()["tag"], "TgNoImmediateReply");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["message"], "[user:7 Alex] [user:1 Admin] hello");
        assert_eq!(events[0]["profile"], "fixture-profile");
        assert_eq!(events[0]["metadata"]["speaker_id"], "telegram:user:7");
        for kind in [
            Json::Null,
            json!(true),
            json!(7),
            json!({}),
            json!([]),
            json!("unknown"),
        ] {
            data["message"]["chat"]["type"] = kind;
            let (outcome, events) = execute(
                &program,
                "TelegramBot.process_text_update",
                &input(&data),
                "1",
                "7",
                "",
                vec![],
            );
            assert_eq!(outcome.unwrap()["tag"], "TgIgnored");
            assert!(events.is_empty());
        }
    }
}

#[test]
fn telegram_port_json_kind_numeric_limits_fail_closed_before_io() {
    for program in programs() {
        for number in [
            "9223372036854775808",
            "18446744073709551615",
            "18446744073709551616",
            "-9223372036854775809",
            "1e400",
        ] {
            let raw=format!("{{\"message\":{{\"chat\":{{\"id\":1,\"type\":\"private\"}},\"from\":{{\"id\":{number}}},\"date\":0,\"text\":\"hi\"}}}}");
            let args = format!("{{payload:{}}}", serde_json::to_string(&raw).unwrap());
            let (outcome, events) = execute(
                &program,
                "TelegramBot.process_text_update",
                &args,
                "1",
                "7",
                "",
                vec![],
            );
            assert!(events.is_empty());
            if ["18446744073709551616", "-9223372036854775809"].contains(&number) {
                assert_eq!(outcome.unwrap()["tag"], "TgIgnored");
            } else {
                assert!(outcome.unwrap_err().starts_with("json_kind:"));
            }
        }
        let data = payload(i64::MIN, i64::MAX, "group");
        let (outcome, events) = execute(
            &program,
            "TelegramBot.process_text_update",
            &input(&data),
            &i64::MAX.to_string(),
            "7",
            "",
            vec![],
        );
        assert_eq!(outcome.unwrap()["tag"], "TgNoImmediateReply");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["context"], format!("telegram:group:{}", i64::MIN));
        assert_eq!(
            events[0]["metadata"]["speaker_id"],
            format!("telegram:user:{}", i64::MAX)
        );
    }
}

#[test]
fn telegram_port_text_nested_union_and_coverage_categories() {
    for program in programs() {
        let module = &program.modules[0];
        assert_eq!(
            module.functions.len(),
            37,
            "33 original functions plus text entry and three NEW validators"
        );
        assert!(program.get_function("TelegramBot.process_update").is_some());
        let outcome = module
            .types
            .iter()
            .find_map(|ty| match ty {
                ast::TypeDecl::TaggedUnion(value) if value.name == "TgTextTurn" => Some(value),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            outcome
                .variants
                .iter()
                .map(|v| v.name.as_str())
                .collect::<Vec<_>>(),
            [
                "TgNoImmediateReply",
                "TgDeliveryAttempt",
                "TgIgnored",
                "TgDeferred"
            ]
        );
        // Canonical named payload references retain the parser's Struct spelling;
        // the declaration/catalog resolves this name to the actual union.
        assert_eq!(
            outcome.variants[1].payload,
            [ast::Type::Struct("TgSendAttempt".into())]
        );
        assert!(module.types.iter().any(
            |ty| matches!(ty, ast::TypeDecl::TaggedUnion(value) if value.name=="TgSendAttempt")
        ));
    }
    let manifest: Json = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/telegram-python-port/manifest.json"
    ))
    .unwrap();
    let module = &manifest["modules"][0];
    assert_eq!(module["implemented"].as_array().unwrap().len(), 33);
    assert_eq!(module["pending"].as_array().unwrap().len(), 9);
    assert_eq!(module["introduced_functions"].as_object().unwrap().len(), 4);
    assert_eq!(
        module["normal_shared_contract_functions"],
        json!(SHARED_FUNCTIONS)
    );
    let mut covered = std::collections::BTreeSet::new();
    for category in [
        "direct_vm_paired_functions",
        "normal_shared_contract_functions",
        "normal_io_contract_functions",
        "trusted_result_contract_functions",
    ] {
        for name in module[category].as_array().unwrap() {
            assert!(
                covered.insert(name.as_str().unwrap()),
                "overlapping coverage category"
            );
        }
    }
    assert_eq!(
        covered,
        module["implemented"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect()
    );
    assert_eq!(
        module["new_entry_contract_functions"],
        json!(["process_text_update"])
    );
}

#[test]
fn telegram_port_text_guards_precede_authorization_helpers() {
    for program in programs() {
        let _state = install(&program, "1", "7");
        let stubs = SHARED_FUNCTIONS
            .iter()
            .map(|name| session::FunctionStub {
                function_name: format!("TelegramBot.{name}"),
                patterns: vec![],
                response_expr: "to_int(\"AUTHORIZATION_REACHED\")".into(),
            })
            .collect::<Vec<_>>();
        let valid = payload(1, 1, "private");
        let mut invalid = Vec::new();
        for (field, value) in [
            ("date", json!(false)),
            ("text", json!(true)),
            ("text", Json::Null),
            ("photo", json!({})),
        ] {
            let mut data = valid.clone();
            data["message"][field] = value;
            invalid.push(data);
        }
        let mut wrong_type = valid.clone();
        wrong_type["message"]["chat"]["type"] = json!(true);
        invalid.push(wrong_type);
        for data in invalid {
            let result = eval::eval_async_function(
                &program,
                "TelegramBot.process_text_update",
                &parser::parse_single_expr(&input(&data)).unwrap(),
                CoroutineHandle::new_mock_with_stubs(vec![], stubs.clone()),
            )
            .unwrap();
            let result = observed(result);
            assert!(result["tag"] == "TgIgnored" || result["tag"] == "TgDeferred");
        }
        let error = eval::eval_async_function(
            &program,
            "TelegramBot.process_text_update",
            &parser::parse_single_expr(&input(&valid)).unwrap(),
            CoroutineHandle::new_mock_with_stubs(vec![], stubs),
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("AUTHORIZATION_REACHED"),
            "positive control proves the stubs intercept authorization"
        );
    }
}

#[test]
fn telegram_port_present_empty_media_must_defer_red_gate() {
    let mut failures = Vec::new();
    for (origin, program) in programs().iter().enumerate() {
        let baseline = payload(1, 1, "private");
        let (outcome, events) = execute(
            program,
            "TelegramBot.process_text_update",
            &input(&baseline),
            "1",
            "7",
            "",
            vec![],
        );
        assert_eq!(outcome.unwrap()["tag"], "TgNoImmediateReply");
        assert_eq!(events.len(), 1);
        for key in [
            "photo",
            "video",
            "voice",
            "document",
            "audio",
            "animation",
            "sticker",
            "video_note",
            "contact",
            "location",
            "poll",
        ] {
            let mut present = baseline.clone();
            present["message"][key] = json!("");
            let mut env = eval::Env::new_with_shared_interner(&program.shared_interner);
            let lookup = |data: &Json, env: &mut eval::Env| {
                eval::eval_builtin_or_user(
                    program,
                    "json_get",
                    vec![
                        Value::string(data.to_string()),
                        Value::string(format!("message.{key}")),
                    ],
                    env,
                )
                .unwrap()
            };
            let absent_value = lookup(&baseline, &mut env);
            let present_value = lookup(&present, &mut env);
            assert!(matches!(&absent_value,Value::String(value) if value.is_empty()));
            assert!(
                absent_value.matches(&present_value),
                "accessor erases key presence"
            );
            let (outcome, events) = execute(
                program,
                "TelegramBot.process_text_update",
                &input(&present),
                "1",
                "7",
                "",
                vec![],
            );
            if !events.is_empty()
                || outcome.as_ref().is_err()
                || outcome.as_ref().unwrap()["tag"] != "TgDeferred"
            {
                failures.push(format!(
                    "origin{origin} present-empty {key}: outcome={outcome:?}, IO={}",
                    events.len()
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "presence/type primitive needed; no exemptions:\n{}",
        failures.join("\n")
    );
}
