use serde::{Deserialize, Deserializer, Serialize, Serializer};
use uuid::Uuid;

// The tag is part of the persistent format. UUIDs contain no scope or authority.
macro_rules! identity {
    ($name:ident, $tag:literal) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub(crate) struct $name(Uuid);

        impl $name {
            pub(crate) fn new() -> Self {
                Self(Uuid::new_v4())
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_str(&format!("{}{}", $tag, self.0.hyphenated()))
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let text = String::deserialize(deserializer)?;
                let invalid = || serde::de::Error::custom(concat!("invalid ", $tag, "identity"));
                let raw = text.strip_prefix($tag).ok_or_else(invalid)?;
                let uuid = Uuid::parse_str(raw).map_err(|_| invalid())?;
                if uuid.get_version() != Some(uuid::Version::Random)
                    || uuid.get_variant() != uuid::Variant::RFC4122
                    || uuid.hyphenated().to_string() != raw
                {
                    return Err(invalid());
                }
                Ok(Self(uuid))
            }
        }
    };
}

identity!(WorkspaceId, "ws_");
identity!(ProjectId, "project_");
identity!(DurableTaskId, "task_");
identity!(TaskRunId, "run_");
