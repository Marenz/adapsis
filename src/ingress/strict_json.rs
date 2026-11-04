//! Reject duplicate JSON object members before extracting identity-bearing fields.
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::Value;

struct Unique(Value);
struct ValueVisitor;

impl<'de> Visitor<'de> for ValueVisitor {
    type Value = Unique;
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("JSON with unique object members")
    }
    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Unique, E> {
        Ok(Unique(value.into()))
    }
    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Unique, E> {
        Ok(Unique(value.into()))
    }
    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Unique, E> {
        Ok(Unique(value.into()))
    }
    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Unique, E> {
        serde_json::Number::from_f64(value)
            .map(|v| Unique(Value::Number(v)))
            .ok_or_else(|| E::custom("nonfinite JSON"))
    }
    fn visit_str<E: de::Error>(self, value: &str) -> Result<Unique, E> {
        Ok(Unique(value.into()))
    }
    fn visit_string<E: de::Error>(self, value: String) -> Result<Unique, E> {
        Ok(Unique(value.into()))
    }
    fn visit_unit<E: de::Error>(self) -> Result<Unique, E> {
        Ok(Unique(Value::Null))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Unique, A::Error> {
        let mut values = Vec::new();
        while let Some(Unique(value)) = sequence.next_element()? {
            values.push(value);
        }
        Ok(Unique(Value::Array(values)))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Unique, A::Error> {
        let mut values = serde_json::Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(de::Error::custom("duplicate JSON member"));
            }
            let Unique(value) = map.next_value()?;
            values.insert(key, value);
        }
        Ok(Unique(Value::Object(values)))
    }
}

impl<'de> Deserialize<'de> for Unique {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(ValueVisitor)
    }
}

pub(super) fn parse(text: &str) -> Result<Value, super::IngressError> {
    serde_json::from_str::<Unique>(text)
        .map(|value| value.0)
        .map_err(|_| super::IngressError::Protocol)
}
