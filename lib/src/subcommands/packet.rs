use ark_groth16::Proof;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{Deserialize, Serialize};

pub struct SerializationError {}

use crate::snark::{Fp, Pairing};

#[derive(Deserialize, Serialize)]
pub(crate) struct Packet {
  serialized_proof: Vec<u8>,
  serialized_public_inputs: Vec<Vec<u8>>,
}

pub fn pack(proof: Proof<Pairing>, public_inputs: Vec<Fp>) -> String {
  let mut packet = Packet {
    serialized_proof: vec![],
    serialized_public_inputs: vec![],
  };

  proof.serialize(&mut packet.serialized_proof).unwrap();
  packet.serialized_public_inputs = public_inputs
    .iter()
    .map(|val| {
      let mut buff: Vec<u8> = vec![];
      val.serialize(&mut buff).unwrap();
      buff
    })
    .collect();

  serde_json::to_string(&packet).unwrap()
}

pub fn unpack(input: &str) -> Result<(Proof<Pairing>, Vec<Fp>), SerializationError> {
  let packet: Packet = serde_json::from_str(input).map_err(|_| SerializationError {})?;
  let proof = Proof::<Pairing>::deserialize(packet.serialized_proof.as_slice())
    .map_err(|_| SerializationError {})?;
  let public_inputs = packet
    .serialized_public_inputs
    .iter()
    .map(|bytes| Fp::deserialize(bytes.as_slice()).map_err(|_| SerializationError {}))
    .collect::<Result<Vec<Fp>, SerializationError>>()?;
  Ok((proof, public_inputs))
}
