use std::{collections::HashMap, path::PathBuf};

use ark_groth16::{Groth16, ProvingKey};
use ark_serialize::CanonicalSerialize;
use ark_snark::SNARK;
use luminal::{graph::Graph, prelude::*};
use luminal_training::{mse_loss, sgd_on_graph, Autograd};
use rand::{rngs::StdRng, SeedableRng};

use crate::{model::Model, snark::MLSnark};

pub struct Client {
  input: String,
  proving_key: ProvingKey<crate::snark::Curve>,
  weights: HashMap<NodeIndex, Vec<f32>>,
  url: String,
}

impl Client {
  pub fn new(
    input_path_buf: PathBuf,
    proving_key_path_buf: PathBuf,
    weights_path_buf: PathBuf,
    url: String,
  ) -> Self {
    // TODO: parse input properly
    let input = crate::utils::deserialize_from_file(&input_path_buf);
    let proving_key = crate::utils::canonical_deserialize_from_file(&proving_key_path_buf);
    let weights = crate::utils::deserialize_from_file::<HashMap<u32, Vec<f32>>>(&weights_path_buf)
      .iter()
      .map(|(key, val)| (NodeIndex::from(key.clone()), val.clone()))
      .collect();

    Self {
      input,
      proving_key,
      weights,
      url,
    }
  }

  pub async fn run(self) {
    let client = reqwest::Client::new();

    let mut rng = StdRng::seed_from_u64(1);

    let mut cx = Graph::new();
    let model = <Model>::initialize(&mut cx);
    let mut input = cx.tensor::<R1<9>>();
    let mut target = cx.tensor::<R1<1>>();
    let mut output = model.forward(input).retrieve();
    let mut loss = mse_loss(output, target).retrieve();

    let mut weights = params(&model);
    let compiler = Autograd::new(&weights, loss);
    let grads = cx.compile(compiler, ());
    let (mut new_weights, lr) = sgd_on_graph(&mut cx, &weights, &grads);
    cx.keep_tensors(&new_weights);
    cx.keep_tensors(&weights);
    lr.set(5e-3);

    #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
    cx.compile(
      GenericCompiler::default(),
      (
        &mut input,
        &mut target,
        &mut loss,
        &mut output,
        &mut weights,
        &mut new_weights,
      ),
    );
    for (key, val) in self.weights {
      cx.tensors.insert((key, 0), Tensor::new(val));
    }
    // TODO: replace ... with proper object, so that weights will be treated as the public input,
    // then, the following lines can be uncommented
    // self.weights is a map of weights,
    // self.input should be an input vector
    // cx is parsed neural network
    /*let circuit= crate::lib::compile(...);

        let proof = Groth16::prove(&self.proving_key, circuit, &mut rng).unwrap();
    */
    let mut buff = Vec::<u8>::new();
    //proof.serialize(&mut buff);

    let response = match client.get(self.url).body(buff).send().await {
      Ok(response) => response,
      Err(err) => panic!("{}", err),
    };
    let res = response.text().await.unwrap();
    println!("{}", res);
  }
}
