use std::{collections::HashMap, fmt::Debug};

use itertools::Itertools;
/// 
/// Produce snark from the computation after scalar and integer transformations.
///

use luminal::{op::Add, prelude::{petgraph::{self, visit::EdgeRef, Direction::Incoming}, NodeIndex}};
use ark_ff::Field;
use ark_relations::{
    lc,
    r1cs::{ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, SynthesisError},
};
use tracing::{info, instrument};

use crate::scalar::ScalarGraph;

#[derive(Debug)]
pub enum SourceType<F> {
  Private(Option<F>),
  Public(F)
}

#[derive(Debug)]
pub struct MLSnark<F> {
  pub graph : ScalarGraph,
  pub source_nodes_map : HashMap<NodeIndex, SourceType<F>>
}

impl<F: Field> ConstraintSynthesizer<F> for MLSnark<F> {

    // THIS-WORKS

    #[instrument(level = "debug", name = "generate_constraints")]
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
      
      let graph = &self.graph.graph;
      let source_map = &self.source_nodes_map;

      let pi = petgraph::algo::toposort(&graph.graph, None).unwrap();
      let mut vars: HashMap<NodeIndex, ark_relations::r1cs::Variable> = HashMap::new();
      // would actaully want:
      // let mut assignments: HashMap<NodeIndex, Box<dyn Fn()-> Result<F, SynthesisError>> > = HashMap::new();
      // but the below is easier to manage ownership with
      let mut assignments: HashMap<NodeIndex, Option<F> > = HashMap::new();
      // ^ thats silly that we need to track assignments but thats really because of the low level nature of arkworks api

      for x in pi {
        info!("x={:?}", x);
        
        let incoming: Vec<_> = graph.edges_directed(x, Incoming)
          .filter_map(|e| e.weight().as_data().map(|d| (d, e.source())))
          .sorted_by_key(|((inp,_,_),_)| *inp )
          .collect();
        
        // x is source
        if incoming.is_empty() {
          
          let src_ty = source_map.get(&x).unwrap_or_else(|| panic!("No source for node {:?}", x));
          use SourceType::*;
          let (v, ass) = match src_ty {
            Private(mn) =>
              (cs.new_witness_variable(|| mn.ok_or(SynthesisError::AssignmentMissing))?, mn.clone())
            ,
            Public(n) =>
              (cs.new_input_variable(|| Ok(*n))?, Some(*n))
          };
          vars.insert(x, v);
          assignments.insert(x, ass);
        }
        else if let Some((((_, _, _), x), )) = incoming.iter().collect_tuple() {
          todo!("Unop")
        }
        // x is binop
        else if let Some(((_, l), (_, r))) = incoming.into_iter().collect_tuple() {
          // assumes toposort order for unwraps
          let ll = vars.get(&l).unwrap().clone(); 
          let rr = vars.get(&r).unwrap().clone();
          let ll_val = assignments.get(&l).unwrap().clone();
          let rr_val = assignments.get(&r).unwrap().clone();

          let (v, ass) = if graph.check_node_type::<Add>(x){
              // how nice would it be to do: (,) <*> ll_val <$> rr_val 
              let ass = ll_val
                .and_then(|l| rr_val.map(|r| (l, r)))
                .map(|(l, r)| l + r );
              let v = cs.new_witness_variable(|| ass.ok_or( SynthesisError::AssignmentMissing ))?;
              cs.enforce_constraint(lc!() + ll + rr, lc!() + ConstraintSystem::<F>::one(), lc!() + v)?; // ll + rr s== v
              (v, ass)
            } else {
              todo!("Unsupported binops other than Add")
            };
          vars.insert(x, v);
          assignments.insert(x, ass);
        }
      }
      Ok(())
    }
}

#[cfg(test)]
mod test {
    // use super::*;
    // use ark_bls12_381::{Bls12_381, Fr as BlsFr};
    // use ark_groth16::Groth16;
    // use ark_snark::SNARK;
// 
    // #[test]
    // fn test_groth16_circuit_cubic() {
    //     let rng = &mut ark_std::test_rng();

    //     // generate the setup parameters
    //     let (pk, vk) = Groth16::<Bls12_381>::circuit_specific_setup(
    //         CubicDemoCircuit::<BlsFr> { x: None },
    //         rng,
    //     )
    //     .unwrap();

    //     // calculate the proof by passing witness variable value
    //     let proof1 = Groth16::<Bls12_381>::prove(
    //         &pk,
    //         CubicDemoCircuit::<BlsFr> {
    //             x: Some(BlsFr::from(3)),
    //         },
    //         rng,
    //     )
    //     .unwrap();

    //     // validate the proof
    //     assert!(Groth16::<Bls12_381>::verify(&vk, &[BlsFr::from(35)], &proof1).unwrap());

    //     // calculate the proof by passing witness variable value
    //     let proof2 = Groth16::<Bls12_381>::prove(
    //         &pk,
    //         CubicDemoCircuit::<BlsFr> {
    //             x: Some(BlsFr::from(4)),
    //         },
    //         rng,
    //     )
    //     .unwrap();
    //     assert!(Groth16::<Bls12_381>::verify(&vk, &[BlsFr::from(73)], &proof2).unwrap());

    //     assert!(!Groth16::<Bls12_381>::verify(&vk, &[BlsFr::from(35)], &proof2).unwrap());
    //     assert!(!Groth16::<Bls12_381>::verify(&vk, &[BlsFr::from(73)], &proof1).unwrap());
    // }
}