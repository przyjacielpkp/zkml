// THIS-WORKS

// 
// Let's refer to the source ml graph as defined by the ml model as "ml graph",
// to the scalarized version of the source graph as "scalar graph",
// and to the graph that defines zk circuit - "zk graph".
// 
// 
// 
// luminal's 'Graph' type : dag representing a computation. more or less `dyn Fn(Tensors, Shapes) -> Tensors` in node weights. Tracks dimensions.
// luminal's 'SelectGraph' type : graph variant helding more symbolic representation of operator types at nodes. used for pattern matching on 'Graph'
// 
// Q: We have options: 
//   - define separate new types for steps of our transformation, 
//   - or copy 'Graph' overwriting node weight type,
//   - or store 'Graph' alongside a matching NodeIndex -> <our weight held in the given node>

// problems (computations we cannot support due to some mismatch between ml computation and what is possible with zk):
//  - dynamic tensor dimensions (probably, maybe doable with zk private inputs certifying tensor dimensions)
//  - activation functions of: sin, tan, log2, exp2 (what about sqrt and reciprocal?)

// The operator types of graph nodes in source graph:
// 
//  - Function /* Fn(Vec<Tensor, Shape>) -> Tensor */
//  - Constant(pub ConstantValue /* see below */ , pub *const FxHashMap<char, usize> /* variable assignments in expression */);
//  - Contiguous /* assert tensor held contiguously in memory */
//  - Log2
//  - Exp2
//  - Sin
//  - Recip
//  - Sqrt
//  - Add
//  - Mul
//  - Mod
//  - LessThan
//  - MaxReduce(pub usize /* which dimension to reduce */)
//  - SumReduce(pub usize)
// 
// where:
// 
// pub enum ConstantValue {
//   Expression(BigExpression),
//   Float(f32),
